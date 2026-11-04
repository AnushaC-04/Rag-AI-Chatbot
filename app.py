import os
import base64
import hashlib
import tempfile
from langchain_core.documents import Document
import streamlit as st

try:
    from langchain_core.documents import Document
except ImportError:  # older LangChain versions
    from langchain.schema import Document

from document_loader import load_pdf
from chunker import split_documents
from vectorstore import (
    get_collection,
    create_vector_store,
    get_uploaded_documents,
    delete_document,
)
from rag_pipeline import RAGPipeline

# PAGE CONFIG

st.set_page_config(
    page_title="DocuMind | AI Document Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# THEME  (palette taken from the reference image)

st.markdown(
    """
    <style>
    :root {
        --bg: #02070f;
        --panel: #07111f;
        --panel-2: #0b1a2e;
        --cyan: #1fb6ff;
        --blue: #3a8bff;
        --purple: #b04cf2;
        --text: #f4f8ff;
        --muted: #a9b8d0;
        --line: rgba(31, 182, 255, .35);
        --grad: linear-gradient(90deg, #1ea7fd 0%, #6a7cf6 55%, #b55bea 100%);
    }

    /* ---------- base ---------- */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background:
            radial-gradient(ellipse at 50% 0%, rgba(31,182,255,.16), transparent 55%),
            radial-gradient(ellipse at 85% 100%, rgba(176,76,242,.10), transparent 45%),
            #02070f !important;
        color: var(--text);
    }
    [data-testid="stHeader"], [data-testid="stToolbar"], footer { display: none !important; }
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] { display: none !important; }

    .block-container {
        max-width: 980px;
        padding-top: 1.2rem;
        padding-bottom: 7rem;
    }
    h1, h2, h3, p, li, label, span, div { color: var(--text); }

    /* ---------- hero ---------- */
    .hero {
        display: flex; flex-direction: column; align-items: center;
        text-align: center; padding: 8px 0 6px;
    }
    .robot-wrap { position: relative; display: flex; justify-content: center; }
    .robot { animation: float 4s ease-in-out infinite;
             filter: drop-shadow(0 0 26px rgba(31,182,255,.45)); }
    .robot img, .robot svg { display: block; }
    @keyframes float { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-9px); } }

    .arc {
        width: 760px; max-width: 100%; height: 120px; margin-top: -34px;
        border-radius: 50% 50% 0 0 / 100% 100% 0 0;
        border-top: 2px solid rgba(31,182,255,.65);
        background: radial-gradient(ellipse at 50% 0%, rgba(31,182,255,.28), transparent 70%);
        box-shadow: 0 -10px 40px rgba(31,182,255,.25);
    }
    .hero-title {
        font-size: clamp(30px, 4.4vw, 50px); font-weight: 800;
        line-height: 1.12; letter-spacing: -1px; margin-top: -92px;
    }
    .hero-title .white { color: #fff; display: block; }
    .hero-title .grad {
        display: block;
        background: linear-gradient(90deg, #1ea7fd, #4a8dff 55%, #8f6bf5);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero-sub {
        color: var(--muted) !important; font-size: 16px; line-height: 1.6;
        max-width: 560px; margin: 14px auto 8px;
    }

    /* ---------- compact header (after first question) ---------- */
    .compact { display: flex; align-items: center; gap: 14px; padding: 4px 0 8px; }
    .compact .brand {
        font-size: 24px; font-weight: 800; line-height: 1.1;
        background: var(--grad); -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .compact .doc { color: var(--muted) !important; font-size: 13px; margin-top: 2px; }

    .brandbar { display:flex; align-items:center; gap:10px; margin-bottom: 4px; }
    .brandbar .brand {
        font-size: 24px; font-weight: 800;
        background: var(--grad); -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* ---------- buttons ---------- */
    .stButton > button, [data-testid="stBaseButton-primary"] {
        color: #fff !important; font-weight: 650 !important; min-height: 44px;
        border: none !important; border-radius: 999px !important;
        background: var(--grad) !important;
        box-shadow: 0 4px 18px rgba(31,182,255,.25);
    }
    .stButton > button:hover { box-shadow: 0 4px 24px rgba(176,76,242,.45); }
    [data-testid="stBaseButton-secondary"] {
        background: transparent !important; color: var(--text) !important;
        border: 1px solid rgba(169,184,208,.35) !important; box-shadow: none !important;
    }
    [data-testid="stBaseButton-secondary"]:hover { border-color: var(--cyan) !important; }

    /* ---------- tabs ---------- */
    [data-baseweb="tab-list"] { gap: 8px; justify-content: center; border-bottom: none; }
    button[role="tab"] {
        border: 1px solid rgba(169,184,208,.25) !important; border-radius: 999px !important;
        padding: 8px 22px !important; background: rgba(255,255,255,.02) !important;
    }
    button[role="tab"] p { color: var(--muted) !important; font-weight: 600; }
    button[role="tab"][aria-selected="true"] {
        border-color: var(--cyan) !important;
        background: linear-gradient(90deg, rgba(30,167,253,.25), rgba(181,91,234,.25)) !important;
    }
    button[role="tab"][aria-selected="true"] p { color: #fff !important; }
    [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none !important; }

    /* ---------- inputs ---------- */
    [data-testid="stFileUploaderDropzone"] {
        background: var(--panel-2); border: 1px dashed var(--line); border-radius: 16px;
    }
    [data-testid="stFileUploaderDropzone"] * { color: var(--text) !important; }
    [data-testid="stFileUploader"] small { color: var(--muted) !important; }

    .stTextInput input, .stTextArea textarea, [data-baseweb="select"] > div {
        background: var(--panel-2) !important; color: #fff !important;
        border: 1px solid var(--line) !important; border-radius: 14px !important;
    }
    .stTextInput input::placeholder, .stTextArea textarea::placeholder { color: #7f90ab !important; }
    .stTextInput label, .stTextArea label, .stSelectbox label { color: var(--muted) !important; }

    /* ---------- chat ---------- */
    [data-testid="stChatMessage"] {
        background: var(--panel-2); border: 1px solid rgba(31,182,255,.20);
        border-radius: 18px; padding: 14px 18px; margin-bottom: 14px;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: linear-gradient(100deg, rgba(30,167,253,.28), rgba(181,91,234,.28));
        border-color: rgba(176,76,242,.45);
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div {
        color: #ffffff !important; font-size: 16px; line-height: 1.7;
    }
    [data-testid="stChatMessage"] code { color: #7fe3ff !important; background: rgba(0,0,0,.35); }

    [data-testid="stExpander"] {
        background: rgba(2,7,15,.55); border: 1px solid rgba(31,182,255,.25); border-radius: 12px;
    }
    [data-testid="stExpander"] summary p { color: var(--cyan) !important; font-weight: 600; }

    /* bottom input bar */
    [data-testid="stBottom"], [data-testid="stBottom"] > div { background: transparent !important; }
    [data-testid="stChatInput"] {
        background: #0b1a2e !important; border: 1.5px solid var(--cyan) !important;
        border-radius: 999px !important; box-shadow: 0 0 22px rgba(31,182,255,.25);
    }
    [data-testid="stChatInput"] textarea {
        color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;
        font-size: 16px !important; caret-color: var(--cyan);
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #9fb2cf !important; -webkit-text-fill-color: #9fb2cf !important; opacity: 1;
    }

    [data-testid="stAlert"] { border-radius: 12px; }
    [data-testid="stSpinner"] * { color: var(--text) !important; }
    hr { border-color: rgba(31,182,255,.2) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# HELPERS

ROBOT_SVG = """
<svg width="{s}" height="{s}" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="hd" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#6fa8ff"/><stop offset="0.55" stop-color="#b66cf0"/>
      <stop offset="1" stop-color="#e86fd0"/>
    </linearGradient>
    <linearGradient id="bd" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#4fc3ff"/><stop offset="1" stop-color="#1e6bff"/>
    </linearGradient>
  </defs>
  <line x1="100" y1="14" x2="100" y2="34" stroke="#b66cf0" stroke-width="4"/>
  <circle cx="100" cy="12" r="6" fill="#4fc3ff"/>
  <path d="M28 128 Q10 112 26 100 Q50 108 66 124 Z" fill="#7aa2ff"/>
  <path d="M172 128 Q190 112 174 100 Q150 108 134 124 Z" fill="#7aa2ff"/>
  <rect x="62" y="116" width="76" height="62" rx="30" fill="url(#bd)"/>
  <rect x="34" y="34" width="132" height="92" rx="44" fill="url(#hd)"/>
  <rect x="48" y="52" width="104" height="52" rx="26" fill="#0a1020"/>
  <circle cx="78" cy="78" r="13" fill="#fff"/><circle cx="122" cy="78" r="13" fill="#fff"/>
  <circle cx="80" cy="79" r="6" fill="#0a1020"/><circle cx="124" cy="79" r="6" fill="#0a1020"/>
  <path d="M90 94 Q100 104 110 94 Z" fill="#ff7ac8"/>
  <path d="M52 48 Q70 36 96 38" stroke="rgba(255,255,255,.55)" stroke-width="4"
        fill="none" stroke-linecap="round"/>
</svg>
"""


@st.cache_data(show_spinner=False)
def _robot_b64():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot.png")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


def robot_html(size: int) -> str:
    """Anime robot: uses robot.png if present, else a built-in SVG."""
    b64 = _robot_b64()
    if b64:
        inner = (
            f'<img src="data:image/png;base64,{b64}" '
            f'style="width:{size}px;height:auto;max-width:100%;border-radius:16px">'
        )
    else:
        inner=ROBOT_SVG.format(s=size)
    return f'<div class="robot">{inner}</div>'


def index_documents(documents, document_id, document_name):
    chunks=split_documents(documents)
    if not chunks:
        return 0
    create_vector_store(
        chunks=chunks,document_id=document_id,document_name=document_name
    )
    st.session_state.selected_document = document_id
    st.session_state.chat_history.setdefault(document_id, [])
    st.session_state.chat_started = False
    return len(chunks)

def process_pdf(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    document_id = hashlib.sha256(file_bytes).hexdigest()[:20]
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            temp_path = tmp.name
        documents = load_pdf(temp_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    if not documents:
        raise ValueError("No text could be extracted from this PDF.")
    return index_documents(documents, document_id, uploaded_file.name)

def process_text(title, text):
    document_id = hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
    docs = [Document(page_content=text, metadata={"page": 0, "source": title})]
    return index_documents(docs, document_id, title)

# INIT

@st.cache_resource
def initialize_pipeline():
    return RAGPipeline(get_collection())


try:
    rag = initialize_pipeline()
except Exception as exc:
    st.error(f"Could not initialize the RAG system: {exc}")
    st.stop()

st.session_state.setdefault("chat_history", {})
st.session_state.setdefault("selected_document", None)
st.session_state.setdefault("chat_started", False)

available_documents = get_uploaded_documents()
selected_document = st.session_state.selected_document
has_document = bool(available_documents) and selected_document in available_documents

messages = (
    st.session_state.chat_history.setdefault(selected_document, [])
    if has_document
    else []
)
chat_started = has_document and (bool(messages) or st.session_state.chat_started)


# TOP BAR  (compact header once the chat has started)

if chat_started:
    c1, c2, c3 = st.columns([6, 1.3, 1.1], vertical_alignment="center")
    with c1:
        st.markdown(
            f'<div class="compact">{robot_html(72)}'
            f'<div><div class="brand">DocuMind</div>'
            f'<div class="doc">📄 {available_documents[selected_document]}</div></div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        if st.button("＋ New", type="secondary", use_container_width=True):
            st.session_state.selected_document = None
            st.session_state.chat_started = False
            st.rerun()
    with c3:
        if st.button("🗑 Delete", type="secondary", use_container_width=True):
            delete_document(selected_document)
            st.session_state.chat_history.pop(selected_document, None)
            st.session_state.selected_document = None
            st.session_state.chat_started = False
            st.rerun()


# HERO  (before the user uploads / asks anything)

else:
    if has_document:
        title_html = (
            '<span class="white">Your document is ready</span>'
            '<span class="grad">Ask DocuMind Anything</span>'
        )
        sub = f"Chatting with <b>{available_documents[selected_document]}</b>. Type your question below."
    else:
        title_html = (
            '<span class="white">The Future of Documents</span>'
            '<span class="grad">AI-Powered Q&amp;A Assistant</span>'
        )
        sub = (
            "Meet DocuMind. Upload a PDF or type your project, "
            "then ask questions in plain language."
        )

    st.markdown(
        f'<div class="hero"><div class="robot-wrap">{robot_html(230)}</div>'
        f'<div class="arc"></div>'
        f'<div class="hero-title">{title_html}</div>'
        f'<div class="hero-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )

    if has_document:
        _, mid, _ = st.columns([1.3, 1, 1.3])
        with mid:
            if st.button("＋ New document", type="secondary", use_container_width=True):
                st.session_state.selected_document = None
                st.rerun()

    # ---------- two input tabs ----------
    if not has_document:
        _, center, _ = st.columns([0.6, 3, 0.6])
        with center:
            tab_upload, tab_type = st.tabs(["📄 Upload Document", "✍️ Type Project"])

            with tab_upload:
                uploaded_files = st.file_uploader(
                    "Upload PDF(s)",
                    type=["pdf"],
                    accept_multiple_files=True,
                    label_visibility="collapsed",
                )
                if st.button("Process Documents", use_container_width=True, key="go_pdf"):
                    if not uploaded_files:
                        st.warning("Please upload at least one PDF.")
                    else:
                        done = False
                        for f in uploaded_files:
                            try:
                                with st.spinner(f"Processing {f.name}..."):
                                    process_pdf(f)
                                done = True
                            except Exception as exc:
                                st.error(f"{f.name}: {exc}")
                        if done:
                            st.rerun()

            with tab_type:
                project_title = st.text_input(
                    "Project name", placeholder="e.g. Smart Attendance System"
                )
                project_text = st.text_area(
                    "Project details",
                    placeholder="Type or paste your project description here...",
                    height=190,
                )
                if st.button("Save Project", use_container_width=True, key="go_text"):
                    if not project_text.strip():
                        st.warning("Please type some project details first.")
                    else:
                        try:
                            with st.spinner("Processing your project..."):
                                process_text(
                                    project_title.strip() or "Typed project",
                                    project_text.strip(),
                                )
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Could not process the text: {exc}")

            if available_documents:
                st.markdown("&nbsp;", unsafe_allow_html=True)
                chosen = st.selectbox(
                    "Or continue with a saved document",
                    options=list(available_documents.keys()),
                    format_func=lambda d: available_documents[d],
                    index=None,
                    placeholder="Choose a document...",
                )
                if chosen:
                    st.session_state.selected_document = chosen
                    st.session_state.chat_started = bool(
                        st.session_state.chat_history.get(chosen)
                    )
                    st.rerun()


# CHAT

if has_document:
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📖 Sources"):
                    for i, source in enumerate(message["sources"], start=1):
                        page = source.get("metadata", {}).get("page")
                        page_no = page + 1 if isinstance(page, int) else "?"
                        st.markdown(f"**Source {i} · Page {page_no}**")
                        st.write(source.get("content", ""))

    query = st.chat_input("Ask a question about your document...")

    if query:
        st.session_state.chat_started = True
        messages.append({"role": "user", "content": query})

        try:
            with st.spinner("Searching your document..."):
                result = rag.ask(query=query, document_id=selected_document, top_k=3)
            messages.append(
                {
                    "role": "assistant",
                    "content": result.get(
                        "answer", "I couldn't generate an answer from the document."
                    ),
                    "sources": result.get("sources", []),
                }
            )
        except Exception as exc:
            messages.append(
                {
                    "role": "assistant",
                    "content": f"Something went wrong: {exc}",
                    "sources": [],
                }
            )

        st.rerun()
