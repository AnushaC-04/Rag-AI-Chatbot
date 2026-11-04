import chromadb
from embeddings import GeminiEmbedding
CHROMA_PATH="./chroma_db"
COLLECTION_NAME="rag_documents"
def get_collection():
    client=chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(name=COLLECTION_NAME)
def create_vector_store(chunks, document_id, document_name="Uploaded document"):
    if not chunks:
        raise ValueError("No chunks provided.")
    collection=get_collection()
    documents=[chunk.page_content for chunk in chunks]
    metadatas=[
        {
            **chunk.metadata,
            "document_id": document_id,
            "document_name": document_name
        }
        for chunk in chunks
    ]
    ids=[
        f"{document_id}_chunk_{i}"
        for i in range(len(chunks))
    ]
    embedding_model=GeminiEmbedding()
    embeddings=embedding_model.embed_documents(documents)
    # Get old chunks for this document
    existing=collection.get(
        where={"document_id": document_id}
    )
    existing_ids=set(existing["ids"])
    new_ids=set(ids)
    # Save current chunks
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )
    # Remove outdated chunks from this same document
    stale_ids=list(existing_ids - new_ids)
    if stale_ids:
        collection.delete(ids=stale_ids)
    return collection

def get_uploaded_documents():
    collection=get_collection()

    records=collection.get(
        include=["metadatas"]
    )
    uploaded_documents={}
    for metadata in records.get("metadatas", []):
        if not metadata:
            continue
        document_id=metadata.get("document_id")
        document_name=metadata.get("document_name", "Uploaded document")
        if document_id:
            uploaded_documents[document_id]=document_name
    return uploaded_documents

def delete_document(document_id):
    collection=get_collection()
    collection.delete(
        where={"document_id": document_id}
    )
