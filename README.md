# 🤖 RAG AI Chatbot

An AI-powered **Retrieval-Augmented Generation (RAG) Chatbot** that allows users to interact with documents and retrieve contextually relevant answers using Large Language Models (LLMs).

The application combines document processing, text chunking, embeddings, vector search, and Google's Gemini API to generate answers grounded in the uploaded document content.

## 🚀 Project Overview
 
Traditional chatbots generate answers from their pretrained knowledge. This project uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from documents before generating a response.

Users can provide documents and ask questions in natural language. The system retrieves relevant document chunks and passes the retrieved context to the LLM to generate a context-aware response.

## ✨ Key Features

- **Document Loading:** Extracts text from PDF documents.
- **Text Chunking:** Splits large documents into smaller, manageable chunks.
- **Text Embeddings:** Converts text chunks into numerical vector representations.
- **Vector Database:** Uses ChromaDB to store and retrieve document embeddings.
- **Semantic Search:** Retrieves relevant document content based on user queries.
- **AI-Powered Responses:** Uses Google's Gemini API to generate contextual answers.
- **RAG Pipeline:** Integrates document processing, retrieval, and response generation.
- **Question Answering:** Enables natural-language interaction with document content.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Google Gemini API | LLM-based response generation |
| ChromaDB | Vector database |
| Embeddings | Semantic representation of text |
| RAG | Retrieval-Augmented Generation |
| PDF Processing | Document text extraction |
| Git & GitHub | Version control |

## 🏗️ System Architecture

```text
          PDF Documents
                |
                v
        Document Loader
                |
                v
          Text Extraction
                |
                v
           Text Chunking
                |
                v
          Embedding Model
                |
                v
           ChromaDB
                |
                |
       User Enters a Query
                |
                v
         Query Embedding
                |
                v
        Semantic Retrieval
                |
                v
       Relevant Document Chunks
                |
                v
          Gemini LLM API
                |
                v
       Contextual AI Response
```

## 📂 Project Structure

```text
Rag-AI-Chatbot/
│
├── assets/
│   └── robot_mascot.png
│
├── data/
│   └── sample.pdf
│
├── app.py
├── main.py
├── chunker.py
├── document_loader.py
├── embeddings.py
├── generator.py
├── rag_pipeline.py
├── retriever.py
├── vectorstore.py
├── content.txt
├── robot.png
├── requirements.txt
└── .gitignore
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/AnushaC-04/Rag-AI-Chatbot.git
```

### 2. Navigate to the project directory

```bash
cd Rag-AI-Chatbot
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Key

Create a `.env` file in the project root directory and add your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Replace the placeholder with your actual API key.

**Important:** Never upload your API key or `.env` file to GitHub.

### 5. Run the application

Run the appropriate application entry point:

```bash
python app.py
```

## 🔄 How It Works

1. **Document Ingestion:** The system loads the input PDF and extracts its text.
2. **Text Chunking:** Extracted text is divided into smaller chunks for efficient processing.
3. **Embedding Generation:** Each chunk is converted into a vector representation.
4. **Vector Storage:** Embeddings are stored in ChromaDB.
5. **Query Processing:** The user's question is converted into an embedding.
6. **Context Retrieval:** The vector database retrieves relevant document chunks.
7. **Response Generation:** Retrieved context is provided to Gemini, which generates a natural-language answer.

## 🎯 Learning Outcomes

Through this project, I gained practical experience in:

- Understanding Retrieval-Augmented Generation architecture.
- Working with Large Language Models and API integration.
- Document loading and text preprocessing.
- Text chunking and embedding generation.
- Vector databases and semantic search.
- Building modular Python applications.
- Integrating retrieval and generation into an AI pipeline.

## 🔮 Future Enhancements

- Support for multiple document formats.
- Improved document retrieval and ranking.
- Conversation history and memory.
- Source citations for generated answers.
- Enhanced chatbot interface.
- Deployment as a web application.

## 👩‍💻 Developer

**Anusha C**

B.Tech Information Technology | 2026 Graduate

- GitHub: [AnushaC-04](https://github.com/AnushaC-04)
- Project Repository: [RAG AI Chatbot](https://github.com/AnushaC-04/Rag-AI-Chatbot)

---

⭐ If you find this project interesting, feel free to explore the repository!
