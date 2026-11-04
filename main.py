from vectorstore import create_vector_store
from rag_pipeline import RAGPipeline
from document_loader import load_pdf
from chunker import split_documents
import hashlib

def main():

    # 1. Get PDF path
    file_path=input("Enter PDF path: ").strip()
    if not file_path:
        print("PDF path cannot be empty.")
        return

    # 2. Load PDF
    documents=load_pdf(file_path)

    if not documents:
        print("Failed to load PDF")
        return

    print(f"\nLoaded {len(documents)} pages")

    # 3. Split documents
    chunks=split_documents(documents)

    if not chunks:
        print("Failed to create chunks")
        return
    print(f"Created {len(chunks)} chunks")

    # 4. Store embeddings
    document_id=hashlib.sha256(
    file_path.encode("utf-8")).hexdigest()[:16]
    collection=create_vector_store(chunks,
    document_id=document_id)

    # 5. Create RAG pipeline
    rag=RAGPipeline(collection)

    print("\nRAG chatbot is ready!")
    print("Type 'exit' to stop.\n")

    # 6. Chat loop
    while True:
        query=input("You: ").strip()
        if query.lower()=="exit":
            print("Goodbye!")
            break
        if not query:
            continue

        try:
            result=rag.ask(query=query,top_k=3)
            print("\nAssistant:")
            print(result["answer"])
            print("\nRetrieved sources:")
            for index, source in enumerate(result["sources"],start=1):
                metadata=source["metadata"]
                print(
                    f"  {index}. "
                    f"Page: {metadata.get('page', 'N/A')} "
                    f"| Distance: "
                    f"{source['distance']:.4f}"
                )
            print()
        except Exception as e:
            print(f"\nError processing question: {e}\n")

if __name__=="__main__":
    main()
