from retriever import DocumentRetriever
from generator import GeminiGenerator

class RAGPipeline:
    def __init__(self, collection):
        self.collection=collection
        self.retriever=DocumentRetriever(
            collection=collection,
            top_k=3
        )
        self.generator=GeminiGenerator()
    def ask(self, query, document_id=None, top_k=3):

        if not query or not query.strip():
            return {
                "answer": "Please enter a question.",
                "sources": []
            }
        retrieved_chunks=self.retriever.retrieve(
            query=query,
            document_id=document_id,
            top_k=top_k
        )
        if not retrieved_chunks:
            return {
                "answer": "I couldn't find relevant information in the selected document.",
                "sources": []
            }
        answer=self.generator.generate_answer(
            query=query,
            retrieved_chunks=retrieved_chunks
        )
        return {
            "answer": answer,
            "sources": retrieved_chunks
        }
