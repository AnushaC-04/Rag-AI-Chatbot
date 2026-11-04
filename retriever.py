from embeddings import GeminiEmbedding
class DocumentRetriever:
    def __init__(self,collection,top_k=3):
        self.collection=collection
        self.top_k=top_k
        self.embedding_model=GeminiEmbedding()
    def retrieve(self,query,document_id=None,top_k=None):
        query_embedding=self.embedding_model.embed_query(query)
        search_args={
            "query_embeddings": [query_embedding],
            "n_results": top_k or self.top_k,
            "include": [
                "documents",
                "metadatas",
                "distances"
            ]
        }
        if document_id:
            search_args["where"]={
                "document_id": document_id
            }
        results=self.collection.query(**search_args)
        documents=results.get("documents",[[]])[0]
        metadatas=results.get("metadatas",[[]])[0]
        distances=results.get("distances",[[]])[0]
        retrieved_chunks=[]
        for document,metadata,distance in zip(
            documents,
            metadatas,
            distances
        ):
            retrieved_chunks.append({
                "content": document,
                "metadata": metadata or {},
                "distance": distance
            })
        return retrieved_chunks
