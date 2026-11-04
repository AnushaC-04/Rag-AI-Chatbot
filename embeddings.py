import os
from google import genai
from dotenv import load_dotenv
from google.genai import types

load_dotenv()

class GeminiEmbedding:
    """Generate Gemini embeddings for documents and queries."""

    def __init__(self):
        api_key=os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set in the .env file."
            )
        self.client=genai.Client(api_key=api_key)
        self.model="gemini-embedding-001"

    def embed_documents(self, texts):
        response=self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )

        return [
            embedding.values
            for embedding in response.embeddings
        ]

    def embed_query(self, query):
        response=self.client.models.embed_content(
            model=self.model,
            contents=query,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY")
        )
        return response.embeddings[0].values