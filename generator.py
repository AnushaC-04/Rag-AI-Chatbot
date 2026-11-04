import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

class GeminiGenerator:
    def __init__(self):
        api_key=os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set."
            )
        self.client=genai.Client(
            api_key=api_key
        )
        self.model=os.getenv(
            "GEMINI_MODEL",
            "gemini-3.6-flash"
        )
    def generate_answer(self, query, retrieved_chunks):
        if not retrieved_chunks:
            return (
                "I could not find relevant information "
                "in the selected document."
            )
        context_parts=[]
        for index, chunk in enumerate(
            retrieved_chunks,
            start=1
        ):
            metadata=chunk["metadata"]
            page=metadata.get("page", "Unknown")
            context_parts.append(
                f"""
                Source {index}
                Page: {page}
                Content:
                {chunk['content']}
                """
            )

        context="\n\n".join(context_parts)
        prompt=f"""
You are a reliable document question-answering assistant.

Answer the user's question using ONLY the supplied document context.

Instructions:

1. For factual questions, accurately extract the relevant information.
2. For analytical questions, combine relevant information from
   different sections and draw reasonable conclusions.
3. Never invent facts, figures, policies, skills, or examples.
4. Clearly distinguish information explicitly stated in the
   document from evidence-based conclusions.
5. If the context does not contain sufficient information,
   say that the answer could not be found in the provided document.
6. Use a suitable structure for the question.
7. Use headings, bullet points, numbered lists, or tables
   when they improve readability.
8. Do not unnecessarily repeat information.

Document Context:

{context}

User Question:

{query}

Answer:
"""
        response=self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return response.text