from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path):
    """Load and extract text from a PDF document."""
    try:
        loader=PyPDFLoader(file_path)
        documents=loader.load()
        if not documents:
            print("No content found in the PDF.")

        return documents
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return []