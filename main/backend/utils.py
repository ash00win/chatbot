import fitz  # PyMuPDF
from backend.vec import store_embeddings

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def extract_text(file):
    text = None

    if file.type == "application/pdf":
        import fitz  # PyMuPDF
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        text = "\n".join([page.get_text() or "" for page in pdf_document])

    elif file.type == "text/plain":
        text = file.read().decode("utf-8")

    if text:
        chunks = chunk_text(text)  
        for chunk in chunks:
            store_embeddings(chunk)  

    return text
