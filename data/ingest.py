from data.pdf_loader import load_pdf
from data.chunker import chunk_text


def ingest_pdf(path):

    text = load_pdf(path)

    chunks = chunk_text(text)

    return chunks