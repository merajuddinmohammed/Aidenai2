import fitz  # PyMuPDF


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract all text from a PDF file."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)


def chunk_text(text: str, max_chars: int = 2000) -> list[str]:
    """Split text into chunks at sentence boundaries."""
    sentences = text.replace("\n", " ").split(". ")
    chunks = []
    current = ""

    for sentence in sentences:
        candidate = f"{current}. {sentence}" if current else sentence
        if len(candidate) > max_chars and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = candidate

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text]


def pdf_to_chunks(file_bytes: bytes, max_chars: int = 2000) -> list[str]:
    """Extract text from PDF and split into chunks."""
    text = extract_text_from_pdf(file_bytes)
    return chunk_text(text, max_chars)
