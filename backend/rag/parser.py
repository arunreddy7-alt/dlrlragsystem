"""PDF parsing utilities."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(path: Path, filename: str) -> list[Document]:
    """Load a PDF with PyPDFLoader and normalize metadata."""

    loader = PyPDFLoader(str(path))
    pages = loader.load()
    for page in pages:
        page_number = int(page.metadata.get("page", 0)) + 1
        page.metadata.update({"filename": filename, "page_number": page_number})
    return pages
