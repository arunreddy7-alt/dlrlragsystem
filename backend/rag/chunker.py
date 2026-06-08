"""Text chunking for PDF pages."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.core.config import get_settings


def split_documents(documents: list[Document]) -> list[Document]:
    """Split page documents into overlapping chunks."""

    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
        chunk.metadata["page_number"] = int(chunk.metadata.get("page_number", chunk.metadata.get("page", 0) + 1))
    return chunks
