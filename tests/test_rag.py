"""RAG utility tests."""

from langchain_core.documents import Document

from backend.rag.chunker import split_documents
from backend.rag.generator import _format_context


def test_split_documents_preserves_page_number() -> None:
    docs = [Document(page_content="This is a small page.", metadata={"filename": "a.pdf", "page_number": 3})]
    chunks = split_documents(docs)
    assert chunks
    assert chunks[0].metadata["page_number"] == 3


def test_format_context_keeps_citation_labels() -> None:
    docs = [Document(page_content="Important local fact.", metadata={"filename": "a.pdf", "page_number": 4})]
    context = _format_context(docs)
    assert "Source 1: a.pdf (Page 4)" in context
    assert "Important local fact." in context
