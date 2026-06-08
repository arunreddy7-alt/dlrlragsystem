"""PDF ingestion and index rebuilding."""

import shutil
from pathlib import Path

from langchain_core.documents import Document

from backend.core.database import db
from backend.rag.bm25_store import bm25_path, build_bm25_index, clear_bm25_cache
from backend.rag.chunker import split_documents
from backend.rag.parser import load_pdf
from backend.rag.vector_store import add_to_faiss_index, build_faiss_index, clear_faiss_cache, user_index_dir


def ingest_pdf(user_id: int, document_id: int, path: Path, filename: str) -> int:
    """Parse, chunk, store, and update indexes for a PDF."""

    pages = load_pdf(path, filename)
    chunks = split_documents(pages)
    rows = [
        (int(chunk.metadata.get("page_number", 0)), chunk.page_content, dict(chunk.metadata))
        for chunk in chunks
        if chunk.page_content.strip()
    ]
    db.add_chunks(document_id, rows)
    new_documents = [
        Document(
            page_content=chunk.page_content,
            metadata={
                "filename": filename,
                "page_number": int(chunk.metadata.get("page_number", 0)),
                "document_id": document_id,
            },
        )
        for chunk in chunks
        if chunk.page_content.strip()
    ]
    add_to_faiss_index(user_id, new_documents)
    rebuild_bm25_index(user_id)
    return len(rows)


def _documents_from_database(user_id: int) -> list[Document]:
    rows = db.list_chunks(user_id)
    return [
        Document(
            page_content=row["chunk_text"],
            metadata={
                "filename": row["filename"],
                "page_number": int(row["page_number"]),
                "document_id": int(row["document_id"]),
                "chunk_id": int(row["id"]),
            },
        )
        for row in rows
    ]


def rebuild_bm25_index(user_id: int) -> None:
    """Rebuild the BM25 index from SQLite chunks."""

    documents = _documents_from_database(user_id)
    bm25_file = bm25_path(user_id)
    if not documents:
        if bm25_file.exists():
            bm25_file.unlink()
        clear_bm25_cache(user_id)
        return
    build_bm25_index(user_id, documents)


def rebuild_indexes(user_id: int) -> None:
    """Rebuild persisted FAISS and BM25 indexes from SQLite chunks."""

    documents = _documents_from_database(user_id)
    faiss_dir = user_index_dir(user_id)
    bm25_file = bm25_path(user_id)
    if not documents:
        if faiss_dir.exists():
            shutil.rmtree(faiss_dir)
        if bm25_file.exists():
            bm25_file.unlink()
        clear_faiss_cache(user_id)
        clear_bm25_cache(user_id)
        return
    build_faiss_index(user_id, documents)
    build_bm25_index(user_id, documents)