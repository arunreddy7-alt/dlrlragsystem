"""Hybrid FAISS and BM25 retrieval."""

from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from backend.core.config import get_settings
from backend.rag.bm25_store import load_bm25_index
from backend.rag.vector_store import load_faiss_index


def _key(document: Document) -> tuple[str, int, str]:
    return (
        str(document.metadata.get("filename", "")),
        int(document.metadata.get("page_number", 0)),
        document.page_content.strip(),
    )


def retrieve(user_id: int, question: str) -> list[Document]:
    """Retrieve up to eight deduplicated chunks using EnsembleRetriever."""

    settings = get_settings()
    faiss = load_faiss_index(user_id)
    bm25 = load_bm25_index(user_id)
    retrievers = []
    weights = []
    if faiss is not None:
        retrievers.append(faiss.as_retriever(search_kwargs={"k": settings.faiss_top_k}))
        weights.append(0.7)
    if bm25 is not None:
        bm25.k = settings.bm25_top_k
        retrievers.append(bm25)
        weights.append(0.3)
    if not retrievers:
        return []
    if len(retrievers) == 1:
        raw_results = retrievers[0].invoke(question)
    else:
        ensemble = EnsembleRetriever(retrievers=retrievers, weights=weights)
        raw_results = ensemble.invoke(question)

    seen: set[tuple[str, int, str]] = set()
    deduped: list[Document] = []
    for document in raw_results:
        doc_key = _key(document)
        if doc_key in seen:
            continue
        seen.add(doc_key)
        deduped.append(document)
        if len(deduped) >= settings.final_context_chunks:
            break
    return deduped
