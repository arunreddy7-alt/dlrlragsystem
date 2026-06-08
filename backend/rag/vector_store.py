"""FAISS vector index management."""

from pathlib import Path
from threading import RLock

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from backend.core.config import get_settings
from backend.rag.embeddings import get_embeddings

_cache_lock = RLock()
_faiss_cache: dict[int, tuple[float, FAISS]] = {}


def user_index_dir(user_id: int) -> Path:
    """Return the FAISS directory for a user."""

    path = get_settings().index_dir / f"user_{user_id}" / "faiss"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _index_mtime(path: Path) -> float:
    index_file = path / "index.faiss"
    pkl_file = path / "index.pkl"
    if not index_file.exists() or not pkl_file.exists():
        return 0.0
    return max(index_file.stat().st_mtime, pkl_file.stat().st_mtime)


def build_faiss_index(user_id: int, documents: list[Document]) -> FAISS | None:
    """Build and persist a FAISS index for user documents."""

    if not documents:
        clear_faiss_cache(user_id)
        return None
    vector_store = FAISS.from_documents(documents, get_embeddings())
    path = user_index_dir(user_id)
    vector_store.save_local(str(path))
    with _cache_lock:
        _faiss_cache[user_id] = (_index_mtime(path), vector_store)
    return vector_store


def add_to_faiss_index(user_id: int, documents: list[Document]) -> FAISS | None:
    """Append new documents to a user's persisted FAISS index."""

    if not documents:
        return load_faiss_index(user_id)
    path = user_index_dir(user_id)
    vector_store = load_faiss_index(user_id)
    if vector_store is None:
        vector_store = FAISS.from_documents(documents, get_embeddings())
    else:
        vector_store.add_documents(documents)
    vector_store.save_local(str(path))
    with _cache_lock:
        _faiss_cache[user_id] = (_index_mtime(path), vector_store)
    return vector_store


def load_faiss_index(user_id: int) -> FAISS | None:
    """Load a persisted FAISS index if it exists."""

    path = user_index_dir(user_id)
    if not (path / "index.faiss").exists():
        clear_faiss_cache(user_id)
        return None
    mtime = _index_mtime(path)
    with _cache_lock:
        cached = _faiss_cache.get(user_id)
        if cached is not None and cached[0] == mtime:
            return cached[1]
    vector_store = FAISS.load_local(
        str(path),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
    with _cache_lock:
        _faiss_cache[user_id] = (mtime, vector_store)
    return vector_store


def clear_faiss_cache(user_id: int) -> None:
    """Drop a user's FAISS index from memory."""

    with _cache_lock:
        _faiss_cache.pop(user_id, None)