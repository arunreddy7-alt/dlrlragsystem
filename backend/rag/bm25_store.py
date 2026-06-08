"""BM25 keyword retriever management."""

import pickle
from pathlib import Path
from threading import RLock

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from backend.core.config import get_settings

_cache_lock = RLock()
_bm25_cache: dict[int, tuple[float, BM25Retriever]] = {}


def bm25_path(user_id: int) -> Path:
    """Return the persisted BM25 retriever path."""

    path = get_settings().index_dir / f"user_{user_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path / "bm25.pkl"


def build_bm25_index(user_id: int, documents: list[Document]) -> BM25Retriever | None:
    """Build and persist a BM25 retriever."""

    if not documents:
        clear_bm25_cache(user_id)
        return None
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = get_settings().bm25_top_k
    path = bm25_path(user_id)
    with path.open("wb") as handle:
        pickle.dump(retriever, handle)
    with _cache_lock:
        _bm25_cache[user_id] = (path.stat().st_mtime, retriever)
    return retriever


def load_bm25_index(user_id: int) -> BM25Retriever | None:
    """Load a persisted BM25 retriever if present."""

    path = bm25_path(user_id)
    if not path.exists():
        clear_bm25_cache(user_id)
        return None
    mtime = path.stat().st_mtime
    with _cache_lock:
        cached = _bm25_cache.get(user_id)
        if cached is not None and cached[0] == mtime:
            cached[1].k = get_settings().bm25_top_k
            return cached[1]
    with path.open("rb") as handle:
        retriever = pickle.load(handle)
    retriever.k = get_settings().bm25_top_k
    with _cache_lock:
        _bm25_cache[user_id] = (mtime, retriever)
    return retriever


def clear_bm25_cache(user_id: int) -> None:
    """Drop a user's BM25 retriever from memory."""

    with _cache_lock:
        _bm25_cache.pop(user_id, None)