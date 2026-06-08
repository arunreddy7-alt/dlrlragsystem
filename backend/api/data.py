"""User data maintenance routes."""

import shutil
import sqlite3

from fastapi import APIRouter, Depends, status
from loguru import logger

from backend.api.auth import get_current_user
from backend.core.config import get_settings
from backend.core.database import db
from backend.rag.bm25_store import clear_bm25_cache
from backend.rag.vector_store import clear_faiss_cache

router = APIRouter(tags=["data"])


@router.delete("/data/clear", status_code=status.HTTP_204_NO_CONTENT)
def clear_data(user: sqlite3.Row = Depends(get_current_user)) -> None:
    """Clear the authenticated user's PDFs, indexes, and chat history."""

    user_id = int(user["id"])
    db.clear_user_data(user_id)
    settings = get_settings()
    upload_dir = settings.upload_dir / f"user_{user_id}"
    index_dir = settings.index_dir / f"user_{user_id}"
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    if index_dir.exists():
        shutil.rmtree(index_dir)
    clear_faiss_cache(user_id)
    clear_bm25_cache(user_id)
    logger.info("Cleared local data for user_id={}", user_id)