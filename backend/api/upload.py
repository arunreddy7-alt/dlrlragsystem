"""PDF upload route."""

import sqlite3
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from loguru import logger

from backend.api.auth import get_current_user
from backend.core.config import get_settings
from backend.core.database import db
from backend.models.schemas import DocumentResponse
from backend.rag.ingestion import ingest_pdf

router = APIRouter(tags=["upload"])


def _safe_filename(name: str) -> str:
    candidate = Path(name).name.replace("\\", "_").replace("/", "_").strip()
    if not candidate.lower().endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF uploads are supported")
    if not candidate:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid filename")
    return candidate


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_pdf(file: UploadFile = File(...), user: sqlite3.Row = Depends(get_current_user)) -> DocumentResponse:
    """Upload a PDF, persist chunks, and refresh indexes."""

    settings = get_settings()
    if db.count_documents(user["id"]) >= settings.max_pdfs_per_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Maximum 5 PDFs per user is enforced")
    filename = _safe_filename(file.filename or "")
    user_dir = settings.upload_dir / f"user_{user['id']}"
    user_dir.mkdir(parents=True, exist_ok=True)
    document = db.add_document(user["id"], filename)
    stored_name = f"{document['id']}_{filename}"
    path = user_dir / stored_name
    try:
        data = await file.read()
        if not data.startswith(b"%PDF"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is not a valid PDF")
        path.write_bytes(data)
        chunk_count = ingest_pdf(user["id"], document["id"], path, filename)
        logger.info("Uploaded PDF user_id={} document_id={} chunks={}", user["id"], document["id"], chunk_count)
    except Exception:
        if path.exists():
            path.unlink()
        db.delete_document(user["id"], document["id"])
        raise
    return DocumentResponse(
        id=document["id"],
        user_id=document["user_id"],
        filename=document["filename"],
        upload_time=document["upload_time"],
    )
