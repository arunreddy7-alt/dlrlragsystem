"""Document listing and deletion routes."""

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from backend.api.auth import get_current_user
from backend.core.config import get_settings
from backend.core.database import db
from backend.models.schemas import DocumentResponse
from backend.rag.ingestion import rebuild_indexes

router = APIRouter(tags=["documents"])


def _row_to_document(row: sqlite3.Row) -> DocumentResponse:
    return DocumentResponse(
        id=row["id"],
        user_id=row["user_id"],
        filename=row["filename"],
        upload_time=row["upload_time"],
    )


@router.get("/documents", response_model=list[DocumentResponse])
def list_documents(user: sqlite3.Row = Depends(get_current_user)) -> list[DocumentResponse]:
    """List PDFs uploaded by the authenticated user."""

    return [_row_to_document(row) for row in db.list_documents(user["id"])]


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: int, user: sqlite3.Row = Depends(get_current_user)) -> None:
    """Delete a PDF and rebuild indexes."""

    document = db.get_document(user["id"], document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    deleted = db.delete_document(user["id"], document_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    upload_path = get_settings().upload_dir / f"user_{user['id']}" / f"{document_id}_{document['filename']}"
    if upload_path.exists():
        upload_path.unlink()
    rebuild_indexes(user["id"])
    logger.info("Deleted PDF user_id={} document_id={}", user["id"], document_id)
