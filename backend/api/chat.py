"""Chat routes."""

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from backend.api.auth import get_current_user
from backend.core.database import db
from backend.models.schemas import ChatHistoryItem, ChatRequest, ChatResponse, Source
from backend.rag.generator import generate_answer, generate_general_answer
from backend.rag.hybrid_retriever import retrieve

router = APIRouter(tags=["chat"])


def _sources_from_answer(answer: str) -> list[Source]:
    sources: list[Source] = []
    for line in answer.splitlines():
        text = line.strip()
        if not text.startswith("* ") or " (Page " not in text:
            continue
        filename, _, page_part = text[2:].partition(" (Page ")
        page_value = page_part.rstrip(")")
        if filename and page_value.isdigit():
            item = Source(filename=filename, page_number=int(page_value))
            if item not in sources:
                sources.append(item)
    return sources


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, user: sqlite3.Row = Depends(get_current_user)) -> ChatResponse:
    """Answer a user question with hybrid retrieval and source citations."""

    try:
        history_rows = db.list_chat_history(user["id"], limit=100)
        history = [{"role": row["role"], "message": row["message"]} for row in history_rows]
        documents = retrieve(user["id"], payload.message)
        answer = generate_answer(payload.message, documents, history)
        db.add_chat_message(user["id"], "user", payload.message)
        db.add_chat_message(user["id"], "assistant", answer)
        sources = _sources_from_answer(answer)
        logger.info("Chat answered user_id={} sources={}", user["id"], len(sources))
        return ChatResponse(answer=answer, sources=sources)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Chat failed for user_id={}", user["id"])
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post("/chat/general", response_model=ChatResponse)
def general_chat(payload: ChatRequest, user: sqlite3.Row = Depends(get_current_user)) -> ChatResponse:
    """Answer a general local chatbot message without PDF retrieval."""

    try:
        history_rows = db.list_chat_history(user["id"], limit=100)
        history = [{"role": row["role"], "message": row["message"]} for row in history_rows]
        answer = generate_general_answer(payload.message, history)
        db.add_chat_message(user["id"], "user", payload.message)
        db.add_chat_message(user["id"], "assistant", answer)
        logger.info("General chat answered user_id={}", user["id"])
        return ChatResponse(answer=answer, sources=[])
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("General chat failed for user_id={}", user["id"])
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.get("/chat/history", response_model=list[ChatHistoryItem])
def chat_history(user: sqlite3.Row = Depends(get_current_user)) -> list[ChatHistoryItem]:
    """Return persisted chat history."""

    return [
        ChatHistoryItem(
            id=row["id"],
            user_id=row["user_id"],
            role=row["role"],
            message=row["message"],
            timestamp=row["timestamp"],
        )
        for row in db.list_chat_history(user["id"], limit=200)
    ]