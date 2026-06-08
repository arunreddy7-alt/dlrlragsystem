"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

from backend.api import auth, chat, data, documents, upload
from backend.core.config import get_settings
from backend.core.database import db
from backend.core.logging import configure_logging
from backend.models.schemas import HealthResponse

configure_logging()

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Initialize database and local directories on startup."""

    db.initialize()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Started {}", settings.app_name)
    yield


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)

app.include_router(auth.router)
app.include_router(upload.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(data.router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log requests and response status codes."""

    logger.info("{} {}", request.method, request.url.path)
    response = await call_next(request)
    logger.info("{} {} -> {}", request.method, request.url.path, response.status_code)
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return JSON for unexpected errors."""

    logger.exception("Unhandled error for {} {}", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check endpoint."""

    return HealthResponse(status="ok", app=settings.app_name)