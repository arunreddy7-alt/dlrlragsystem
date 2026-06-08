"""Local Ollama embeddings."""

from langchain_community.embeddings import OllamaEmbeddings

from backend.core.config import get_settings


def get_embeddings() -> OllamaEmbeddings:
    """Return the allowed local Ollama embedding model."""

    settings = get_settings()
    return OllamaEmbeddings(model=settings.embedding_model, base_url=settings.ollama_base_url)
