"""Application configuration for the offline RAG assistant."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or .env."""

    app_name: str = "Offline RAG Assistant"
    database_path: Path = Path("backend/data/app.db")
    upload_dir: Path = Path("backend/uploads")
    index_dir: Path = Path("backend/indexes")
    jwt_secret_key: str = Field(default="change-this-local-secret-before-use", min_length=16)
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440
    ollama_base_url: str = "http://localhost:11434"
    max_pdfs_per_user: int = 5
    chunk_size: int = 1200
    chunk_overlap: int = 200
    faiss_top_k: int = 5
    bm25_top_k: int = 5
    final_context_chunks: int = 8

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def generation_model(self) -> str:
        """Allowed local Ollama generation model."""

        return "phi3:mini"

    @property
    def embedding_model(self) -> str:
        """Allowed local Ollama embedding model."""

        return "nomic-embed-text"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings and ensure local directories exist."""

    settings = Settings()
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    return settings
