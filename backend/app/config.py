from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Atman RAG"

    llm_provider: str = "openrouter"
    openrouter_api_key: str | None = None
    openrouter_model: str = "openrouter/free"
    openrouter_site_url: str = "http://localhost:3000"
    openrouter_app_name: str = "Atman RAG"

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash-lite"
    gemini_embedding_model: str = "gemini-embedding-001"
    embedding_provider: str = "fastembed"
    fastembed_model: str = "BAAI/bge-small-en-v1.5"
    local_embedding_dimensions: int = 768

    frontend_origin: str = "http://localhost:3000"
    frontend_origin_regex: str | None = None
    qdrant_path: Path = PROJECT_ROOT / "storage" / "qdrant"
    collection_name: str = "atman_rag_chunks_fastembed"

    data_dir: Path = PROJECT_ROOT / "data"
    sample_sources_path: Path = PROJECT_ROOT / "data" / "sample_sources.json"
    sample_docs_dir: Path = PROJECT_ROOT / "data" / "sample_docs"
    uploads_dir: Path = PROJECT_ROOT / "data" / "uploads"
    registry_path: Path = PROJECT_ROOT / "storage" / "documents.json"

    chunk_chars: int = 3200
    chunk_overlap_chars: int = 450
    top_k: int = 5
    min_relevance_score: float = 0.55
    max_context_chars: int = 14000

    auto_ingest_samples: bool = False
    allow_sample_download: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
