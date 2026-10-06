"""Typed application settings, read from environment variables (and an optional .env file)."""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OCR
    ocr_dpi: int = Field(150, ge=72, le=600, description="Render resolution for OCR")

    # NER
    spacy_model: str = "en_core_web_sm"

    # Fuzzy matching (the requirement is 90%)
    similarity_threshold: int = Field(90, ge=0, le=100)

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Vector store
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_timeout_s: float = 10.0
    qdrant_collection: str = "pdf_documents"
    retrieval_top_k: int = Field(3, ge=1, le=50)
    retrieval_score_threshold: float = Field(0.5, ge=-1.0, le=1.0)

    # RAG chunking
    chunk_size: int = Field(500, ge=50, description="Target chunk size in characters")

    # LLM (OpenAI-compatible chat completions API)
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-3.5-turbo"
    llm_api_key: SecretStr | None = Field(None, validation_alias="OPENAI_API_KEY")
    llm_timeout_s: float = 30.0

    # Uploads
    max_upload_mb: int = Field(25, ge=1)


@lru_cache
def get_settings() -> Settings:
    return Settings()
