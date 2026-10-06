"""Typed application settings, read from environment variables (and an optional .env file)."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # frozen: settings are immutable after startup and hashable (factories cache per settings)
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", frozen=True)

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = Field(False, description="JSON lines (containers) instead of console rendering")

    # Model artifacts generated locally (e.g. GLiNER ONNX export)
    models_dir: Path = Path(".models")

    # OCR — RapidOCR (PP-OCRv5 on ONNX) by default (spike 01); `tesseract` is the fallback
    ocr_engine: Literal["rapidocr", "tesseract"] = "rapidocr"
    ocr_dpi: int = Field(150, ge=72, le=600, description="Render resolution for OCR")
    ocr_tighten_boxes: bool = Field(True, description="Shrink word boxes vertically to the ink they contain")

    # NER — GLiNER int8 by default (spike 02); `spacy` is the fallback
    ner_engine: Literal["gliner", "spacy"] = "gliner"
    gliner_model: str = "urchade/gliner_small-v2.1"
    gliner_quantized: bool = True
    gliner_threshold: float = Field(0.3, ge=0.0, le=1.0, description="Calibrated for the int8 model")
    spacy_model: str = "en_core_web_sm"

    # Fuzzy matching (the requirement is 90%)
    similarity_threshold: int = Field(90, ge=0, le=100)

    # Embeddings — bge-small via fastembed by default (spike 03)
    embedding_engine: Literal["fastembed", "sentence-transformers"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    embedding_document_prefix: str = ""

    # Vector store
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_timeout_s: int = 10
    qdrant_collection: str = "pdf_documents"
    retrieval_top_k: int = Field(3, ge=1, le=50)
    retrieval_score_threshold: float = Field(0.5, ge=-1.0, le=1.0)

    # Semantic answer cache (spike 04): cosine threshold + QuestionGuard, TTL via payload filter
    answer_cache_enabled: bool = True
    answer_cache_collection: str = "answer_cache"
    answer_cache_threshold: float = Field(0.90, ge=0.0, le=1.0)
    answer_cache_ttl_s: int = Field(24 * 3600, ge=60)

    # RAG chunking
    chunk_size: int = Field(500, ge=50, description="Target chunk size in characters")

    # Answering backend for /api/ask: the agent service (default) or the in-process
    # OpenAI-compatible client (reference implementation exercised by the provided tests)
    answer_backend: Literal["agent", "inline"] = "agent"
    agent_url: str = "http://localhost:8080"
    agent_timeout_s: float = 60.0

    # Agent service model (Strands). Gemini on Vertex via ADC locally; Bedrock on AgentCore in AWS.
    # auto: Gemini if GOOGLE_CLOUD_PROJECT and credentials are present, else none (extractive answers).
    llm_provider: Literal["auto", "gemini", "bedrock", "none"] = "auto"
    gemini_model: str = "gemini-3.8-flash"
    gemini_project: str | None = Field(None, validation_alias="GOOGLE_CLOUD_PROJECT")
    gemini_location: str = Field("global", validation_alias="GOOGLE_CLOUD_LOCATION")
    bedrock_model_id: str = "anthropic.claude-haiku-4-5-20251001-v1:0"
    aws_region: str = "us-east-1"
    llm_temperature: float = Field(0.1, ge=0.0, le=2.0)
    llm_max_output_tokens: int = Field(1024, ge=64)

    # Inline backend only (ANSWER_BACKEND=inline): OpenAI-compatible chat completions API
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-3.5-turbo"
    llm_api_key: SecretStr | None = Field(None, validation_alias="OPENAI_API_KEY")
    llm_timeout_s: float = 30.0

    # Uploads
    max_upload_mb: int = Field(25, ge=1)
    index_on_extract: bool = Field(True, description="Index /api/extract documents for /api/ask in the background")
    max_pages: int = Field(200, ge=1, description="Reject documents with more pages (non-retryable)")

    # Object storage (local volume; S3 in AWS)
    storage_dir: Path = Path(".data/storage")

    # Postgres (jobs). Pools are per process and sized from config.
    database_url: SecretStr = SecretStr("postgresql://app:app@localhost:55432/app")  # compose default
    db_pool_min_size: int = Field(1, ge=0)
    db_pool_max_size: int = Field(5, ge=1)
    db_pool_timeout_s: float = Field(2.0, gt=0, description="Max wait to acquire a connection")
    db_pool_max_waiting: int = Field(50, ge=0, description="Queue length before rejecting (backpressure)")
    db_statement_timeout_ms: int = Field(15_000, ge=0)

    # Temporal
    temporal_address: str = "localhost:7233"
    temporal_namespace: str = "default"
    workflow_task_queue: str = "extraction-workflows"
    cpu_task_queue: str = "extraction-cpu"
    io_task_queue: str = "extraction-io"
    cpu_worker_concurrency: int = Field(2, ge=1, description="Concurrent CPU activities per worker")
    reconcile_interval_s: int = Field(60, ge=5)
    reconcile_stale_after_s: int = Field(120, ge=10, description="Queued jobs older than this are re-started")

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 2**20


@lru_cache
def get_settings() -> Settings:
    return Settings()
