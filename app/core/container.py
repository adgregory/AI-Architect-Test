"""Application container: builds the configured services once and owns their lifecycle.

Created in the FastAPI lifespan (models are loaded and warmed at startup, never per
request) and exposed to routes through dependencies.
"""

from __future__ import annotations

import threading
import time
from functools import cached_property

from app.core.config import Settings
from app.core.factories import (
    EmbeddingServiceFactory,
    LLMClientFactory,
    NERServiceFactory,
    OCRServiceFactory,
    VectorStoreFactory,
    default_qdrant_client,
)
from app.core.lazy import Lazy
from app.core.logging import get_logger
from app.db import JobEventHub, JobRepository, Pool
from app.services.bbox_service import ConsecutiveWordNameLocator
from app.services.extraction_service import ExtractionEngines, ExtractionSession
from app.services.fuzzy_service import TokenSortNameMatcher
from app.services.indexing_service import DocumentIndexer
from app.services.ner_service import PersonNameNormalizer
from app.services.rag_service import RAGService, TextChunker
from app.storage import LocalFileStorage, ObjectStorage
from app.workflows.client import JobOrchestrator

log = get_logger(__name__)


class Container:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.RLock()  # first use may race across threadpool workers
        # Async resources (opened in the lifespan / worker startup)
        self.db_pool: Pool | None = None
        self.job_repository: JobRepository | None = None
        self.event_hub: JobEventHub | None = None
        self.orchestrator: JobOrchestrator | None = None

    def _build(self, name: str, build):
        with self._lock:
            started = time.perf_counter()
            value = build()
            log.info("container.built", component=name, seconds=round(time.perf_counter() - started, 2))
            return value

    # Engines -------------------------------------------------------------- #
    @cached_property
    def ocr(self):
        return self._build("ocr", lambda: OCRServiceFactory.create(self.settings))

    @cached_property
    def ner(self):
        return self._build("ner", lambda: NERServiceFactory.create(self.settings))

    @cached_property
    def embeddings(self):
        return self._build("embeddings", lambda: EmbeddingServiceFactory.create(self.settings))

    @cached_property
    def vector_store(self):
        return self._build("vector_store", lambda: VectorStoreFactory.create(self.settings))

    @cached_property
    def llm(self):
        return self._build("llm", lambda: LLMClientFactory.create(self.settings))

    @cached_property
    def extraction_engines(self) -> ExtractionEngines:
        return ExtractionEngines(
            ocr=self.ocr,
            ner=self.ner,
            locator=ConsecutiveWordNameLocator(),
            matcher=TokenSortNameMatcher(self.settings.similarity_threshold),
            normalizer=PersonNameNormalizer(),
        )

    @cached_property
    def chunker(self) -> TextChunker:
        return TextChunker()

    @cached_property
    def indexer(self) -> DocumentIndexer:
        """Full indexer (chunk + embed + store) for in-process use (/api/ingest)."""
        return DocumentIndexer(self.chunker, self.settings.chunk_size, self.embeddings, self.vector_store)

    @cached_property
    def http_clients(self):
        """Shared HTTP clients for the agent service: pooled, bounded, with timeouts."""
        import httpx

        limits = httpx.Limits(max_connections=50, max_keepalive_connections=10)
        timeout = httpx.Timeout(self.settings.agent_timeout_s, connect=5.0)
        return httpx.Client(limits=limits, timeout=timeout), httpx.AsyncClient(limits=limits, timeout=timeout)

    @cached_property
    def rag(self):
        """Answering backend for /api/ask: the agent service (default) or the in-process
        reference RAGService. Both expose answer(question) -> {answer, sources}."""
        if self.settings.answer_backend == "agent":
            from app.services.agent_client import AgentAnswerClient

            http, ahttp = self.http_clients
            return AgentAnswerClient(self.settings.agent_url, http, ahttp)
        return self.inline_rag

    @cached_property
    def inline_rag(self) -> RAGService:
        # Collaborators resolve on first use: building the service never loads a model.
        return RAGService(
            embed_query=lambda q: self.embeddings.embed_query(q),
            retrieve=lambda vector, top_k: self.vector_store.search(vector, top_k),
            llm=Lazy(lambda: self.llm),
            top_k=self.settings.retrieval_top_k,
        )

    @cached_property
    def storage(self) -> ObjectStorage:
        return LocalFileStorage(self.settings.storage_dir)

    # Per request ---------------------------------------------------------- #
    def extraction_session(self) -> ExtractionSession:
        return ExtractionSession(lambda: self.extraction_engines)

    # Lifecycle ------------------------------------------------------------ #
    def warm_up(self) -> None:
        """Load every model now so the first request doesn't pay for it."""
        _ = self.extraction_engines
        self.embeddings.embed_query("warm-up")
        self.ner.extract_names("Warm-up text mentioning Jane Doe.")

    def close(self) -> None:
        if "vector_store" in self.__dict__:
            default_qdrant_client(self.settings).close()

    async def open_async(self) -> None:
        """Jobs infrastructure. Each part degrades independently: the sync API keeps working
        without Postgres/Temporal; job starts fall back to the reconciler; SSE falls back to polling."""
        from app.db import PostgresJobRepository, create_pool
        from app.workflows.client import LazyTemporalOrchestrator

        pool = create_pool(self.settings)
        await pool.open(wait=False)  # don't block startup on the database
        self.db_pool = pool
        self.job_repository = PostgresJobRepository(pool)
        self.orchestrator = LazyTemporalOrchestrator(self.settings)
        hub = JobEventHub(self.settings.database_url.get_secret_value())
        try:
            await hub.start()
            self.event_hub = hub
        except Exception as exc:  # noqa: BLE001 - startup must not fail on optional infra
            log.warning("job_event_hub.unavailable", error=str(exc))

    async def close_async(self) -> None:
        if "http_clients" in self.__dict__:
            http, ahttp = self.http_clients
            http.close()
            await ahttp.aclose()
        if self.event_hub:
            await self.event_hub.stop()
        if self.db_pool:
            await self.db_pool.close()
