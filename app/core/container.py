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
from app.services.bbox_service import ConsecutiveWordNameLocator
from app.services.extraction_service import ExtractionEngines, ExtractionSession
from app.services.fuzzy_service import TokenSortNameMatcher
from app.services.ner_service import PersonNameNormalizer
from app.services.rag_service import RAGService, TextChunker

log = get_logger(__name__)


class Container:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.RLock()  # first use may race across threadpool workers

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
    def rag(self) -> RAGService:
        # Collaborators resolve on first use: building the service never loads a model.
        return RAGService(
            embed_query=lambda q: self.embeddings.embed_query(q),
            retrieve=lambda vector, top_k: self.vector_store.search(vector, top_k),
            llm=Lazy(lambda: self.llm),
            top_k=self.settings.retrieval_top_k,
        )

    # Per request ---------------------------------------------------------- #
    def extraction_session(self) -> ExtractionSession:
        return ExtractionSession(lambda: self.extraction_engines)

    # Lifecycle ------------------------------------------------------------ #
    def warm_up(self) -> None:
        """Load every model now so the first request doesn't pay for it."""
        self.extraction_engines
        self.embeddings.embed_query("warm-up")
        self.ner.extract_names("Warm-up text mentioning Jane Doe.")

    def close(self) -> None:
        if "vector_store" in self.__dict__:
            default_qdrant_client(self.settings).close()
