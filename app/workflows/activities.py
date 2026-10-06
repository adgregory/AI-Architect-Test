"""Temporal activities. Dependencies are injected through the constructor.

CpuActivities run model inference (sync, in a bounded thread pool) and never touch the
database. IoActivities own all database / vector-store writes (async).
"""

from __future__ import annotations

from dataclasses import asdict

import fitz
from temporalio import activity
from temporalio.exceptions import ApplicationError

from app.core.logging import get_logger
from app.db import JobRepository
from app.services.embedding_service import EmbeddingService
from app.services.extraction_service import ExtractionEngines, NameExtraction
from app.services.ocr_service import OCRResult
from app.services.rag_service import TextChunker
from app.services.vector_service import VectorStore
from app.storage import JobArtifacts, ObjectStorage
from app.workflows.models import (
    CHUNK_AND_EMBED,
    COMPLETE_JOB,
    EXTRACT_AND_MATCH,
    FAIL_JOB,
    FIND_STALE_JOBS,
    INVALID_DOCUMENT,
    MARK_RUNNING,
    OCR_PAGE,
    PREPARE_DOCUMENT,
    UPSERT_CHUNKS,
    ExtractRequest,
    ExtractSummary,
    ExtractTask,
    FailRequest,
    IndexRequest,
    MarkRunning,
    PageTask,
    ReconcileRequest,
)

log = get_logger(__name__)


def _combine(pages: list[dict]) -> OCRResult:
    return OCRResult(text="\n".join(p["text"] for p in pages), words=[w for p in pages for w in p["words"]])


class CpuActivities:
    def __init__(self, engines: ExtractionEngines, embeddings: EmbeddingService, storage: ObjectStorage,
                 chunker: TextChunker, chunk_size: int, max_pages: int):
        self._engines = engines
        self._embeddings = embeddings
        self._storage = storage
        self._chunker = chunker
        self._chunk_size = chunk_size
        self._max_pages = max_pages

    def _pages(self, art: JobArtifacts, page_count: int) -> list[dict]:
        return [art.get_json(art.page(n)) for n in range(page_count)]

    @activity.defn(name=PREPARE_DOCUMENT)
    def prepare_document(self, job_id: str) -> int:
        """Validate the stored PDF and return its page count. Invalid input is non-retryable."""
        art = JobArtifacts(self._storage, job_id)
        try:
            with art.storage.local_path(art.input_pdf) as path:
                pages = self._engines.ocr.page_count(str(path))
        except FileNotFoundError:
            raise ApplicationError("input PDF is missing from storage", type=INVALID_DOCUMENT, non_retryable=True)
        except (fitz.FileDataError, RuntimeError, ValueError) as exc:
            raise ApplicationError(f"not a readable PDF: {exc}", type=INVALID_DOCUMENT, non_retryable=True)
        if pages == 0:
            raise ApplicationError("PDF has no pages", type=INVALID_DOCUMENT, non_retryable=True)
        if pages > self._max_pages:
            raise ApplicationError(f"PDF has {pages} pages; the limit is {self._max_pages}",
                                   type=INVALID_DOCUMENT, non_retryable=True)
        return pages

    @activity.defn(name=OCR_PAGE)
    def ocr_page(self, task: PageTask) -> str:
        activity.heartbeat(task.page_number)
        art = JobArtifacts(self._storage, task.job_id)
        with art.storage.local_path(art.input_pdf) as path:
            result = self._engines.ocr.read_page(str(path), task.page_number)
        return art.put_json(art.page(task.page_number), asdict(result))

    @activity.defn(name=EXTRACT_AND_MATCH)
    def extract_and_match(self, task: ExtractTask) -> ExtractSummary:
        art = JobArtifacts(self._storage, task.job_id)
        result = NameExtraction(self._engines).run(_combine(self._pages(art, task.page_count)), task.query_names)
        art.put_json(art.result, result)
        return ExtractSummary(names=len(result["extracted_names"]), matches=len(result["fuzzy_matches"]))

    @activity.defn(name=CHUNK_AND_EMBED)
    def chunk_and_embed(self, req: IndexRequest) -> int:
        art = JobArtifacts(self._storage, req.job_id)
        text = _combine(self._pages(art, req.page_count)).text
        chunks = [c for c in self._chunker.chunk(text, self._chunk_size) if c.strip()]
        art.put_json(art.chunks, {"texts": chunks, "vectors": self._embeddings.embed_documents(chunks)})
        return len(chunks)


class IoActivities:
    def __init__(self, repo: JobRepository, storage: ObjectStorage, vector_store: VectorStore):
        self._repo = repo
        self._storage = storage
        self._vector_store = vector_store

    @activity.defn(name=MARK_RUNNING)
    async def mark_running(self, req: MarkRunning) -> None:
        await self._repo.mark_running(req.job_id, req.page_count)

    @activity.defn(name=COMPLETE_JOB)
    async def complete_job(self, job_id: str) -> None:
        art = JobArtifacts(self._storage, job_id)
        await self._repo.complete(job_id, art.get_json(art.result))

    @activity.defn(name=FAIL_JOB)
    async def fail_job(self, req: FailRequest) -> None:
        await self._repo.fail(req.job_id, req.error)

    @activity.defn(name=UPSERT_CHUNKS)
    async def upsert_chunks(self, req: IndexRequest) -> int:
        art = JobArtifacts(self._storage, req.job_id)
        data = art.get_json(art.chunks)
        if not data["texts"]:
            return 0
        self._vector_store.ensure_collection()
        self._vector_store.upsert(data["texts"], data["vectors"], document_id=req.job_id,
                                  metadata=[{"source": req.filename}] * len(data["texts"]))
        return len(data["texts"])

    @activity.defn(name=FIND_STALE_JOBS)
    async def find_stale_jobs(self, req: ReconcileRequest) -> list[ExtractRequest]:
        jobs = await self._repo.stale_queued(req.older_than_s, req.limit)
        return [ExtractRequest(j.id, j.query_names, j.filename, req.queues) for j in jobs]
