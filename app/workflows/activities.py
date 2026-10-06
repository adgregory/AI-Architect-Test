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
from app.services.extraction_service import ExtractionEngines, NameExtraction
from app.services.indexing_service import DocumentIndexer, EmbeddedChunks
from app.services.ocr_service import OCRResult
from app.storage import JobArtifacts, ObjectStorage, document_id_for
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
    PreparedDocument,
    ReconcileRequest,
)

log = get_logger(__name__)


def _combine(pages: list[dict]) -> OCRResult:
    return OCRResult(text="\n".join(p["text"] for p in pages), words=[w for p in pages for w in p["words"]])


class CpuActivities:
    def __init__(self, engines: ExtractionEngines, indexer: DocumentIndexer, storage: ObjectStorage, max_pages: int):
        self._engines = engines
        self._indexer = indexer
        self._storage = storage
        self._max_pages = max_pages

    def _pages(self, art: JobArtifacts, page_count: int) -> list[dict]:
        return [art.get_json(art.page(n)) for n in range(page_count)]

    @activity.defn(name=PREPARE_DOCUMENT)
    def prepare_document(self, job_id: str) -> PreparedDocument:
        """Validate the stored PDF; return its page count and content-addressed document ID.
        Invalid input is non-retryable."""
        art = JobArtifacts(self._storage, job_id)
        try:
            document_id = document_id_for(art.storage.get_bytes(art.input_pdf))
            with art.storage.local_path(art.input_pdf) as path:
                pages = self._engines.ocr.page_count(str(path))
        except FileNotFoundError as exc:
            raise ApplicationError(
                "input PDF is missing from storage", type=INVALID_DOCUMENT, non_retryable=True
            ) from exc
        except (fitz.FileDataError, RuntimeError, ValueError) as exc:
            raise ApplicationError(f"not a readable PDF: {exc}", type=INVALID_DOCUMENT, non_retryable=True) from exc
        if pages == 0:
            raise ApplicationError("PDF has no pages", type=INVALID_DOCUMENT, non_retryable=True)
        if pages > self._max_pages:
            raise ApplicationError(
                f"PDF has {pages} pages; the limit is {self._max_pages}", type=INVALID_DOCUMENT, non_retryable=True
            )
        return PreparedDocument(page_count=pages, document_id=document_id)

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
        chunks = self._indexer.embed(_combine(self._pages(art, req.page_count)).text)
        art.put_json(art.chunks, chunks.to_dict())
        return len(chunks.texts)


class IoActivities:
    def __init__(self, repo: JobRepository, storage: ObjectStorage, indexer: DocumentIndexer):
        self._repo = repo
        self._storage = storage
        self._indexer = indexer

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
        chunks = EmbeddedChunks.from_dict(art.get_json(art.chunks))
        return self._indexer.store(req.document_id or req.job_id, chunks, source=req.filename)

    @activity.defn(name=FIND_STALE_JOBS)
    async def find_stale_jobs(self, req: ReconcileRequest) -> list[ExtractRequest]:
        jobs = await self._repo.stale_queued(req.older_than_s, req.limit)
        return [ExtractRequest(j.id, j.query_names, j.filename, req.queues) for j in jobs]
