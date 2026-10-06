"""Workflow behaviour on Temporal's time-skipping test server, with the real activity classes
wired to in-memory fakes (no models, no database)."""

import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack

import pytest
from temporalio.client import WorkflowFailureError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from app.db import JobStatus
from app.services.bbox_service import ConsecutiveWordNameLocator
from app.services.extraction_service import ExtractionEngines
from app.services.fuzzy_service import TokenSortNameMatcher
from app.services.ner_service import PersonNameNormalizer
from app.services.ocr_service import OCRResult
from app.services.rag_service import TextChunker
from app.storage import JobArtifacts, LocalFileStorage, document_id_for
from app.workflows.activities import CpuActivities, IoActivities
from app.workflows.models import ExtractRequest, ReconcileRequest, TaskQueues
from app.workflows.workflows import ExtractNamesWorkflow, IndexDocumentWorkflow, ReconcileQueuedJobsWorkflow
from tests.fakes import FakeEmbeddings, FakeNER, FakeOCR, InMemoryJobRepository, InMemoryVectorStore, word

QUEUES = TaskQueues(workflows="wf", cpu="cpu", io="io")
OCR = OCRResult(
    text="",
    words=[
        word("Richard", 0, 60, 166),
        word("Hernandez", 0, 95, 166),
        word("met", 0, 140, 166),
        word("Jennifer", 1, 10, 40),
        word("Liu", 1, 50, 40),
    ],
)


class FlakyOCR(FakeOCR):
    """Fails the first read of page 1, then succeeds (a transient worker error)."""

    def __init__(self, result):
        super().__init__(result)
        self.failed = False

    def read_page(self, pdf_path, page_number):
        if page_number == 1 and not self.failed:
            self.failed = True
            raise ConnectionError("transient")
        return super().read_page(pdf_path, page_number)


class BrokenPdfOCR(FakeOCR):
    def page_count(self, pdf_path):
        raise ValueError("cannot open broken document")


@pytest.fixture
async def env():
    async with await WorkflowEnvironment.start_time_skipping() as env:
        yield env


class Harness:
    def __init__(self, tmp_path, ocr):
        self.repo = InMemoryJobRepository()
        self.storage = LocalFileStorage(tmp_path)
        self.store = InMemoryVectorStore()
        self.ocr = ocr
        engines = ExtractionEngines(
            ocr,
            FakeNER(["Richard Hernandez", "Jennifer Liu"]),
            ConsecutiveWordNameLocator(),
            TokenSortNameMatcher(),
            PersonNameNormalizer(),
        )
        self.cpu = CpuActivities(engines, FakeEmbeddings(), self.storage, TextChunker(), chunk_size=50, max_pages=10)
        self.io = IoActivities(self.repo, self.storage, self.store)

    async def new_job(self, query_names) -> ExtractRequest:
        job_id = str(uuid.uuid4())
        art = JobArtifacts(self.storage, job_id)
        self.storage.put_bytes(art.input_pdf, b"%PDF-1.7 fake")
        await self.repo.create(job_id, "doc.pdf", art.input_pdf, query_names)
        return ExtractRequest(job_id, query_names, "doc.pdf", QUEUES)

    def workers(self, client, stack: AsyncExitStack, executor):
        cpu = [self.cpu.prepare_document, self.cpu.ocr_page, self.cpu.extract_and_match, self.cpu.chunk_and_embed]
        io = [
            self.io.mark_running,
            self.io.complete_job,
            self.io.fail_job,
            self.io.upsert_chunks,
            self.io.find_stale_jobs,
        ]
        return [
            Worker(
                client,
                task_queue=QUEUES.workflows,
                workflows=[ExtractNamesWorkflow, IndexDocumentWorkflow, ReconcileQueuedJobsWorkflow],
            ),
            Worker(client, task_queue=QUEUES.cpu, activities=cpu, activity_executor=executor),
            Worker(client, task_queue=QUEUES.io, activities=io),
        ]


async def run_workers(env, harness, coro):
    with ThreadPoolExecutor(4) as executor:
        async with AsyncExitStack() as stack:
            for w in harness.workers(env.client, stack, executor):
                await stack.enter_async_context(w)
            return await coro()


async def test_extracts_pages_in_parallel_completes_and_indexes(env, tmp_path):
    h = Harness(tmp_path, FakeOCR(OCR))
    req = await h.new_job([{"first_name": "Jenifer", "last_name": "Liu"}])

    async def scenario():
        outcome = await env.client.execute_workflow(
            ExtractNamesWorkflow.run, req, id=req.job_id, task_queue=QUEUES.workflows
        )
        indexed = await env.client.get_workflow_handle(f"index-{document_id_for(b'%PDF-1.7 fake')}").result()
        return outcome, indexed

    outcome, indexed = await run_workers(env, h, scenario)

    job = h.repo.jobs[req.job_id]
    assert job.status is JobStatus.SUCCEEDED and job.page_count == 2 and job.attempts == 1
    assert [n["name"] for n in job.result["extracted_names"]] == ["Richard Hernandez", "Jennifer Liu"]
    assert job.result["extracted_names"][1]["bounding_box"]["page_number"] == 1
    assert job.result["fuzzy_matches"][0]["extracted_name"] == "Jennifer Liu"
    assert (outcome.pages, outcome.names, outcome.matches) == (2, 2, 1)
    assert sorted(r.rsplit("#", 1)[1] for r in h.ocr.reads if "#" in r) == ["0", "1"]  # one OCR per page
    assert indexed >= 1 and h.store.points  # child workflow indexed the document...
    assert {p["payload"]["document_id"] for p in h.store.points.values()} == {document_id_for(b"%PDF-1.7 fake")}
    # ...under its content-addressed ID


async def test_invalid_document_fails_fast_and_marks_job_failed(env, tmp_path):
    h = Harness(tmp_path, BrokenPdfOCR(OCR))
    req = await h.new_job([])

    async def scenario():
        with pytest.raises(WorkflowFailureError):
            await env.client.execute_workflow(ExtractNamesWorkflow.run, req, id=req.job_id, task_queue=QUEUES.workflows)

    await run_workers(env, h, scenario)
    job = h.repo.jobs[req.job_id]
    assert job.status is JobStatus.FAILED and "not a readable PDF" in job.error
    assert job.attempts == 0  # never marked running: failed at validation, without retries


async def test_transient_page_failure_is_retried(env, tmp_path):
    h = Harness(tmp_path, FlakyOCR(OCR))
    req = await h.new_job([])

    async def scenario():
        return await env.client.execute_workflow(
            ExtractNamesWorkflow.run, req, id=req.job_id, task_queue=QUEUES.workflows
        )

    outcome = await run_workers(env, h, scenario)
    assert h.ocr.failed and outcome.names == 2
    assert h.repo.jobs[req.job_id].status is JobStatus.SUCCEEDED


async def test_reconciler_restarts_stale_jobs_once(env, tmp_path):
    h = Harness(tmp_path, FakeOCR(OCR))
    req = await h.new_job([])
    h.repo.stale = [h.repo.jobs[req.job_id]]

    async def scenario():
        first = await env.client.execute_workflow(
            ReconcileQueuedJobsWorkflow.run,
            ReconcileRequest(60, QUEUES),
            id=f"reconcile-{uuid.uuid4()}",
            task_queue=QUEUES.workflows,
        )
        await env.client.get_workflow_handle(req.job_id).result()
        return first

    first = await run_workers(env, h, scenario)
    assert first.restarted == [req.job_id]
    assert h.repo.jobs[req.job_id].status is JobStatus.SUCCEEDED


async def test_same_pdf_submitted_twice_is_indexed_once(env, tmp_path):
    h = Harness(tmp_path, FakeOCR(OCR))
    first, second = await h.new_job([]), await h.new_job([])  # identical bytes, different jobs
    document_id = document_id_for(b"%PDF-1.7 fake")

    async def scenario():
        for req in (first, second):
            await env.client.execute_workflow(ExtractNamesWorkflow.run, req, id=req.job_id, task_queue=QUEUES.workflows)
            await env.client.get_workflow_handle(f"index-{document_id}").result()

    await run_workers(env, h, scenario)
    assert {p["payload"]["document_id"] for p in h.store.points.values()} == {document_id}
    assert len(h.store.points) == len({p["payload"]["chunk_index"] for p in h.store.points.values()})
