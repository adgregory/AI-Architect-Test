"""Temporal workflows. Deterministic orchestration only — all I/O and models live in activities."""

from __future__ import annotations

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import (
    ActivityError,
    ApplicationError,
    ChildWorkflowError,
    WorkflowAlreadyStartedError,
)
from temporalio.workflow import ParentClosePolicy

with workflow.unsafe.imports_passed_through():
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
        ExtractOutcome,
        ExtractRequest,
        ExtractSummary,
        ExtractTask,
        FailRequest,
        IndexRequest,
        MarkRunning,
        PageTask,
        ReconcileOutcome,
        ReconcileRequest,
    )

# Transient failures (timeouts, a busy database, a crashed worker) retry with backoff;
# invalid input fails fast.
CPU_RETRY = RetryPolicy(initial_interval=timedelta(seconds=1), backoff_coefficient=2.0,
                        maximum_interval=timedelta(seconds=30), maximum_attempts=3,
                        non_retryable_error_types=[INVALID_DOCUMENT])
IO_RETRY = RetryPolicy(initial_interval=timedelta(seconds=1), backoff_coefficient=2.0,
                       maximum_interval=timedelta(seconds=20), maximum_attempts=10)


def _root_message(err: BaseException) -> str:
    while isinstance(err, (ActivityError, ChildWorkflowError)) and err.cause is not None:
        err = err.cause
    return str(err) or type(err).__name__


@workflow.defn
class ExtractNamesWorkflow:
    """prepare → mark running → OCR every page in parallel → NER + boxes + matching → complete.

    The workflow ID is the job ID, so a job can never run twice concurrently. Any terminal
    failure marks the job failed (clients are never left waiting). Indexing for RAG runs as
    an abandoned child workflow so it can't delay or fail extraction.
    """

    @workflow.run
    async def run(self, req: ExtractRequest) -> ExtractOutcome:
        q = req.queues
        try:
            pages: int = await workflow.execute_activity(
                PREPARE_DOCUMENT, req.job_id, task_queue=q.cpu,
                start_to_close_timeout=timedelta(seconds=60), retry_policy=CPU_RETRY)
            await workflow.execute_activity(
                MARK_RUNNING, MarkRunning(req.job_id, pages), task_queue=q.io,
                start_to_close_timeout=timedelta(seconds=15), retry_policy=IO_RETRY)
            await asyncio.gather(*(
                workflow.execute_activity(
                    OCR_PAGE, PageTask(req.job_id, n), task_queue=q.cpu,
                    start_to_close_timeout=timedelta(minutes=3), heartbeat_timeout=timedelta(seconds=30),
                    retry_policy=CPU_RETRY)
                for n in range(pages)
            ))
            summary: ExtractSummary = await workflow.execute_activity(
                EXTRACT_AND_MATCH, ExtractTask(req.job_id, pages, req.query_names), task_queue=q.cpu,
                start_to_close_timeout=timedelta(minutes=2), retry_policy=CPU_RETRY,
                result_type=ExtractSummary)
            await workflow.execute_activity(
                COMPLETE_JOB, req.job_id, task_queue=q.io,
                start_to_close_timeout=timedelta(seconds=15), retry_policy=IO_RETRY)
        except ActivityError as err:
            message = _root_message(err)
            await workflow.execute_activity(
                FAIL_JOB, FailRequest(req.job_id, message), task_queue=q.io,
                start_to_close_timeout=timedelta(seconds=15), retry_policy=IO_RETRY)
            raise ApplicationError(f"job {req.job_id} failed: {message}", non_retryable=True) from err

        await workflow.start_child_workflow(
            IndexDocumentWorkflow.run, IndexRequest(req.job_id, pages, req.filename, q),
            id=f"index-{req.job_id}", task_queue=q.workflows,
            parent_close_policy=ParentClosePolicy.ABANDON)
        return ExtractOutcome(req.job_id, pages, summary.names, summary.matches)


@workflow.defn
class IndexDocumentWorkflow:
    """Chunk + embed the OCR'd pages (cpu), then upsert into the vector store (io)."""

    @workflow.run
    async def run(self, req: IndexRequest) -> int:
        await workflow.execute_activity(
            CHUNK_AND_EMBED, req, task_queue=req.queues.cpu,
            start_to_close_timeout=timedelta(minutes=5), retry_policy=CPU_RETRY)
        return await workflow.execute_activity(
            UPSERT_CHUNKS, req, task_queue=req.queues.io,
            start_to_close_timeout=timedelta(seconds=60), retry_policy=IO_RETRY)


@workflow.defn
class ReconcileQueuedJobsWorkflow:
    """Run on a schedule: re-start jobs left `queued` (the API's direct start failed).
    Starting with workflow_id = job_id makes this idempotent."""

    @workflow.run
    async def run(self, req: ReconcileRequest) -> ReconcileOutcome:
        stale: list[ExtractRequest] = await workflow.execute_activity(
            FIND_STALE_JOBS, req, task_queue=req.queues.io,
            start_to_close_timeout=timedelta(seconds=30), retry_policy=IO_RETRY,
            result_type=list[ExtractRequest])
        restarted = []
        for job in stale:
            try:
                await workflow.start_child_workflow(
                    ExtractNamesWorkflow.run, job, id=job.job_id, task_queue=req.queues.workflows,
                    parent_close_policy=ParentClosePolicy.ABANDON)
                restarted.append(job.job_id)
            except WorkflowAlreadyStartedError:
                pass  # already running: nothing to do
        if restarted:
            workflow.logger.info("reconciler restarted %d job(s)", len(restarted))
        return ReconcileOutcome(restarted)
