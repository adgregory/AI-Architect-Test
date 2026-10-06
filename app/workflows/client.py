"""Starting workflows from the API, and the reconciler schedule."""

from __future__ import annotations

from datetime import timedelta
from typing import Protocol, runtime_checkable

from temporalio.client import (
    Client,
    Schedule,
    ScheduleActionStartWorkflow,
    ScheduleAlreadyRunningError,
    ScheduleIntervalSpec,
    SchedulePolicy,
    ScheduleOverlapPolicy,
    ScheduleSpec,
)
from temporalio.exceptions import WorkflowAlreadyStartedError

from app.core.config import Settings
from app.core.logging import get_logger
from app.workflows.models import ExtractRequest, ReconcileRequest, TaskQueues
from app.workflows.workflows import ExtractNamesWorkflow, ReconcileQueuedJobsWorkflow

log = get_logger(__name__)
RECONCILE_SCHEDULE_ID = "reconcile-queued-jobs"


def task_queues(settings: Settings) -> TaskQueues:
    return TaskQueues(settings.workflow_task_queue, settings.cpu_task_queue, settings.io_task_queue)


async def connect(settings: Settings) -> Client:
    return await Client.connect(settings.temporal_address, namespace=settings.temporal_namespace)


@runtime_checkable
class JobOrchestrator(Protocol):
    async def start_extraction(self, job_id: str, filename: str, query_names: list[dict]) -> None: ...


class TemporalJobOrchestrator:
    """Starts ExtractNamesWorkflow with workflow_id = job_id (idempotent)."""

    def __init__(self, client: Client, settings: Settings):
        self._client = client
        self._queues = task_queues(settings)

    async def start_extraction(self, job_id: str, filename: str, query_names: list[dict]) -> None:
        try:
            await self._client.start_workflow(
                ExtractNamesWorkflow.run, ExtractRequest(job_id, query_names, filename, self._queues),
                id=job_id, task_queue=self._queues.workflows,
            )
        except WorkflowAlreadyStartedError:
            log.info("workflow.already_started", job_id=job_id)


class LazyTemporalOrchestrator:
    """Connects on first use and reconnects after failures, so the API starts (and keeps
    serving) when Temporal is down; failed starts are picked up by the reconciler."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._delegate: TemporalJobOrchestrator | None = None

    async def start_extraction(self, job_id: str, filename: str, query_names: list[dict]) -> None:
        if self._delegate is None:
            self._delegate = TemporalJobOrchestrator(await connect(self._settings), self._settings)
        try:
            await self._delegate.start_extraction(job_id, filename, query_names)
        except Exception:
            self._delegate = None  # reconnect next time
            raise


async def ensure_reconcile_schedule(client: Client, settings: Settings) -> None:
    """Create the reconciler schedule once (idempotent across worker restarts)."""
    try:
        await client.create_schedule(
            RECONCILE_SCHEDULE_ID,
            Schedule(
                action=ScheduleActionStartWorkflow(
                    ReconcileQueuedJobsWorkflow.run,
                    ReconcileRequest(settings.reconcile_stale_after_s, task_queues(settings)),
                    id="reconcile-queued-jobs-run",
                    task_queue=settings.workflow_task_queue,
                ),
                spec=ScheduleSpec(intervals=[ScheduleIntervalSpec(every=timedelta(seconds=settings.reconcile_interval_s))]),
                policy=SchedulePolicy(overlap=ScheduleOverlapPolicy.SKIP),
            ),
        )
        log.info("reconcile_schedule.created", every_s=settings.reconcile_interval_s)
    except ScheduleAlreadyRunningError:
        log.info("reconcile_schedule.exists")
