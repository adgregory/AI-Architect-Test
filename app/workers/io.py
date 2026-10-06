"""IO worker: hosts the workflows and the database / vector-store activities.

Sole owner of the worker-side Postgres pool (CPU workers hold no connections), so the
connection count doesn't grow with CPU workers. Also creates the reconciler schedule.

    python -m app.workers.io
"""

from __future__ import annotations

import asyncio

from temporalio.worker import Worker

from app.core.config import get_settings
from app.core.container import Container
from app.core.logging import configure_logging, get_logger
from app.db import PostgresJobRepository, create_pool
from app.workers.runtime import connect_with_retry, run_until_signalled
from app.workflows.activities import IoActivities
from app.workflows.client import ensure_reconcile_schedule
from app.workflows.workflows import ExtractNamesWorkflow, IndexDocumentWorkflow, ReconcileQueuedJobsWorkflow

log = get_logger(__name__)


async def main() -> None:
    settings = get_settings()
    configure_logging(settings)
    container = Container(settings)
    pool = create_pool(settings)
    await pool.open()
    try:
        activities = IoActivities(PostgresJobRepository(pool), container.storage, container.vector_store)
        client = await connect_with_retry(settings)
        await ensure_reconcile_schedule(client, settings)
        workers = [
            Worker(
                client,
                task_queue=settings.workflow_task_queue,
                workflows=[ExtractNamesWorkflow, IndexDocumentWorkflow, ReconcileQueuedJobsWorkflow],
            ),
            Worker(
                client,
                task_queue=settings.io_task_queue,
                activities=[
                    activities.mark_running,
                    activities.complete_job,
                    activities.fail_job,
                    activities.upsert_chunks,
                    activities.find_stale_jobs,
                ],
            ),
        ]
        await run_until_signalled(workers)
    finally:
        await pool.close()
        container.close()


if __name__ == "__main__":
    asyncio.run(main())
