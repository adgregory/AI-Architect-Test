"""CPU worker: OCR, NER, matching and embedding activities. Holds no database connections.

    python -m app.workers.cpu
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from temporalio.worker import Worker

from app.core.config import get_settings
from app.core.container import Container
from app.core.logging import configure_logging, get_logger
from app.workers.runtime import connect_with_retry, run_until_signalled
from app.workflows.activities import CpuActivities

log = get_logger(__name__)


async def main() -> None:
    settings = get_settings()
    configure_logging(settings)
    container = Container(settings)
    await asyncio.to_thread(container.warm_up)  # load models once, before polling for work
    activities = CpuActivities(
        container.extraction_engines, container.embeddings, container.storage, container.chunker,
        chunk_size=settings.chunk_size, max_pages=settings.max_pages,
    )
    client = await connect_with_retry(settings)
    with ThreadPoolExecutor(max_workers=settings.cpu_worker_concurrency, thread_name_prefix="cpu-activity") as pool:
        worker = Worker(
            client,
            task_queue=settings.cpu_task_queue,
            activities=[activities.prepare_document, activities.ocr_page,
                        activities.extract_and_match, activities.chunk_and_embed],
            activity_executor=pool,
            max_concurrent_activities=settings.cpu_worker_concurrency,
        )
        await run_until_signalled([worker])


if __name__ == "__main__":
    asyncio.run(main())
