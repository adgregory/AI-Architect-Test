"""Shared worker process plumbing: logging, Temporal connection with retry, graceful shutdown."""

from __future__ import annotations

import asyncio
import signal
from contextlib import AsyncExitStack

from temporalio.client import Client
from temporalio.worker import Worker

from app.core.config import Settings
from app.core.logging import get_logger
from app.workflows.client import connect

log = get_logger(__name__)


async def connect_with_retry(settings: Settings, attempts: int = 30, delay_s: float = 2.0) -> Client:
    for attempt in range(1, attempts + 1):
        try:
            return await connect(settings)
        except Exception as exc:  # noqa: BLE001 - Temporal may still be starting
            log.warning("temporal.connect_retry", attempt=attempt, error=str(exc))
            await asyncio.sleep(delay_s)
    raise RuntimeError(f"could not connect to Temporal at {settings.temporal_address}")


async def run_until_signalled(workers: list[Worker]) -> None:
    """Run workers until SIGINT/SIGTERM, then let in-flight activities finish (graceful shutdown)."""
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    async with AsyncExitStack() as stack:
        for worker in workers:
            await stack.enter_async_context(worker)
        log.info("worker.started", task_queues=[w.task_queue for w in workers])
        await stop.wait()
        log.info("worker.stopping")
