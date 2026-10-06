"""Postgres access for jobs: pooled async repository and a LISTEN/NOTIFY event hub."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable

from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from app.core.config import Settings
from app.core.logging import get_logger

log = get_logger(__name__)

JOB_EVENTS_CHANNEL = "job_events"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in (JobStatus.SUCCEEDED, JobStatus.FAILED)


@dataclass(frozen=True)
class Job:
    id: str
    status: JobStatus
    filename: str
    input_key: str
    query_names: list[dict]
    page_count: int | None = None
    result: dict | None = None
    error: str | None = None
    attempts: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @classmethod
    def from_row(cls, row: dict) -> Job:
        return cls(**{**row, "id": str(row["id"]), "status": JobStatus(row["status"])})


@runtime_checkable
class JobRepository(Protocol):
    async def create(self, job_id: str, filename: str, input_key: str, query_names: list[dict]) -> Job: ...

    async def get(self, job_id: str) -> Job | None: ...

    async def mark_running(self, job_id: str, page_count: int | None = None) -> None: ...

    async def complete(self, job_id: str, result: dict) -> None: ...

    async def fail(self, job_id: str, error: str) -> None: ...

    async def stale_queued(self, older_than_s: int, limit: int = 100) -> list[Job]: ...


Pool = AsyncConnectionPool[AsyncConnection[DictRow]]


def create_pool(settings: Settings) -> Pool:
    """Per-process pool. Sized, bounded and recycled from config; opened by the caller."""

    async def configure(conn: AsyncConnection[DictRow]) -> None:
        await conn.execute(f"SET statement_timeout = {int(settings.db_statement_timeout_ms)}")
        await conn.execute("SET idle_in_transaction_session_timeout = 30000")
        await conn.commit()

    return AsyncConnectionPool(
        conninfo=settings.database_url.get_secret_value(),
        connection_class=AsyncConnection[DictRow],
        kwargs={"row_factory": dict_row},
        min_size=settings.db_pool_min_size,
        max_size=settings.db_pool_max_size,
        timeout=settings.db_pool_timeout_s,  # acquire wait → PoolTimeout (backpressure)
        max_waiting=settings.db_pool_max_waiting,  # beyond this, reject immediately (TooManyRequests)
        max_lifetime=30 * 60,
        max_idle=5 * 60,
        configure=configure,
        check=AsyncConnectionPool.check_connection,  # never hand out a dead connection
        open=False,
        name="app",
    )


class PostgresJobRepository:
    """Short transactions only: a connection is held for one statement, never across OCR/NER."""

    def __init__(self, pool: Pool):
        self._pool = pool

    async def create(self, job_id, filename, input_key, query_names) -> Job:
        async with self._pool.connection() as conn:
            row = await (
                await conn.execute(
                    """INSERT INTO jobs (id, filename, input_key, query_names)
                   VALUES (%s, %s, %s, %s) RETURNING *""",
                    (uuid.UUID(job_id), filename, input_key, Jsonb(query_names)),
                )
            ).fetchone()
        assert row is not None  # INSERT ... RETURNING always yields the row
        return Job.from_row(row)

    async def get(self, job_id: str) -> Job | None:
        async with self._pool.connection() as conn:
            row = await (await conn.execute("SELECT * FROM jobs WHERE id = %s", (uuid.UUID(job_id),))).fetchone()
        return Job.from_row(row) if row else None

    async def mark_running(self, job_id: str, page_count: int | None = None) -> None:
        await self._transition(
            job_id,
            """UPDATE jobs SET status = 'running', attempts = attempts + 1,
                      page_count = COALESCE(%s, page_count),
                      started_at = COALESCE(started_at, now()), updated_at = now()
               WHERE id = %s AND status IN ('queued', 'running')""",
            (page_count, uuid.UUID(job_id)),
        )

    async def complete(self, job_id: str, result: dict) -> None:
        await self._transition(
            job_id,
            """UPDATE jobs SET status = 'succeeded', result = %s, error = NULL,
                      finished_at = now(), updated_at = now()
               WHERE id = %s""",
            (Jsonb(result), uuid.UUID(job_id)),
        )

    async def fail(self, job_id: str, error: str) -> None:
        await self._transition(
            job_id,
            """UPDATE jobs SET status = 'failed', error = %s, finished_at = now(), updated_at = now()
               WHERE id = %s""",
            (error[:2000], uuid.UUID(job_id)),
        )

    async def stale_queued(self, older_than_s: int, limit: int = 100) -> list[Job]:
        async with self._pool.connection() as conn:
            rows = await (
                await conn.execute(
                    """SELECT * FROM jobs WHERE status = 'queued'
                   AND created_at < now() - make_interval(secs => %s)
                   ORDER BY created_at LIMIT %s""",
                    (older_than_s, limit),
                )
            ).fetchall()
        return [Job.from_row(r) for r in rows]

    async def _transition(self, job_id: str, sql: str, params: tuple) -> None:
        # The NOTIFY is part of the same transaction: listeners hear about committed state only.
        async with self._pool.connection() as conn, conn.transaction():
            await conn.execute(sql, params)
            await conn.execute("SELECT pg_notify(%s, %s)", (JOB_EVENTS_CHANNEL, job_id))


class JobEventHub:
    """One dedicated LISTEN connection per process (LISTEN is session-scoped, so it lives
    outside the pool), fanning notifications out to per-job subscribers."""

    def __init__(self, conninfo: str):
        self._conninfo = conninfo
        self._subscribers: dict[str, set[asyncio.Queue]] = {}
        self._task: asyncio.Task | None = None
        self._conn: AsyncConnection | None = None

    async def start(self) -> None:
        self._conn = await AsyncConnection.connect(self._conninfo, autocommit=True)
        await self._conn.execute(f"LISTEN {JOB_EVENTS_CHANNEL}")
        self._task = asyncio.create_task(self._pump(), name="job-event-hub")

    async def _pump(self) -> None:
        assert self._conn is not None
        try:
            async for note in self._conn.notifies():
                for queue in list(self._subscribers.get(note.payload, ())):
                    queue.put_nowait(note.payload)
        except asyncio.CancelledError:
            raise
        except Exception:  # connection lost: subscribers fall back to polling
            log.exception("job_event_hub.stopped")

    @contextlib.asynccontextmanager
    async def subscribe(self, job_id: str) -> AsyncIterator[asyncio.Queue]:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(job_id, set()).add(queue)
        try:
            yield queue
        finally:
            self._subscribers[job_id].discard(queue)
            if not self._subscribers[job_id]:
                del self._subscribers[job_id]

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if self._conn:
            await self._conn.close()
