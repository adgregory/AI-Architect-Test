"""PostgresJobRepository and JobEventHub against a real Postgres (docker compose up -d postgres + alembic)."""

import asyncio
import uuid

import psycopg
import pytest
from psycopg_pool import PoolTimeout

from app.core.config import Settings
from app.db import JobEventHub, JobStatus, PostgresJobRepository, create_pool

pytestmark = pytest.mark.integration

SETTINGS = Settings(_env_file=None)
DSN = SETTINGS.database_url.get_secret_value()


def _db_ready() -> bool:
    try:
        with psycopg.connect(DSN, connect_timeout=2) as conn:
            return conn.execute("SELECT to_regclass('public.jobs')").fetchone()[0] is not None
    except psycopg.Error:
        return False


if not _db_ready():
    pytest.skip("Postgres with the jobs schema not available (docker compose up -d postgres && "
                "uv run alembic upgrade head)", allow_module_level=True)


CREATED: list[str] = []


def new_id() -> str:
    job_id = str(uuid.uuid4())
    CREATED.append(job_id)
    return job_id


@pytest.fixture
async def repo():
    pool = create_pool(SETTINGS)
    await pool.open()
    yield PostgresJobRepository(pool)
    # Leave the shared dev database clean (the reconciler would otherwise pick these up).
    async with pool.connection() as conn:
        await conn.execute("DELETE FROM jobs WHERE id = ANY(%s)", ([uuid.UUID(i) for i in CREATED],))
    CREATED.clear()
    await pool.close()


async def test_lifecycle(repo):
    job_id = new_id()
    created = await repo.create(job_id, "memo.pdf", f"jobs/{job_id}/input.pdf", [{"first_name": "A", "last_name": "B"}])
    assert created.status is JobStatus.QUEUED and created.query_names == [{"first_name": "A", "last_name": "B"}]

    await repo.mark_running(job_id, page_count=2)
    running = await repo.get(job_id)
    assert running.status is JobStatus.RUNNING and running.page_count == 2 and running.attempts == 1
    assert running.started_at is not None

    await repo.complete(job_id, {"extracted_names": [], "fuzzy_matches": []})
    done = await repo.get(job_id)
    assert done.status is JobStatus.SUCCEEDED and done.status.terminal
    assert done.result == {"extracted_names": [], "fuzzy_matches": []} and done.finished_at is not None


async def test_fail_records_error(repo):
    job_id = new_id()
    await repo.create(job_id, "bad.pdf", "k", [])
    await repo.fail(job_id, "corrupt PDF")
    job = await repo.get(job_id)
    assert job.status is JobStatus.FAILED and job.error == "corrupt PDF"


async def test_get_missing_returns_none(repo):
    assert await repo.get(new_id()) is None


async def test_stale_queued_only_returns_old_queued_jobs(repo):
    old, fresh, running = new_id(), new_id(), new_id()
    for job_id in (old, fresh, running):
        await repo.create(job_id, "f.pdf", "k", [])
    await repo.mark_running(running)
    async with repo._pool.connection() as conn:
        await conn.execute("UPDATE jobs SET created_at = now() - interval '10 minutes' WHERE id IN (%s, %s)",
                           (uuid.UUID(old), uuid.UUID(running)))
    stale = {j.id for j in await repo.stale_queued(older_than_s=60, limit=1000)}
    assert old in stale and fresh not in stale and running not in stale


async def test_event_hub_delivers_committed_transitions(repo):
    job_id = new_id()
    await repo.create(job_id, "f.pdf", "k", [])
    hub = JobEventHub(DSN)
    await hub.start()
    try:
        async with hub.subscribe(job_id) as events:
            await repo.complete(job_id, {"ok": True})
            assert await asyncio.wait_for(events.get(), timeout=5) == job_id
    finally:
        await hub.stop()


async def test_pool_applies_backpressure():
    settings = Settings(_env_file=None, db_pool_min_size=0, db_pool_max_size=1, db_pool_timeout_s=0.3)
    pool = create_pool(settings)
    await pool.open()
    try:
        async with pool.connection():
            with pytest.raises(PoolTimeout):
                async with pool.connection():
                    pass
    finally:
        await pool.close()


async def test_connections_carry_statement_timeout(repo):
    async with repo._pool.connection() as conn:
        row = await (await conn.execute("SHOW statement_timeout")).fetchone()
    assert row["statement_timeout"] == "15s"
