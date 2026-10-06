"""Asynchronous extraction jobs: submit (202), poll, and stream status (SSE)."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse

from app.api.deps import get_app_settings, get_container
from app.api.sse import sse_comment, sse_event
from app.api.uploads import parse_names, read_pdf_upload
from app.core.config import Settings
from app.core.container import Container
from app.core.logging import get_logger
from app.db import Job, JobRepository
from app.models.schemas import JobAccepted, JobView
from app.storage import JobArtifacts

router = APIRouter()
log = get_logger(__name__)

SSE_POLL_S = 5.0  # re-check the database even without a notification (hub down / missed event)
SSE_MAX_DURATION_S = 600  # clients reconnect after this


def get_job_repository(container: Container = Depends(get_container)) -> JobRepository:
    if container.job_repository is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "job store unavailable")
    return container.job_repository


def to_view(job: Job) -> JobView:
    return JobView(
        job_id=job.id,
        status=job.status.value,
        filename=job.filename,
        page_count=job.page_count,
        attempts=job.attempts,
        error=job.error,
        result=job.result,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


@router.post("/jobs", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
async def submit_job(
    request: Request,
    pdf_file: UploadFile = File(...),
    names: str = Form(...),
    container: Container = Depends(get_container),
    repo: JobRepository = Depends(get_job_repository),
    settings: Settings = Depends(get_app_settings),
) -> JobAccepted:
    """Validate and store the PDF, record the job, start the workflow directly, return 202.

    If the workflow can't be started now (Temporal unavailable), the job stays `queued`
    and the reconciler starts it later — the client still gets its job ID."""
    content = read_pdf_upload(pdf_file, settings.max_upload_bytes)
    query_names = [q.model_dump() for q in parse_names(names)]
    job_id = str(uuid.uuid4())
    art = JobArtifacts(container.storage, job_id)

    await asyncio.to_thread(container.storage.put_bytes, art.input_pdf, content)
    await repo.create(job_id, pdf_file.filename or "document.pdf", art.input_pdf, query_names)
    try:
        if container.orchestrator is None:
            raise RuntimeError("orchestrator not configured")
        await container.orchestrator.start_extraction(job_id, pdf_file.filename or "document.pdf", query_names)
    except Exception as exc:  # noqa: BLE001 - the reconciler will start it
        log.warning("workflow.start_deferred", job_id=job_id, error=str(exc))

    base = str(request.url_for("get_job", job_id=job_id))
    return JobAccepted(job_id=job_id, status="queued", links={"self": base, "events": f"{base}/events"})


@router.get("/jobs/{job_id}", response_model=JobView, name="get_job")
async def get_job(job_id: uuid.UUID, repo: JobRepository = Depends(get_job_repository)) -> JobView:
    job = await repo.get(str(job_id))
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    return to_view(job)


@router.get("/jobs/{job_id}/events")
async def job_events(
    job_id: uuid.UUID,
    container: Container = Depends(get_container),
    repo: JobRepository = Depends(get_job_repository),
) -> StreamingResponse:
    """Server-sent events: the job's state now, then on every change, until it is terminal.

    Woken by Postgres NOTIFY through the process's single LISTEN connection; also re-reads
    the job every few seconds, so a missed notification or a down hub only adds latency."""
    job_key = str(job_id)
    if await repo.get(job_key) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")

    async def stream() -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + SSE_MAX_DURATION_S
        last = None
        hub = container.event_hub
        if hub is None:
            subscription, events = None, None
        else:
            subscription = hub.subscribe(job_key)
            events = await subscription.__aenter__()
        try:
            while loop.time() < deadline:
                job = await repo.get(job_key)
                if job is None:  # deleted while streaming
                    return
                view = to_view(job).model_dump(mode="json")
                if view != last:
                    yield sse_event(view["status"], view)
                    last = view
                if job.status.terminal:
                    return
                try:
                    if events is not None:
                        await asyncio.wait_for(events.get(), timeout=SSE_POLL_S)
                    else:
                        await asyncio.sleep(SSE_POLL_S)
                except TimeoutError:
                    yield sse_comment("keep-alive")
        finally:
            if subscription is not None:
                await subscription.__aexit__(None, None, None)

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
