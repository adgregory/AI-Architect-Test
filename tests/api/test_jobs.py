"""Async jobs API through dependency overrides: submit (202), status, SSE, failure modes."""

import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_container
from app.db import JobStatus
from app.main import app
from app.storage import JobArtifacts, LocalFileStorage
from tests.api.test_routes import PDF, FakeContainer
from tests.fakes import FakeOrchestrator, InMemoryJobRepository

NAMES = [{"first_name": "Richard", "last_name": "Hernandez"}]


class JobsContainer(FakeContainer):
    def __init__(self, tmp_path, orchestrator=None, repository=True):
        super().__init__()
        self.storage = LocalFileStorage(tmp_path)
        self.job_repository = InMemoryJobRepository() if repository else None
        self.orchestrator = orchestrator or FakeOrchestrator()
        self.event_hub = None  # SSE falls back to polling


@pytest.fixture
def make_client():
    def _make(container):
        app.dependency_overrides[get_container] = lambda: container
        return TestClient(app)
    yield _make
    app.dependency_overrides.clear()


def submit(client, content=PDF, filename="minutes.pdf", names=NAMES):
    return client.post("/api/jobs", files={"pdf_file": (filename, content, "application/pdf")},
                       data={"names": json.dumps(names)})


def test_submit_stores_records_and_starts_the_workflow(make_client, tmp_path):
    container = JobsContainer(tmp_path)
    r = submit(make_client(container))
    assert r.status_code == 202
    body = r.json()
    job_id = body["job_id"]
    assert body["status"] == "queued"
    assert body["links"]["events"].endswith(f"/api/jobs/{job_id}/events")

    job = container.job_repository.jobs[job_id]
    assert job.status is JobStatus.QUEUED and job.query_names == NAMES and job.filename == "minutes.pdf"
    assert container.storage.get_bytes(JobArtifacts(container.storage, job_id).input_pdf) == PDF
    assert container.orchestrator.started == [(job_id, "minutes.pdf", NAMES)]


def test_temporal_down_still_accepts_the_job_for_the_reconciler(make_client, tmp_path):
    container = JobsContainer(tmp_path, orchestrator=FakeOrchestrator(fail=True))
    r = submit(make_client(container))
    assert r.status_code == 202
    assert container.job_repository.jobs[r.json()["job_id"]].status is JobStatus.QUEUED


def test_invalid_uploads_are_rejected_before_anything_is_recorded(make_client, tmp_path):
    container = JobsContainer(tmp_path)
    client = make_client(container)
    assert submit(client, filename="notes.txt").status_code == 400
    assert submit(client, names=[{"first_name": "A"}]).status_code == 422
    assert container.job_repository.jobs == {} and container.orchestrator.started == []


def test_status_and_result(make_client, tmp_path):
    container = JobsContainer(tmp_path)
    client = make_client(container)
    job_id = submit(client).json()["job_id"]
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "queued"

    repo: InMemoryJobRepository = container.job_repository
    result = {"extracted_names": [{"name": "Richard Hernandez", "bounding_box": {
        "page_number": 0, "x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0}}],
        "fuzzy_matches": [{"extracted_name": "Richard Hernandez", "matched_name": "Richard Hernandez", "score": 1.0}]}
    repo.jobs[job_id] = replace(repo.jobs[job_id], status=JobStatus.SUCCEEDED, page_count=1, result=result)
    view = client.get(f"/api/jobs/{job_id}").json()
    assert view["status"] == "succeeded" and view["result"] == result and view["page_count"] == 1


def test_unknown_and_malformed_job_ids(make_client, tmp_path):
    client = make_client(JobsContainer(tmp_path))
    assert client.get("/api/jobs/00000000-0000-0000-0000-000000000000").status_code == 404
    assert client.get("/api/jobs/not-a-uuid").status_code == 422
    assert client.get("/api/jobs/00000000-0000-0000-0000-000000000000/events").status_code == 404


def test_events_stream_ends_on_terminal_state(make_client, tmp_path):
    container = JobsContainer(tmp_path)
    client = make_client(container)
    job_id = submit(client).json()["job_id"]
    repo: InMemoryJobRepository = container.job_repository
    repo.jobs[job_id] = replace(repo.jobs[job_id], status=JobStatus.FAILED, error="not a readable PDF")

    with client.stream("GET", f"/api/jobs/{job_id}/events") as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())
    assert body.startswith("event: failed\ndata: ")
    assert json.loads(body.split("data: ", 1)[1])["error"] == "not a readable PDF"


def test_job_store_unavailable_returns_503(make_client, tmp_path):
    client = make_client(JobsContainer(tmp_path, repository=False))
    assert submit(client).status_code == 503
