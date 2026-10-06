"""Workflow/activity payloads. Plain dataclasses with no heavy imports (safe in the workflow sandbox).

Payloads carry identifiers and storage keys only — never document contents.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Activity names (workflows refer to activities by name, so they never import model code).
PREPARE_DOCUMENT = "prepare_document"
OCR_PAGE = "ocr_page"
EXTRACT_AND_MATCH = "extract_and_match"
CHUNK_AND_EMBED = "chunk_and_embed"
MARK_RUNNING = "mark_running"
COMPLETE_JOB = "complete_job"
FAIL_JOB = "fail_job"
UPSERT_CHUNKS = "upsert_chunks"
FIND_STALE_JOBS = "find_stale_jobs"

INVALID_DOCUMENT = "InvalidDocument"  # non-retryable error type


@dataclass(frozen=True)
class TaskQueues:
    workflows: str
    cpu: str
    io: str


@dataclass(frozen=True)
class ExtractRequest:
    job_id: str
    query_names: list[dict]
    filename: str
    queues: TaskQueues


@dataclass(frozen=True)
class PageTask:
    job_id: str
    page_number: int


@dataclass(frozen=True)
class MarkRunning:
    job_id: str
    page_count: int


@dataclass(frozen=True)
class ExtractTask:
    job_id: str
    page_count: int
    query_names: list[dict]


@dataclass(frozen=True)
class ExtractSummary:
    names: int
    matches: int


@dataclass(frozen=True)
class FailRequest:
    job_id: str
    error: str


@dataclass(frozen=True)
class IndexRequest:
    job_id: str
    page_count: int
    filename: str
    queues: TaskQueues


@dataclass(frozen=True)
class ReconcileRequest:
    older_than_s: int
    queues: TaskQueues
    limit: int = 100


@dataclass(frozen=True)
class ExtractOutcome:
    job_id: str
    pages: int
    names: int
    matches: int


@dataclass(frozen=True)
class ReconcileOutcome:
    restarted: list[str] = field(default_factory=list)
