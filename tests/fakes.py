"""In-memory fakes of the service interfaces, for tests that must not load models or touch the network.

Fakes implement the same Protocols as the real services (asserted in tests/unit/test_fakes.py),
so tests exercise orchestration logic without patching library internals.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import numpy as np

from app.db import Job, JobStatus
from app.services.ocr_service import OCRResult


class FakeOCR:
    """Returns a fixed OCRResult and counts how often each document is read."""

    def __init__(self, result: OCRResult):
        self.result = result
        self.reads: list[str] = []

    def read(self, pdf_path: str) -> OCRResult:
        self.reads.append(pdf_path)
        return self.result

    def page_count(self, pdf_path: str) -> int:
        return max((w["page"] for w in self.result.words), default=-1) + 1

    def read_page(self, pdf_path: str, page_number: int) -> OCRResult:
        self.reads.append(f"{pdf_path}#{page_number}")
        words = [w for w in self.result.words if w["page"] == page_number]
        return OCRResult(text=" ".join(w["word"] for w in words), words=words)

    def extract_text(self, pdf_path: str) -> str:
        return self.read(pdf_path).text

    def get_word_boxes(self, pdf_path: str) -> list[dict]:
        return self.read(pdf_path).words


class FakeNER:
    """Returns the configured names that occur in the text, in text order."""

    def __init__(self, names: list[str]):
        self.names = names

    def extract_names_with_positions(self, text: str) -> list[dict]:
        found = []
        for name in self.names:
            start = text.find(name)
            while start != -1:
                found.append({"name": name, "start_char": start, "end_char": start + len(name), "label": "PERSON"})
                start = text.find(name, start + 1)
        return sorted(found, key=lambda e: e["start_char"])

    def extract_names(self, text: str) -> list[str]:
        return [e["name"] for e in self.extract_names_with_positions(text)]


class FakeEmbeddings:
    """Deterministic bag-of-words hashing embeddings: texts sharing words are similar."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        v = np.zeros(self.dim, dtype=np.float32)
        for token in text.lower().split():
            v[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = np.linalg.norm(v)
        return (v / norm if norm else v).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._vector(query)


@dataclass
class InMemoryVectorStore:
    """Brute-force cosine search over stored vectors."""

    score_threshold: float = 0.0
    points: dict[str, dict] = field(default_factory=dict)
    ensured: int = 0

    def ensure_collection(self) -> None:
        self.ensured += 1

    def upsert(self, texts, vectors, document_id=None, metadata=None) -> str:
        document_id = document_id or f"doc-{len(self.points)}"
        for i, (text, vector) in enumerate(zip(texts, vectors)):
            extra = (metadata or [{}] * len(texts))[i]
            payload = {"text": text, "document_id": document_id, "chunk_index": i, **extra}
            self.points[f"{document_id}:{i}"] = {"vector": np.asarray(vector), "payload": payload}
        return document_id

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        q = np.asarray(query_vector)
        scored = [(float(p["vector"] @ q), p["payload"]["text"]) for p in self.points.values()]
        scored = [s for s in scored if s[0] >= self.score_threshold]
        return [{"text": t, "score": s} for s, t in sorted(scored, reverse=True)[:top_k]]


class FakeLLM:
    """Records prompts and returns a canned answer."""

    def __init__(self, answer: str = "fake answer"):
        self.answer = answer
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


class FakeQdrantClient:
    """Just enough of QdrantClient for QdrantVectorStore."""

    def __init__(self, existing: tuple[str, ...] = ()):
        self.collections = list(existing)
        self.created: list[dict] = []
        self.upserts: list[dict] = []
        self.search_hits: list = []
        self.search_calls: list[dict] = []

    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in self.collections])

    def create_collection(self, **kwargs):
        self.created.append(kwargs)
        self.collections.append(kwargs["collection_name"])

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return self.search_hits


class InMemoryJobRepository:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.stale: list[Job] = []

    async def create(self, job_id, filename, input_key, query_names) -> Job:
        self.jobs[job_id] = Job(job_id, JobStatus.QUEUED, filename, input_key, query_names)
        return self.jobs[job_id]

    async def get(self, job_id):
        return self.jobs.get(job_id)

    def _set(self, job_id, **changes):
        self.jobs[job_id] = replace(self.jobs[job_id], **changes)

    async def mark_running(self, job_id, page_count=None):
        self._set(job_id, status=JobStatus.RUNNING, page_count=page_count, attempts=self.jobs[job_id].attempts + 1)

    async def complete(self, job_id, result):
        self._set(job_id, status=JobStatus.SUCCEEDED, result=result)

    async def fail(self, job_id, error):
        self._set(job_id, status=JobStatus.FAILED, error=error)

    async def stale_queued(self, older_than_s, limit=100):
        return self.stale[:limit]


class FakeOrchestrator:
    """Records workflow starts; can simulate Temporal being unavailable."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.started: list[tuple[str, str, list[dict]]] = []

    async def start_extraction(self, job_id: str, filename: str, query_names: list[dict]) -> None:
        if self.fail:
            raise ConnectionError("temporal unavailable")
        self.started.append((job_id, filename, query_names))


def word(text: str, page: int, x: float, y: float, w: float = 30.0, h: float = 10.0) -> dict:
    return {"word": text, "page": page, "x": x, "y": y, "width": w, "height": h}
