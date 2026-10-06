"""API behaviour through FastAPI dependency overrides — no patching of module names.

The real app is used; only `get_container` is overridden with a container of in-memory fakes.
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_container
from app.core.config import Settings
from app.core.factories import MissingEngineError
from app.main import app
from app.services.bbox_service import ConsecutiveWordNameLocator
from app.services.extraction_service import ExtractionEngines, ExtractionSession
from app.services.fuzzy_service import TokenSortNameMatcher
from app.services.indexing_service import DocumentIndexer
from app.services.ner_service import PersonNameNormalizer
from app.services.ocr_service import OCRResult
from app.services.rag_service import RAGService, TextChunker
from tests.fakes import FakeEmbeddings, FakeLLM, FakeNER, FakeOCR, InMemoryVectorStore, word

PDF = b"%PDF-1.7 test document"
OCR = OCRResult(
    text="Minutes\nRichard Hernandez opened the meeting.\nDr. Aisha Patel reviewed costs.",
    words=[
        word("Richard", 0, 60, 166, w=40),
        word("Hernandez", 0, 102, 166, w=44),
        word("Dr.", 1, 10, 50, w=12),
        word("Aisha", 1, 25, 50),
        word("Patel", 1, 58, 50),
    ],
)


class FakeContainer:
    """Same surface the routes use on the real Container, wired with fakes."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings(_env_file=None)
        self.ocr = FakeOCR(OCR)
        self.ner = FakeNER(["Richard Hernandez", "Dr. Aisha Patel"])
        self.embeddings = FakeEmbeddings()
        self.vector_store = InMemoryVectorStore(score_threshold=0.1)
        self.llm = FakeLLM("Richard Hernandez chaired the meeting.")
        self.chunker = TextChunker()
        self.indexer = DocumentIndexer(self.chunker, self.settings.chunk_size, self.embeddings, self.vector_store)
        self.orchestrator = None  # no Temporal: /api/extract skips background indexing
        self.storage = None
        self.rag = RAGService(self.embeddings.embed_query, self.vector_store.search, self.llm, top_k=2)
        self._engines = ExtractionEngines(
            self.ocr,
            self.ner,
            ConsecutiveWordNameLocator(),
            TokenSortNameMatcher(self.settings.similarity_threshold),
            PersonNameNormalizer(),
        )

    def extraction_session(self) -> ExtractionSession:
        return ExtractionSession(lambda: self._engines)


@pytest.fixture
def container():
    return FakeContainer()


@pytest.fixture
def client(container):
    app.dependency_overrides[get_container] = lambda: container
    yield TestClient(app)
    app.dependency_overrides.clear()


def post_extract(client, names, content=PDF, filename="doc.pdf"):
    return client.post(
        "/api/extract",
        files={"pdf_file": (filename, content, "application/pdf")},
        data={"names": json.dumps(names) if not isinstance(names, str) else names},
    )


class TestExtract:
    def test_returns_names_boxes_pages_and_matches(self, client, container):
        r = post_extract(
            client,
            [
                {"first_name": "Richard", "last_name": "Hernandez"},
                {"first_name": "Aisha", "last_name": "Patl"},
                {"first_name": "Zara", "last_name": "Xu"},
            ],
        )
        assert r.status_code == 200
        body = r.json()
        assert body["extracted_names"] == [
            {
                "name": "Richard Hernandez",
                "bounding_box": {"page_number": 0, "x": 60.0, "y": 166.0, "width": 86.0, "height": 10.0},
            },
            {
                "name": "Aisha Patel",
                "bounding_box": {"page_number": 1, "x": 25.0, "y": 50.0, "width": 63.0, "height": 10.0},
            },
        ]
        assert [(m["matched_name"], m["extracted_name"]) for m in body["fuzzy_matches"]] == [
            ("Richard Hernandez", "Richard Hernandez"),
            ("Aisha Patl", "Aisha Patel"),
        ]
        assert len(container.ocr.reads) == 1  # one OCR pass per request

    @pytest.mark.parametrize("filename, content", [("notes.txt", PDF), ("doc.pdf", b"not a pdf")])
    def test_rejects_non_pdf(self, client, filename, content):
        assert post_extract(client, [], content=content, filename=filename).status_code == 400

    @pytest.mark.parametrize("names", ["not json", '[{"first_name": "A"}]', '[{"first_name": "", "last_name": "B"}]'])
    def test_rejects_invalid_names(self, client, names):
        assert post_extract(client, names).status_code == 422

    def test_rejects_oversized_uploads(self, client, container):
        container.settings = Settings(_env_file=None, max_upload_mb=1)
        r = post_extract(client, [], content=PDF + b"0" * (2 * 2**20))
        assert r.status_code == 413

    def test_missing_engine_returns_503(self, client, container):
        def unavailable():
            raise MissingEngineError("engine 'gliner' needs the `chosen` extra")

        container.extraction_session = lambda: ExtractionSession(unavailable)
        r = post_extract(client, [])
        assert r.status_code == 503 and "chosen" in r.json()["detail"]


class TestRAG:
    def test_ingest_then_ask(self, client, container):
        r = client.post("/api/ingest", files={"pdf_file": ("minutes.pdf", PDF, "application/pdf")})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success" and body["chunks_stored"] >= 1 and body["document_id"]
        assert container.vector_store.ensured == 1

        r = client.post("/api/ask", json={"question": "Who opened the meeting?"})
        assert r.status_code == 200
        assert r.json()["answer"] == "Richard Hernandez chaired the meeting."
        assert any("Richard Hernandez opened the meeting" in s for s in r.json()["sources"])
        assert "Who opened the meeting?" in container.llm.prompts[0]

    def test_ask_without_documents(self, client):
        r = client.post("/api/ask", json={"question": "Anything?"})
        assert r.json() == {"answer": "No relevant information found.", "sources": []}

    @pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "x" * 2001}])
    def test_ask_validates_question(self, client, payload):
        assert client.post("/api/ask", json=payload).status_code == 422


class TestOps:
    def test_health(self, client):
        assert client.get("/health").json() == {"status": "ok"}

    def test_request_id_is_echoed_or_generated(self, client):
        assert client.get("/health", headers={"x-request-id": "abc123"}).headers["x-request-id"] == "abc123"
        assert len(client.get("/health").headers["x-request-id"]) == 32


class TestIndexingFromExtractAndIngest:
    @pytest.fixture
    def indexing_container(self, container, tmp_path):
        from app.storage import LocalFileStorage
        from tests.fakes import FakeOrchestrator

        container.storage = LocalFileStorage(tmp_path)
        container.orchestrator = FakeOrchestrator()
        return container

    def test_extract_schedules_indexing_under_the_content_id(self, client, indexing_container):
        from app.storage import JobArtifacts, document_id_for

        r = post_extract(client, [])
        assert r.status_code == 200
        document_id = document_id_for(PDF)
        assert indexing_container.orchestrator.indexing == [(document_id, "doc.pdf", 1)]
        art = JobArtifacts(indexing_container.storage, document_id)
        assert "Richard Hernandez" in art.get_json(art.page(0))["text"]  # the OCR result, reused (no 2nd OCR)
        assert len(indexing_container.ocr.reads) == 1

    def test_extract_can_skip_indexing(self, client, indexing_container):
        indexing_container.settings = Settings(_env_file=None, index_on_extract=False)
        assert post_extract(client, []).status_code == 200
        assert indexing_container.orchestrator.indexing == []

    def test_indexing_failure_never_affects_extraction(self, client, indexing_container):
        indexing_container.orchestrator.fail = True
        r = post_extract(client, [{"first_name": "Richard", "last_name": "Hernandez"}])
        assert r.status_code == 200 and r.json()["fuzzy_matches"]

    def test_ingesting_the_same_pdf_twice_does_not_duplicate(self, client, container):
        first = client.post("/api/ingest", files={"pdf_file": ("a.pdf", PDF, "application/pdf")}).json()
        points_after_first = len(container.vector_store.points)
        second = client.post("/api/ingest", files={"pdf_file": ("copy.pdf", PDF, "application/pdf")}).json()
        assert first["document_id"] == second["document_id"]
        assert len(container.vector_store.points) == points_after_first


def test_temporary_pdf_is_always_removed():
    import os

    from app.api.uploads import temporary_pdf

    with pytest.raises(RuntimeError), temporary_pdf(b"%PDF-1.7") as path:
        assert os.path.exists(path)
        raise RuntimeError("boom")
    assert not os.path.exists(path)
