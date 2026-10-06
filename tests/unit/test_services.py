"""Service orchestration with in-memory fakes: extraction session, RAG, vector store."""

import uuid

from app.services.bbox_service import ConsecutiveWordNameLocator
from app.services.embedding_service import EmbeddingService
from app.services.extraction_service import ExtractionEngines, ExtractionSession
from app.services.fuzzy_service import TokenSortNameMatcher
from app.services.ner_service import NERService, PersonNameNormalizer
from app.services.ocr_service import OCRResult, OCRService
from app.services.rag_service import NO_INFORMATION, LLMClient, RAGService
from app.services.vector_service import QdrantVectorStore, VectorStore
from tests.fakes import (
    FakeEmbeddings,
    FakeLLM,
    FakeNER,
    FakeOCR,
    FakeQdrantClient,
    InMemoryVectorStore,
    word,
)

MEMO = OCRResult(
    text="To: All staff\nFrom: Dr. Aisha Patel, CTO\nRobert Chen joins Acme Corp.\nAisha Patel will review.",
    words=[word("Dr.", 0, 10, 20), word("Aisha", 0, 40, 20), word("Patel,", 0, 75, 20),
           word("Robert", 0, 10, 40), word("Chen", 0, 50, 40),
           word("Aisha", 1, 10, 60), word("Patel", 1, 45, 60)],
)


def make_session(ocr: FakeOCR, names: list[str]) -> ExtractionSession:
    engines = ExtractionEngines(ocr=ocr, ner=FakeNER(names), locator=ConsecutiveWordNameLocator(),
                                matcher=TokenSortNameMatcher(), normalizer=PersonNameNormalizer())
    return ExtractionSession(lambda: engines)


class TestFakesHonourTheInterfaces:
    def test_fakes_implement_protocols(self):
        assert isinstance(FakeOCR(MEMO), OCRService)
        assert isinstance(FakeNER([]), NERService)
        assert isinstance(FakeEmbeddings(), EmbeddingService)
        assert isinstance(InMemoryVectorStore(), VectorStore)
        assert isinstance(FakeLLM(), LLMClient)


class TestExtractionSession:
    def test_runs_ocr_once_per_document(self):
        ocr = FakeOCR(MEMO)
        session = make_session(ocr, ["Dr. Aisha Patel", "Robert Chen"])
        text = session.extract_text("memo.pdf")
        session.find_name_boxes("memo.pdf", text)
        session.find_name_boxes("memo.pdf", text)
        assert ocr.reads == ["memo.pdf"]

    def test_names_are_normalised_then_located_everywhere(self):
        session = make_session(FakeOCR(MEMO), ["Dr. Aisha Patel", "Robert Chen", "Aisha Patel"])
        boxes = session.find_name_boxes("memo.pdf", session.extract_text("memo.pdf"))
        assert [(b["name"], b["page"]) for b in boxes] == [
            ("Aisha Patel", 0), ("Robert Chen", 0), ("Aisha Patel", 1),
        ]

    def test_match_deduplicates_occurrences(self):
        session = make_session(FakeOCR(MEMO), [])
        matches = session.match(["Aisha Patel", "Aisha Patel"], [{"first_name": "Aisha", "last_name": "Patel"}])
        assert len(matches) == 1 and matches[0]["score"] == 1.0

    def test_engines_resolve_lazily(self):
        built = []
        session = ExtractionSession(lambda: built.append(1) or None)
        assert built == []  # creating a session never builds engines


class TestRAGService:
    def setup_method(self):
        self.embeddings = FakeEmbeddings()
        self.store = InMemoryVectorStore(score_threshold=0.1)
        self.llm = FakeLLM("Robert Chen leads engineering.")
        texts = ["Robert Chen was promoted to Vice President of Engineering.",
                 "The cloud migration to AWS is expected by Q3 2024."]
        self.store.upsert(texts, self.embeddings.embed_documents(texts))
        self.rag = RAGService(self.embeddings.embed_query, self.store.search, self.llm, top_k=1)

    def test_prompt_contains_question_and_retrieved_context(self):
        result = self.rag.answer("Who was promoted to Vice President of Engineering?")
        prompt = self.llm.prompts[0]
        assert "Who was promoted to Vice President of Engineering?" in prompt
        assert "Robert Chen was promoted" in prompt
        assert result == {"answer": "Robert Chen leads engineering.",
                          "sources": ["Robert Chen was promoted to Vice President of Engineering."]}

    def test_no_context_skips_the_llm(self):
        rag = RAGService(self.embeddings.embed_query, InMemoryVectorStore().search, self.llm)
        assert rag.answer("Anything?") == {"answer": NO_INFORMATION, "sources": []}
        assert self.llm.prompts == []


class TestQdrantVectorStore:
    def test_creates_cosine_collection_once(self):
        client = FakeQdrantClient()
        store = QdrantVectorStore(client, collection="docs", vector_size=8)
        store.ensure_collection()
        store.ensure_collection()
        assert len(client.created) == 1
        assert client.created[0]["vectors_config"].size == 8

    def test_point_ids_unique_across_documents_and_stable_per_document(self):
        client = FakeQdrantClient()
        store = QdrantVectorStore(client, collection="docs", vector_size=2)
        store.upsert(["a", "b"], [[1, 0], [0, 1]], document_id="doc-1")
        store.upsert(["c", "d"], [[1, 0], [0, 1]], document_id="doc-2")
        store.upsert(["a", "b"], [[1, 0], [0, 1]], document_id="doc-1")
        ids = [[p.id for p in call["points"]] for call in client.upserts]
        assert set(ids[0]).isdisjoint(ids[1])
        assert ids[0] == ids[2]  # re-ingesting overwrites instead of duplicating
        assert all(uuid.UUID(i) for i in ids[0])
        assert client.upserts[0]["points"][1].payload == {"text": "b", "document_id": "doc-1", "chunk_index": 1}

    def test_search_passes_threshold_and_maps_hits(self):
        client = FakeQdrantClient()
        client.search_hits = [type("Hit", (), {"payload": {"text": "chunk"}, "score": 0.8})()]
        store = QdrantVectorStore(client, collection="docs", vector_size=2, score_threshold=0.42)
        assert store.search([0.1, 0.2], top_k=3) == [{"text": "chunk", "score": 0.8}]
        assert client.search_calls[0]["score_threshold"] == 0.42
        assert client.search_calls[0]["limit"] == 3
