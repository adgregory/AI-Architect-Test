"""Contract tests: every implementation of an interface must satisfy the same behaviour.

Each engine is built through its factory from Settings, exactly as the app builds it.
Engines whose extra isn't installed are skipped. Uses real models (marked `contract`).
"""

from pathlib import Path

import numpy as np
import pytest

from app.core.config import Settings
from app.core.factories import (
    EmbeddingServiceFactory,
    MissingEngineError,
    NERServiceFactory,
    OCRServiceFactory,
)
from app.services.embedding_service import EmbeddingService
from app.services.ner_service import NERService
from app.services.ocr_service import OCRService

pytestmark = pytest.mark.contract

SAMPLES = Path(__file__).resolve().parents[2] / "sample_pdfs"
LETTER_POINTS = (612.0, 792.0)  # the sample PDFs are US Letter


def build(factory, **engine):
    try:
        return factory.create(Settings(_env_file=None, **engine))
    except MissingEngineError as exc:
        pytest.skip(str(exc))


# --------------------------------------------------------------------------- #
# OCR
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module", params=["rapidocr", "tesseract"])
def ocr(request) -> OCRService:
    return build(OCRServiceFactory, ocr_engine=request.param)


@pytest.fixture(scope="module")
def minutes(ocr):
    return ocr.read(str(SAMPLES / "meeting_minutes.pdf"))


class TestOCRContract:
    def test_implements_protocol(self, ocr):
        assert isinstance(ocr, OCRService)

    def test_reads_every_page(self, minutes):
        assert "richard hernandez" in minutes.text.lower()  # page 1
        assert "alexander popov" in minutes.text.lower()    # page 2
        assert {w["page"] for w in minutes.words} == {0, 1}

    def test_boxes_are_in_pdf_points_within_the_page(self, minutes):
        width, height = LETTER_POINTS
        for w in minutes.words:
            assert 0 <= w["x"] and w["x"] + w["width"] <= width + 1
            assert 0 <= w["y"] and w["y"] + w["height"] <= height + 1
            assert w["width"] > 0 and w["height"] > 0

    def test_known_word_position(self, ocr):
        # The generator draws "MEMORANDUM" at (100, 100) px at 150 DPI -> (48, 48) pt.
        words = ocr.read(str(SAMPLES / "company_memo.pdf")).words
        memo = next(w for w in words if w["word"].upper() == "MEMORANDUM")
        assert memo["page"] == 0
        assert memo["x"] == pytest.approx(48, abs=4) and memo["y"] == pytest.approx(48, abs=6)

    def test_pages_compose_the_document(self, ocr, minutes):
        path = str(SAMPLES / "meeting_minutes.pdf")
        assert ocr.page_count(path) == 2
        pages = [ocr.read_page(path, n) for n in range(2)]
        assert [w for p in pages for w in p.words] == minutes.words
        assert "\n".join(p.text for p in pages) == minutes.text
        assert all(w["page"] == n for n, p in enumerate(pages) for w in p.words)

    def test_page_out_of_range(self, ocr):
        with pytest.raises(IndexError):
            ocr.read_page(str(SAMPLES / "company_memo.pdf"), 1)

    def test_box_api_agrees_with_read(self, ocr):
        path = str(SAMPLES / "company_memo.pdf")
        assert ocr.get_word_boxes(path) == ocr.read(path).words


# --------------------------------------------------------------------------- #
# NER
# --------------------------------------------------------------------------- #
TEXT = ("Margaret Thompson is the CEO of Acme Corporation. Robert Chen works at Google in London "
        "and reports to James Anderson. Maria Garcia previously led the UX team.")


@pytest.fixture(scope="module", params=["gliner", "spacy"])
def ner(request) -> NERService:
    return build(NERServiceFactory, ner_engine=request.param)


class TestNERContract:
    def test_implements_protocol(self, ner):
        assert isinstance(ner, NERService)

    def test_finds_people(self, ner):
        found = set(ner.extract_names(TEXT))
        assert {"Margaret Thompson", "Robert Chen", "James Anderson", "Maria Garcia"} <= found

    def test_never_returns_organisations_or_places(self, ner):
        found = set(ner.extract_names(TEXT))
        assert not found & {"Acme Corporation", "Google", "London"}

    def test_positions_point_at_the_names(self, ner):
        for ent in ner.extract_names_with_positions(TEXT):
            assert TEXT[ent["start_char"]:ent["end_char"]] == ent["name"]
            assert ent["label"] == "PERSON"

    def test_no_names(self, ner):
        assert ner.extract_names("The quarterly budget increased by twelve percent.") == []


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module", params=["fastembed", "sentence-transformers"])
def embeddings(request) -> EmbeddingService:
    return build(EmbeddingServiceFactory, embedding_engine=request.param)


class TestEmbeddingContract:
    def test_implements_protocol(self, embeddings):
        assert isinstance(embeddings, EmbeddingService)

    def test_dimension_and_normalisation(self, embeddings):
        vectors = np.array(embeddings.embed_documents(["Robert Chen was promoted.", "Revenue grew."]))
        assert vectors.shape == (2, Settings(_env_file=None).embedding_dim)
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3)

    def test_paraphrase_closer_than_unrelated(self, embeddings):
        a, para, other = map(np.array, embeddings.embed_documents([
            "Robert Chen has been promoted to Vice President of Engineering.",
            "Robert Chen is the new VP of Engineering.",
            "Tomatoes grow best in full sun.",
        ]))
        assert a @ para > a @ other

    def test_query_retrieves_the_relevant_document(self, embeddings):
        docs = ["The cloud migration to AWS is expected by Q3 2024.",
                "Maria Garcia has been appointed Head of Product Design.",
                "The NIH grant is worth $600K."]
        q = np.array(embeddings.embed_query("Who leads product design?"))
        assert int(np.argmax(np.array(embeddings.embed_documents(docs)) @ q)) == 1

    def test_empty_input(self, embeddings):
        assert embeddings.embed_documents([]) == []
