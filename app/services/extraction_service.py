"""Name extraction pipeline: OCR → NER → name boxes → fuzzy matching."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.services.bbox_service import NameLocator
from app.services.fuzzy_service import NameMatcher
from app.services.ner_service import NERService, PersonNameNormalizer
from app.services.ocr_service import OCRResult, OCRService


@dataclass(frozen=True)
class ExtractionEngines:
    """The configured, long-lived components (built once at startup)."""

    ocr: OCRService
    ner: NERService
    locator: NameLocator
    matcher: NameMatcher
    normalizer: PersonNameNormalizer


class ExtractionResultBuilder:
    """The API response shape, shared by the synchronous route and the async job workflow."""

    @staticmethod
    def build(name_boxes: list[dict], matches: list[dict]) -> dict:
        return {
            "extracted_names": [
                {
                    "name": nb["name"],
                    "bounding_box": {
                        "page_number": nb["page"],
                        "x": nb["x"],
                        "y": nb["y"],
                        "width": nb["width"],
                        "height": nb["height"],
                    },
                }
                for nb in name_boxes
            ],
            "fuzzy_matches": matches,
        }


class NameExtraction:
    """NER → normalise → locate → match over an OCR result already in hand (used by the
    workflow, which OCRs pages in parallel and combines them)."""

    def __init__(self, engines: ExtractionEngines):
        self._engines = engines

    def run(self, ocr: OCRResult, query_names: list[dict]) -> dict:
        names = self._engines.normalizer.normalize_all(self._engines.ner.extract_names(ocr.text))
        name_boxes = self._engines.locator.locate(names, ocr.words)
        matches = self._engines.matcher.match(list(dict.fromkeys(nb["name"] for nb in name_boxes)), query_names)
        return ExtractionResultBuilder.build(name_boxes, matches)


class ExtractionSession:
    """Per-request unit of work over the shared engines.

    OCR runs once per document and is reused for both the text and the word
    boxes (the original flow ran OCR twice per request). Engines are resolved on
    first use, so creating a session is free.
    """

    def __init__(self, engines: Callable[[], ExtractionEngines]):
        self._get_engines = engines
        self._ocr_cache: dict[str, OCRResult] = {}

    @property
    def _engines(self) -> ExtractionEngines:
        return self._get_engines()

    def read(self, pdf_path: str) -> OCRResult:
        if pdf_path not in self._ocr_cache:
            self._ocr_cache[pdf_path] = self._engines.ocr.read(pdf_path)
        return self._ocr_cache[pdf_path]

    def cached(self, pdf_path: str) -> OCRResult | None:
        """The OCR result already produced in this session, if any (never triggers OCR)."""
        return self._ocr_cache.get(pdf_path)

    def extract_text(self, pdf_path: str) -> str:
        return self.read(pdf_path).text

    def find_name_boxes(self, pdf_path: str, text: str) -> list[dict]:
        names = self._engines.normalizer.normalize_all(self._engines.ner.extract_names(text))
        return self._engines.locator.locate(names, self.read(pdf_path).words)

    def match(self, extracted_names: list[str], query_names: list[dict]) -> list[dict]:
        return self._engines.matcher.match(list(dict.fromkeys(extracted_names)), query_names)
