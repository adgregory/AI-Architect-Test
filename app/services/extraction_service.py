"""Name extraction pipeline: OCR → NER → name boxes → fuzzy matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

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

    def extract_text(self, pdf_path: str) -> str:
        return self.read(pdf_path).text

    def find_name_boxes(self, pdf_path: str, text: str) -> list[dict]:
        names = self._engines.normalizer.normalize_all(self._engines.ner.extract_names(text))
        return self._engines.locator.locate(names, self.read(pdf_path).words)

    def match(self, extracted_names: list[str], query_names: list[dict]) -> list[dict]:
        return self._engines.matcher.match(list(dict.fromkeys(extracted_names)), query_names)
