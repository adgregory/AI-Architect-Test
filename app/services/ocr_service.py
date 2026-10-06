"""OCR: text and word bounding boxes from scanned PDFs."""

from __future__ import annotations

import io
from typing import Protocol, runtime_checkable

import fitz
from PIL import Image

try:  # optional: installed with the `fallback` extra (plus the tesseract binary)
    import pytesseract
except ImportError:  # pragma: no cover - depends on installed extras
    pytesseract = None

from app.core.config import get_settings

PDF_POINTS_PER_INCH = 72


@runtime_checkable
class OCRService(Protocol):
    """Swappable OCR engine. Boxes are returned in PDF coordinate space (points)."""

    def extract_text(self, pdf_path: str) -> str: ...

    def get_word_boxes(self, pdf_path: str) -> list[dict]: ...


class TesseractOCRService:
    """Tesseract (LSTM) via pytesseract, rendering pages with PyMuPDF."""

    def __init__(self, dpi: int = 150):
        self._dpi = dpi

    @staticmethod
    def _require_tesseract() -> None:
        if pytesseract is None:
            raise RuntimeError("Tesseract OCR needs the `fallback` extra: uv sync --extra fallback")

    def _page_images(self, pdf_path: str):
        """Yield (page_number, PIL image) for every page; always closes the document."""
        doc = fitz.open(pdf_path)
        try:
            for page_num in range(len(doc)):
                pix = doc[page_num].get_pixmap(dpi=self._dpi)
                yield page_num, Image.open(io.BytesIO(pix.tobytes("png")))
        finally:
            doc.close()

    def extract_text(self, pdf_path: str) -> str:
        self._require_tesseract()
        return "".join(
            pytesseract.image_to_string(img) + "\n" for _, img in self._page_images(pdf_path)
        )

    def get_word_boxes(self, pdf_path: str) -> list[dict]:
        self._require_tesseract()
        scale = PDF_POINTS_PER_INCH / self._dpi
        results = []
        for page_num, img in self._page_images(pdf_path):
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            for i, raw in enumerate(data["text"]):
                word = raw.strip()
                if not word:
                    continue
                results.append({
                    "word": word,
                    "page": page_num,
                    "x": data["left"][i] * scale,
                    "y": data["top"][i] * scale,
                    "width": data["width"][i] * scale,
                    "height": data["height"][i] * scale,
                })
        return results


# --------------------------------------------------------------------------- #
# Functional API (backwards compatible): module-level entry points backed by
# the reference Tesseract implementation.
# --------------------------------------------------------------------------- #
_default_ocr = TesseractOCRService(dpi=get_settings().ocr_dpi)

extract_text_from_pdf = _default_ocr.extract_text
get_word_bounding_boxes = _default_ocr.get_word_boxes
