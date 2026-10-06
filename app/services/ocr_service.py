"""OCR: text and word bounding boxes from scanned PDFs."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, runtime_checkable

import fitz
import numpy as np
from PIL import Image

try:  # optional: installed with the `fallback` extra (plus the tesseract binary)
    import pytesseract
except ImportError:  # pragma: no cover - depends on installed extras
    pytesseract = None

from app.core.config import get_settings

PDF_POINTS_PER_INCH = 72


@dataclass
class OCRResult:
    """One OCR pass over a document: text plus word boxes in PDF points (origin top-left)."""

    text: str
    words: list[dict] = field(default_factory=list)  # {word, page, x, y, width, height}


@runtime_checkable
class OCRService(Protocol):
    """Swappable OCR engine. Boxes are returned in PDF coordinate space (points)."""

    def read(self, pdf_path: str) -> OCRResult: ...

    def page_count(self, pdf_path: str) -> int: ...

    def read_page(self, pdf_path: str, page_number: int) -> OCRResult:
        """OCR one page (zero-based); `read` is the concatenation of every page."""
        ...

    def extract_text(self, pdf_path: str) -> str: ...

    def get_word_boxes(self, pdf_path: str) -> list[dict]: ...


class PageRenderer:
    """Renders PDF pages to images with PyMuPDF; always closes the document."""

    def __init__(self, dpi: int):
        self.dpi = dpi
        self.scale = PDF_POINTS_PER_INCH / dpi  # image pixels -> PDF points

    def _image(self, doc, page_num: int) -> Image.Image:
        pix = doc[page_num].get_pixmap(dpi=self.dpi)
        return Image.open(io.BytesIO(pix.tobytes("png")))

    def pages(self, pdf_path: str) -> Iterator[tuple[int, Image.Image]]:
        doc = fitz.open(pdf_path)
        try:
            for page_num in range(len(doc)):
                yield page_num, self._image(doc, page_num)
        finally:
            doc.close()

    def page(self, pdf_path: str, page_num: int) -> Image.Image:
        doc = fitz.open(pdf_path)
        try:
            if not 0 <= page_num < len(doc):
                raise IndexError(f"page {page_num} out of range (document has {len(doc)} pages)")
            return self._image(doc, page_num)
        finally:
            doc.close()

    def count(self, pdf_path: str) -> int:
        doc = fitz.open(pdf_path)
        try:
            return len(doc)
        finally:
            doc.close()


class _PageWiseOCR:
    """Shared plumbing: engines implement `_read_image`; whole-document and per-page reads
    are built from it, so `read(pdf) == combine(read_page(pdf, n) for n in pages)`."""

    _renderer: PageRenderer

    def _read_image(self, img: Image.Image, page_num: int) -> tuple[list[str], list[dict]]:
        raise NotImplementedError

    def page_count(self, pdf_path: str) -> int:
        return self._renderer.count(pdf_path)

    def read_page(self, pdf_path: str, page_number: int) -> OCRResult:
        lines, words = self._read_image(self._renderer.page(pdf_path, page_number), page_number)
        return OCRResult(text="\n".join(lines), words=words)

    def read(self, pdf_path: str) -> OCRResult:
        lines, words = [], []
        for page_num, img in self._renderer.pages(pdf_path):
            page_lines, page_words = self._read_image(img, page_num)
            lines.extend(page_lines)
            words.extend(page_words)
        return OCRResult(text="\n".join(lines), words=words)

    def extract_text(self, pdf_path: str) -> str:
        return self.read(pdf_path).text

    def get_word_boxes(self, pdf_path: str) -> list[dict]:
        return self.read(pdf_path).words


class TesseractOCRService(_PageWiseOCR):
    """Tesseract (LSTM) via pytesseract — the reference/fallback engine."""

    def __init__(self, dpi: int = 150):
        self._renderer = PageRenderer(dpi)

    @staticmethod
    def _require_tesseract() -> None:
        if pytesseract is None:
            raise RuntimeError("Tesseract OCR needs the `fallback` extra: uv sync --extra fallback")

    def extract_text(self, pdf_path: str) -> str:
        self._require_tesseract()
        return "".join(
            pytesseract.image_to_string(img) + "\n" for _, img in self._renderer.pages(pdf_path)
        )

    def _read_image(self, img: Image.Image, page_num: int) -> tuple[list[str], list[dict]]:
        """image_to_data yields words, boxes and the line structure in one pass."""
        self._require_tesseract()
        scale = self._renderer.scale
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        n = len(data["text"])
        line_keys = list(zip(*(data.get(k) or [0] * n for k in ("block_num", "par_num", "line_num"))))
        page_lines: dict[tuple, list[str]] = {}
        words = []
        for i, raw in enumerate(data["text"]):
            word = raw.strip()
            if not word:
                continue
            page_lines.setdefault(line_keys[i], []).append(word)
            words.append({
                "word": word,
                "page": page_num,
                "x": data["left"][i] * scale,
                "y": data["top"][i] * scale,
                "width": data["width"][i] * scale,
                "height": data["height"][i] * scale,
            })
        return [" ".join(ws) for ws in page_lines.values()], words


class RapidOCRService(_PageWiseOCR):
    """PaddleOCR PP-OCRv5 mobile models on ONNX Runtime via RapidOCR (chosen in spike 01).

    The RapidOCR engine is injected. Word boxes come from the recogniser's character
    positions and span the full text-line height; with `tighten_boxes` each box is
    shrunk vertically to the ink pixels it contains.
    """

    INK_THRESHOLD = 128  # grayscale value below which a pixel counts as ink

    def __init__(self, engine: Any, dpi: int = 150, tighten_boxes: bool = True):
        self._engine = engine
        self._renderer = PageRenderer(dpi)
        self._tighten = tighten_boxes

    def _read_image(self, img: Image.Image, page_num: int) -> tuple[list[str], list[dict]]:
        scale = self._renderer.scale
        gray = np.asarray(img.convert("L"))
        result = self._engine(np.asarray(img.convert("RGB")), return_word_box=True)
        if result.txts is None:
            return [], []
        words = []
        for line_words in result.word_results:
            for text, x0, y0, x1, y1 in self._merge_pieces(line_words):
                if self._tighten:
                    y0, y1 = self._tighten_vertically(gray, x0, y0, x1, y1)
                words.append({
                    "word": text,
                    "page": page_num,
                    "x": x0 * scale,
                    "y": y0 * scale,
                    "width": (x1 - x0) * scale,
                    "height": (y1 - y0) * scale,
                })
        return list(result.txts), words

    @staticmethod
    def _merge_pieces(pieces) -> list[tuple[str, float, float, float, float]]:
        """Recogniser pieces of one line → whitespace-delimited words with union boxes (pixels)."""
        words, text, box = [], "", None
        for piece_text, _score, quad in pieces:
            xs = [float(p[0]) for p in quad]
            ys = [float(p[1]) for p in quad]
            piece_box = [min(xs), min(ys), max(xs), max(ys)]
            for token in piece_text.split(" "):
                if not token:
                    if text:
                        words.append((text, *box))
                    text, box = "", None
                    continue
                text += token
                box = piece_box if box is None else [min(box[0], piece_box[0]), min(box[1], piece_box[1]),
                                                     max(box[2], piece_box[2]), max(box[3], piece_box[3])]
            if text:  # pieces are words here; a new piece starts a new word
                words.append((text, *box))
                text, box = "", None
        return words

    def _tighten_vertically(self, gray: np.ndarray, x0, y0, x1, y1) -> tuple[float, float]:
        h, w = gray.shape
        c0, c1 = max(0, int(x0)), min(w, int(np.ceil(x1)))
        r0, r1 = max(0, int(y0)), min(h, int(np.ceil(y1)))
        if c1 <= c0 or r1 <= r0:
            return y0, y1
        ink_rows = np.where((gray[r0:r1, c0:c1] < self.INK_THRESHOLD).any(axis=1))[0]
        if ink_rows.size == 0:
            return y0, y1
        return float(r0 + ink_rows[0]), float(r0 + ink_rows[-1] + 1)


# --------------------------------------------------------------------------- #
# Functional API (backwards compatible): module-level entry points backed by
# the reference Tesseract implementation, which the provided tests exercise.
# The application itself uses the engine selected in Settings (see
# app/core/factories.py).
# --------------------------------------------------------------------------- #
_reference_ocr = TesseractOCRService(dpi=get_settings().ocr_dpi)

extract_text_from_pdf = _reference_ocr.extract_text
get_word_bounding_boxes = _reference_ocr.get_word_boxes
