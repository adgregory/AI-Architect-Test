"""Common engine interface so every OCR candidate is measured the same way."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from PIL import Image

Box = list[float]  # [x0, y0, x1, y1] in page-image pixels, origin top-left


@dataclass
class TextElement:
    text: str
    box: Box
    # Original quadrilateral when the engine returns one: [[x, y], ...] x4.
    polygon: list[list[float]] | None = None
    confidence: float | None = None


@dataclass
class OCRResult:
    text: str
    # Every granularity the engine can return natively, e.g. "word", "line", "block".
    elements: dict[str, list[TextElement]] = field(default_factory=dict)


class OCREngine(Protocol):
    name: str
    device: str

    def load(self) -> None:
        """Load models. Called once, timed as part of cold start."""

    def ocr(self, image: Image.Image) -> OCRResult:
        """OCR one page image (RGB). Boxes must be in this image's pixel space."""

    def info(self) -> dict:
        """Versions, models, ONNX usage, actual device used, notes on positional output."""

    def gpu_memory_bytes(self) -> int | None:
        """Current GPU memory held by the engine, if measurable."""
