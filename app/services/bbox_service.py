"""Locate names in OCR word boxes."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.services.ocr_service import get_word_bounding_boxes
from app.services.ner_service import extract_names


@runtime_checkable
class NameLocator(Protocol):
    def locate(self, names: list[str], word_boxes: list[dict]) -> list[dict]: ...


class ConsecutiveWordNameLocator:
    """A name matches a run of consecutive OCR words on the same page, compared
    case-insensitively and ignoring surrounding punctuation. Every occurrence is
    returned, in document order. Pure: no I/O.
    """

    # Punctuation OCR attaches to words ("Smith," / "(Chen)") that must not block a match.
    EDGE_PUNCTUATION = ".,;:!?()[]{}\"'"

    @classmethod
    def normalize(cls, token: str) -> str:
        return token.strip(cls.EDGE_PUNCTUATION).casefold()

    def locate(self, names: list[str], word_boxes: list[dict]) -> list[dict]:
        tokens = [self.normalize(wb["word"]) for wb in word_boxes]
        name_boxes = []
        for name in dict.fromkeys(names):  # dedupe, keep order
            parts = [p for p in (self.normalize(t) for t in name.split()) if p]
            if not parts:
                continue
            for start in range(len(word_boxes) - len(parts) + 1):
                if tokens[start : start + len(parts)] != parts:
                    continue
                run = word_boxes[start : start + len(parts)]
                if len({b["page"] for b in run}) == 1:
                    name_boxes.append(self._merge(name, run))
        name_boxes.sort(key=lambda b: (b["page"], b["y"], b["x"]))
        return name_boxes

    @staticmethod
    def _merge(name: str, run: list[dict]) -> dict:
        min_x = min(b["x"] for b in run)
        min_y = min(b["y"] for b in run)
        max_x = max(b["x"] + b["width"] for b in run)
        max_y = max(b["y"] + b["height"] for b in run)
        return {"name": name, "page": run[0]["page"], "x": min_x, "y": min_y,
                "width": max_x - min_x, "height": max_y - min_y}


# --------------------------------------------------------------------------- #
# Functional API (backwards compatible): NER + OCR boxes + locator. The adapter
# resolves extract_names / get_word_bounding_boxes at call time.
# --------------------------------------------------------------------------- #
class _ModuleNameBoxes:
    def __init__(self, locator: NameLocator):
        self._locator = locator

    def find_name_bounding_boxes(self, pdf_path: str, text: str) -> list[dict]:
        return self._locator.locate(extract_names(text), get_word_bounding_boxes(pdf_path))


find_name_bounding_boxes = _ModuleNameBoxes(ConsecutiveWordNameLocator()).find_name_bounding_boxes
