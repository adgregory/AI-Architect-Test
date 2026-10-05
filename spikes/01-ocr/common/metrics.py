"""Accuracy and localization metrics computed identically for every engine."""

from __future__ import annotations

import re
import unicodedata

import jiwer

from .engine import Box, TextElement

PUNCT = ".,;:!?()[]'\"“”‘’"


def normalize(text: str) -> str:
    """NFKC, unify quotes/dashes, collapse all whitespace (line breaks included)."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"}))
    return re.sub(r"\s+", " ", text).strip()


def cer_wer(reference: str, hypothesis: str) -> tuple[float, float]:
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not hyp:
        return 1.0, 1.0
    return jiwer.cer(ref, hyp), jiwer.wer(ref, hyp)


def name_recall(names: list[str], hypothesis: str) -> tuple[float, float]:
    """Share of expected names found verbatim (case-sensitive, case-insensitive)."""
    if not names:
        return 1.0, 1.0
    hyp = normalize(hypothesis)
    exact = sum(n in hyp for n in names)
    folded = sum(n.lower() in hyp.lower() for n in names)
    return exact / len(names), folded / len(names)


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


def _tok(s: str) -> str:
    return s.strip(PUNCT).lower()


def name_candidates(name: str, elements: list[TextElement], granularity: str) -> list[Box]:
    """Predicted boxes that could represent `name` at this granularity.

    word-level: runs of consecutive words equal to the name's tokens → union box.
    line/block-level: any element whose text contains the name → that element's box.
    """
    parts = [p.lower() for p in name.split()]
    if granularity == "word":
        toks = [_tok(e.text) for e in elements]
        out = []
        for i in range(len(toks) - len(parts) + 1):
            if toks[i : i + len(parts)] == parts:
                boxes = [elements[i + k].box for k in range(len(parts))]
                out.append([min(b[0] for b in boxes), min(b[1] for b in boxes),
                            max(b[2] for b in boxes), max(b[3] for b in boxes)])
        return out
    return [e.box for e in elements if name.lower() in normalize(e.text).lower()]


def name_localization(gt_names: list[dict], elements: list[TextElement], granularity: str) -> list[float]:
    """Best IoU per ground-truth name instance (0.0 when the name isn't found)."""
    return [
        max((iou(gt["box"], c) for c in name_candidates(gt["name"], elements, granularity)), default=0.0)
        for gt in gt_names
    ]
