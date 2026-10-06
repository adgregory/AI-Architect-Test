"""Box helpers shared by adapters."""

from __future__ import annotations

from .engine import Box, TextElement


def quad_to_box(quad) -> Box:
    xs = [float(p[0]) for p in quad]
    ys = [float(p[1]) for p in quad]
    return [min(xs), min(ys), max(xs), max(ys)]


def union(boxes: list[Box]) -> Box:
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def words_from_pieces(pieces: list[tuple[str, Box]]) -> list[TextElement]:
    """Rebuild whitespace-delimited words from recognizer pieces of a single line.

    Engines split lines differently (Paddle emits punctuation and spaces as separate
    pieces); a word is a maximal run of non-space characters, and its box is the
    union of the boxes of the pieces it came from.
    """
    words: list[TextElement] = []
    text, boxes = "", []
    for piece, box in pieces:
        for ch in piece:
            if ch.isspace():
                if text:
                    words.append(TextElement(text, union(boxes)))
                text, boxes = "", []
            else:
                text += ch
                if not boxes or boxes[-1] is not box:
                    boxes.append(box)
    if text:
        words.append(TextElement(text, union(boxes)))
    return words
