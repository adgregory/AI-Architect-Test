from app.services.ocr_service import get_word_bounding_boxes
from app.services.ner_service import extract_names

# Punctuation OCR attaches to words ("Smith," / "(Chen)") that must not block a match.
_EDGE_PUNCTUATION = ".,;:!?()[]{}\"'"


def _normalize(token: str) -> str:
    return token.strip(_EDGE_PUNCTUATION).casefold()


def find_name_bounding_boxes(pdf_path: str, text: str) -> list[dict]:
    """Match extracted names to their bounding boxes in the PDF.

    A name matches a run of consecutive OCR words on the same page, compared
    case-insensitively and ignoring surrounding punctuation. Every occurrence
    is returned, in document order.
    """
    names = list(dict.fromkeys(extract_names(text)))  # dedupe, keep order
    word_boxes = get_word_bounding_boxes(pdf_path)
    tokens = [_normalize(wb["word"]) for wb in word_boxes]

    name_boxes = []
    for name in names:
        parts = [_normalize(p) for p in name.split()]
        parts = [p for p in parts if p]
        if not parts:
            continue

        for start in range(len(word_boxes) - len(parts) + 1):
            run = word_boxes[start : start + len(parts)]
            if tokens[start : start + len(parts)] != parts:
                continue
            if len({b["page"] for b in run}) != 1:
                continue

            min_x = min(b["x"] for b in run)
            min_y = min(b["y"] for b in run)
            max_x = max(b["x"] + b["width"] for b in run)
            max_y = max(b["y"] + b["height"] for b in run)

            name_boxes.append({
                "name": name,
                "page": run[0]["page"],
                "x": min_x,
                "y": min_y,
                "width": max_x - min_x,
                "height": max_y - min_y,
            })

    name_boxes.sort(key=lambda b: (b["page"], b["y"], b["x"]))
    return name_boxes
