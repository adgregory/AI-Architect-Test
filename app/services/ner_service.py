"""Named-entity recognition: person names in text."""

from __future__ import annotations

import re
from typing import Any, Protocol, runtime_checkable

from app.core.factories import default_spacy_model
from app.core.lazy import Lazy

PERSON_LABEL = "PERSON"


@runtime_checkable
class NERService(Protocol):
    """Swappable person-name extractor."""

    def extract_names(self, text: str) -> list[str]: ...

    def extract_names_with_positions(self, text: str) -> list[dict]: ...


class PersonNameNormalizer:
    """Canonical form of a detected name: no titles, possessives, edge punctuation or line breaks.

    "Dr. Aisha Patel" -> "Aisha Patel"; "Kevin O'Brien," -> "Kevin O'Brien".
    """

    TITLES = frozenset({"dr", "prof", "mr", "mrs", "ms", "sir"})
    EDGE_PUNCTUATION = ".,;:()\"'"

    def normalize(self, raw: str) -> str:
        s = re.sub(r"\s+", " ", raw).strip()
        s = re.sub(r"['’]s$", "", s)
        tokens = [t.strip(self.EDGE_PUNCTUATION) for t in s.split()]
        return " ".join(t for t in tokens if t and t.casefold().rstrip(".") not in self.TITLES)

    def normalize_all(self, names: list[str]) -> list[str]:
        """Normalise, drop empties and duplicates, keep first-seen order."""
        return list(dict.fromkeys(n for n in map(self.normalize, names) if n))


class SpacyNERService:
    """spaCy pipeline (reference/fallback); keeps PERSON entities only. The pipeline is injected."""

    def __init__(self, nlp: Any):
        self._nlp = nlp

    def _people(self, text: str):
        return [ent for ent in self._nlp(text).ents if ent.label_ == PERSON_LABEL]

    def extract_names(self, text: str) -> list[str]:
        return [ent.text for ent in self._people(text)]

    def extract_names_with_positions(self, text: str) -> list[dict]:
        return [
            {"name": ent.text, "start_char": ent.start_char, "end_char": ent.end_char, "label": ent.label_}
            for ent in self._people(text)
        ]


class GLiNERNERService:
    """GLiNER zero-shot span model asked for `person` (chosen in spike 02). The model is injected.

    GLiNER silently truncates long inputs, so text is processed in line-aligned windows
    (never cutting a name in half) and offsets are mapped back to the full text.
    """

    def __init__(self, model: Any, threshold: float = 0.3, labels: tuple[str, ...] = ("person",), max_words: int = 200):
        self._model = model
        self._threshold = threshold
        self._labels = list(labels)
        self._max_words = max_words

    def _windows(self, text: str) -> list[tuple[int, str]]:
        out, start, count, pos = [], 0, 0, 0
        for line in text.splitlines(keepends=True):
            n = len(line.split())
            if count and count + n > self._max_words:
                out.append((start, text[start:pos]))
                start, count = pos, 0
            count += n
            pos += len(line)
        if start < len(text):
            out.append((start, text[start:]))
        return out

    def extract_names_with_positions(self, text: str) -> list[dict]:
        found = []
        for offset, chunk in self._windows(text):
            for ent in self._model.predict_entities(chunk, self._labels, threshold=self._threshold):
                found.append(
                    {
                        "name": ent["text"],
                        "start_char": offset + ent["start"],
                        "end_char": offset + ent["end"],
                        "label": PERSON_LABEL,
                        "score": float(ent["score"]),
                    }
                )
        return found

    def extract_names(self, text: str) -> list[str]:
        return [e["name"] for e in self.extract_names_with_positions(text)]


# --------------------------------------------------------------------------- #
# Scaffold API: module-level functions from the original code, kept because the provided
# tests call or patch them; backed by the reference spaCy implementation. `nlp` is a lazy
# proxy: the model loads on first use, not at import; the adapter resolves it at call time.
# The application uses the engine selected in Settings (app/core/factories.py).
# --------------------------------------------------------------------------- #
nlp = Lazy(default_spacy_model)


class _ModuleNER:
    def extract_names(self, text: str) -> list[str]:
        return SpacyNERService(nlp).extract_names(text)

    def extract_names_with_positions(self, text: str) -> list[dict]:
        return SpacyNERService(nlp).extract_names_with_positions(text)


_module_ner = _ModuleNER()
extract_names = _module_ner.extract_names
extract_names_with_positions = _module_ner.extract_names_with_positions
