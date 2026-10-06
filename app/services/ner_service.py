"""Named-entity recognition: person names in text."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from app.core.factories import default_spacy_model
from app.core.lazy import Lazy

PERSON_LABEL = "PERSON"


@runtime_checkable
class NERService(Protocol):
    """Swappable person-name extractor."""

    def extract_names(self, text: str) -> list[str]: ...

    def extract_names_with_positions(self, text: str) -> list[dict]: ...


class SpacyNERService:
    """spaCy pipeline; keeps PERSON entities only. The pipeline is injected."""

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


# --------------------------------------------------------------------------- #
# Functional API (backwards compatible). `nlp` is a lazy proxy: the model loads
# on first use, not at import. The adapter resolves `nlp` at call time so it
# can be replaced (e.g. in tests).
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
