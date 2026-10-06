"""NER candidates behind one interface: extract(text) -> list of person spans."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ONNX_DIR = Path(__file__).resolve().parent / "onnx"
BERT_NER = "dslim/bert-base-NER"
GLINER = "urchade/gliner_small-v2.1"


@dataclass
class Span:
    text: str
    start: int
    end: int
    score: float | None = None


def windows(text: str, max_words: int = 200) -> list[tuple[int, str]]:
    """Split text into line-aligned windows of <= max_words, keeping char offsets.

    GLiNER silently truncates inputs past its max length (384 words), so long pages
    must be windowed; line boundaries avoid cutting a name in half.
    """
    out, start, count, pos = [], 0, 0, 0
    for line in text.splitlines(keepends=True):
        n = len(line.split())
        if count and count + n > max_words:
            out.append((start, text[start:pos]))
            start, count = pos, 0
        count += n
        pos += len(line)
    if start < len(text):
        out.append((start, text[start:]))
    return out


class SpacyExtractor:
    def __init__(self, package: str):
        import spacy

        self.nlp = spacy.load(package)
        self.package = package

    def extract(self, text: str) -> list[Span]:
        return [Span(e.text, e.start_char, e.end_char) for e in self.nlp(text).ents if e.label_ == "PERSON"]

    def info(self) -> dict:
        import spacy

        return {"version": f"spacy {spacy.__version__}, {self.package} {self.nlp.meta['version']}",
                "onnx": False, "onnx_note": "spaCy/thinc pipelines have no ONNX runtime path."}


class BertNerExtractor:
    def __init__(self, backend: str, device: str):
        from transformers import AutoTokenizer, pipeline

        tok = AutoTokenizer.from_pretrained(BERT_NER)
        if backend == "onnx":
            from optimum.onnxruntime import ORTModelForTokenClassification

            model = ORTModelForTokenClassification.from_pretrained(BERT_NER, subfolder="onnx", file_name="model.onnx")
            self.pipe = pipeline("token-classification", model=model, tokenizer=tok,
                                 aggregation_strategy="first", stride=64)
        else:
            self.pipe = pipeline("token-classification", model=BERT_NER, tokenizer=tok, device=device,
                                 aggregation_strategy="first", stride=64)
        self.backend, self.device = backend, device

    def extract(self, text: str) -> list[Span]:
        return [Span(text[e["start"]:e["end"]], e["start"], e["end"], float(e["score"]))
                for e in self.pipe(text) if e["entity_group"] == "PER"]

    def info(self) -> dict:
        import transformers

        return {"version": f"transformers {transformers.__version__}, {BERT_NER}",
                "onnx": self.backend == "onnx",
                "onnx_note": "Repo ships onnx/model.onnx; run with optimum ORTModelForTokenClassification (CPU EP)."}


class GlinerExtractor:
    LABELS = ["person"]
    THRESHOLD = float(os.environ.get("GLINER_THRESHOLD", "0.5"))

    def __init__(self, backend: str, device: str):
        from gliner import GLiNER

        if backend in ("onnx", "onnx-int8"):
            onnx_file = "model_quantized.onnx" if backend == "onnx-int8" else "model.onnx"
            self.model = GLiNER.from_pretrained(str(ONNX_DIR / "gliner"), load_onnx_model=True,
                                                onnx_model_file=onnx_file)
        else:
            self.model = GLiNER.from_pretrained(GLINER).to(device)
        self.backend, self.device = backend, device

    def extract(self, text: str) -> list[Span]:
        spans = []
        for offset, chunk in windows(text):
            for e in self.model.predict_entities(chunk, self.LABELS, threshold=self.THRESHOLD):
                spans.append(Span(e["text"], offset + e["start"], offset + e["end"], float(e["score"])))
        return spans

    def info(self) -> dict:
        from importlib.metadata import version

        return {"version": f"gliner {version('gliner')}, {GLINER}, labels {self.LABELS}, threshold {self.THRESHOLD}",
                "onnx": self.backend.startswith("onnx"),
                "onnx_note": "No ONNX in the repo; exported locally with GLiNER.export_to_onnx (setup_onnx.py); "
                             "onnx-int8 = dynamic quantization (QUInt8 weights)."}


def create(model: str, backend: str, device: str):
    if model == "spacy-sm":
        return SpacyExtractor("en_core_web_sm")
    if model == "spacy-trf":
        return SpacyExtractor("en_core_web_trf")
    if model == "bert-ner":
        return BertNerExtractor(backend, device)
    if model == "gliner":
        return GlinerExtractor(backend, device)
    raise ValueError(model)
