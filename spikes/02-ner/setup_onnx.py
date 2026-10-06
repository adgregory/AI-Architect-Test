"""Export GLiNER to ONNX once (the repo ships no ONNX weights). BERT-NER's repo already has onnx/model.onnx.

    uv run --project spikes/02-ner python spikes/02-ner/setup_onnx.py
"""

from gliner import GLiNER

from extractors import GLINER, ONNX_DIR

out = ONNX_DIR / "gliner"
paths = GLiNER.from_pretrained(GLINER).export_to_onnx(out)
print(paths)
