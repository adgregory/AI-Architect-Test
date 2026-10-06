"""Export GLiNER to ONNX once (the repo ships no ONNX weights), plus a dynamic int8 variant.
BERT-NER's repo already has onnx/model.onnx.

    uv run --project spikes/02-ner python spikes/02-ner/setup_onnx.py
"""

from gliner import GLiNER
from onnxruntime.quantization import QuantType, quantize_dynamic

from extractors import GLINER, ONNX_DIR

out = ONNX_DIR / "gliner"
if not (out / "model.onnx").exists():
    print(GLiNER.from_pretrained(GLINER).export_to_onnx(out))
if not (out / "model_quantized.onnx").exists():
    # Same scheme GLiNER's exporter uses (quantize=True): int8 weights, activations quantized at runtime.
    quantize_dynamic(model_input=str(out / "model.onnx"), model_output=str(out / "model_quantized.onnx"),
                     weight_type=QuantType.QUInt8)
for f in ("model.onnx", "model_quantized.onnx"):
    print(f, round((out / f).stat().st_size / 2**20, 1), "MB")
