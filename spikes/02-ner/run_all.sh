#!/usr/bin/env bash
# Run every NER configuration sequentially. Needs spike 01's dataset and raw OCR outputs.
set -euo pipefail
cd "$(dirname "$0")"
[[ -f onnx/gliner/model_quantized.onnx ]] || uv run python setup_onnx.py
for cfg in "spacy-sm native cpu" "spacy-trf native cpu" \
           "bert-ner torch cpu" "bert-ner torch mps" "bert-ner onnx cpu" \
           "gliner torch cpu" "gliner torch mps" "gliner onnx cpu" "gliner onnx-int8 cpu"; do
  read -r model backend device <<<"$cfg"
  [[ -f "results/${model}-${backend}-${device}.json" && "${FORCE:-0}" != 1 ]] && { echo "skip ${model}-${backend}-${device}"; continue; }
  uv run python run.py --model "$model" --backend "$backend" --device "$device" "$@" 2>&1 \
    | grep --line-buffered -E "^\[|Traceback|Error" || echo "FAILED ${model}-${backend}-${device}"
done
