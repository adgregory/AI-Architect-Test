#!/usr/bin/env bash
# Run every embedding configuration sequentially (model × torch-cpu / torch-mps / onnx-cpu).
set -euo pipefail
cd "$(dirname "$0")"
for model in minilm mpnet bge-small bge-base e5-base nomic arctic-m; do
  for bd in "torch cpu" "torch mps" "onnx cpu"; do
    read -r backend device <<<"$bd"
    [[ -f "results/${model}-${backend}-${device}.json" && "${FORCE:-0}" != 1 ]] && { echo "skip ${model}-${backend}-${device}"; continue; }
    uv run python run.py --model "$model" --backend "$backend" --device "$device" "$@" 2>&1 \
      | grep --line-buffered -E "^\[|Traceback|Error" || echo "FAILED ${model}-${backend}-${device}"
  done
done
