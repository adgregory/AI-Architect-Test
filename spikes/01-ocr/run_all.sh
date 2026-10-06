#!/usr/bin/env bash
# Run every benchmark configuration sequentially (never in parallel: it would skew latency).
set -euo pipefail
cd "$(dirname "$0")"
for cfg in "tesseract cpu" "paddle cpu" "paddle-onnx cpu" "paddle-onnx coreml" "docling cpu" "docling mps"; do
  read -r engine device <<<"$cfg"
  if [[ -f "results/${engine}-${device}.json" && "${FORCE:-0}" != 1 ]]; then
    echo "skip ${engine}-${device} (exists)"; continue
  fi
  uv run --project "engines/${engine}" python run.py --engine "$engine" --device "$device" "$@" 2>&1 \
    | grep --line-buffered -vE "E5RT|Warning|warn\(|Progress|Loading weights|quantize"
done
python3 report.py >/dev/null && echo "report written: results/SUMMARY.md"
