#!/usr/bin/env bash
# GPU / ANE utilisation per configuration via powermetrics (needs root).
#
#   sudo spikes/01-ocr/measure_gpu.sh [config ...]     # default: all six configurations
#
# Runs a short workload (clean + scan pages, 1 rep) per configuration while
# powermetrics samples every 250 ms, then writes results/gpu/<config>.json with
# mean GPU active residency, GPU power and ANE power. Run it on its own: it is a
# separate pass from the latency benchmark so powermetrics can't skew latency.
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "run with sudo (powermetrics needs root)"; exit 1; }
cd "$(dirname "$0")"
USER_NAME="${SUDO_USER:?run via sudo from your user account}"
UV="$(sudo -u "$USER_NAME" -i which uv)"
mkdir -p results/gpu
configs=("$@")
[[ ${#configs[@]} -gt 0 ]] || configs=(tesseract-cpu paddle-cpu paddle-onnx-cpu paddle-onnx-coreml docling-cpu docling-mps)

for cfg in "${configs[@]}"; do
  engine="${cfg%-*}"; device="${cfg##*-}"
  log="results/gpu/${cfg}.powermetrics.txt"
  echo "== $cfg"
  powermetrics --samplers cpu_power,gpu_power -i 250 > "$log" 2>/dev/null &
  pm=$!
  sleep 1
  sudo -u "$USER_NAME" "$UV" run --project "engines/$engine" python run.py --engine "$engine" --device "$device" \
    --profiles clean scan --reps 1 --warmup 2 --cold-start-runs 0 --tag=-gpu >/dev/null 2>&1
  kill "$pm"; wait "$pm" 2>/dev/null || true
  python3 - "$log" "results/gpu/${cfg}.json" <<'PY'
import json, re, sys
text = open(sys.argv[1]).read()
def grab(pattern):
    vals = [float(v) for v in re.findall(pattern, text)]
    return round(sum(vals) / len(vals), 1) if vals else None
out = {
    "gpu_active_pct_mean": grab(r"GPU HW active residency:\s+([\d.]+)%"),
    "gpu_power_mw_mean": grab(r"GPU Power:\s+([\d.]+) mW"),
    "ane_power_mw_mean": grab(r"ANE Power:\s+([\d.]+) mW"),
    "samples": len(re.findall(r"GPU HW active residency", text)),
}
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(out)
PY
  chown "$USER_NAME" "results/gpu/${cfg}.json" "$log"
done
rm -f results/*-gpu.json
sudo -u "$USER_NAME" python3 report.py >/dev/null && echo "report updated: results/SUMMARY.md"
