"""Measure disk footprint per engine: Python env, model files, system deps, Docker image.

    python3 spikes/01-ocr/measure_disk.py [--docker]

--docker builds docker/Dockerfile per engine (CPU-only, linux/arm64 on this host) and
records the image size. Writes results/disk.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

SPIKE = Path(__file__).resolve().parent
HOME = Path.home()
RAPIDOCR_MODELS = SPIKE / "engines/paddle-onnx/.venv/lib/python3.12/site-packages/rapidocr/models"

# Model files each engine actually loads, wherever the engine caches them.
MODELS = {
    "tesseract": [Path("/opt/homebrew/share/tessdata/eng.traineddata")],
    "paddle": [HOME / ".paddlex/official_models/PP-OCRv5_mobile_det",
               HOME / ".paddlex/official_models/en_PP-OCRv5_mobile_rec"],
    "paddle-onnx": [RAPIDOCR_MODELS / "ch_PP-OCRv5_det_mobile.onnx",
                    RAPIDOCR_MODELS / "en_PP-OCRv5_rec_mobile.onnx"],
    "docling": [HOME / ".cache/docling/models", HOME / ".EasyOCR/model"],
}
# Native (non-Python) dependencies installed outside the env.
SYSTEM = {"tesseract": [Path("/opt/homebrew/Cellar/tesseract"), Path("/opt/homebrew/Cellar/leptonica")]}
APT = {"tesseract": "tesseract-ocr tesseract-ocr-eng", "paddle": "libgomp1 libgl1 libglib2.0-0",
       "paddle-onnx": "libgl1 libglib2.0-0", "docling": "libgl1 libglib2.0-0"}


def du_mb(paths: list[Path]) -> float | None:
    existing = [str(p) for p in paths if p.exists()]
    if not existing:
        return None
    out = subprocess.run(["du", "-skL", *existing], capture_output=True, text=True, check=True).stdout
    return round(sum(int(line.split()[0]) for line in out.splitlines()) / 1024, 1)


def docker_size_mb(engine: str) -> float:
    tag = f"ocr-spike-{engine}"
    subprocess.run(["docker", "build", "-f", "docker/Dockerfile", "--build-arg", f"ENGINE={engine}",
                    "--build-arg", f"APT_PACKAGES={APT[engine]}", "-t", tag, "."], cwd=SPIKE, check=True)
    size = subprocess.run(["docker", "image", "inspect", tag, "--format", "{{.Size}}"],
                          capture_output=True, text=True, check=True).stdout
    return round(int(size) / 2**20, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docker", action="store_true")
    args = ap.parse_args()

    path = SPIKE / "results" / "disk.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    for engine in MODELS:
        entry = out.get(engine, {})
        entry.update({
            "env_mb": du_mb([SPIKE / "engines" / engine / ".venv"]),
            "models_mb": du_mb(MODELS[engine]),
            "system_mb": du_mb(SYSTEM.get(engine, [])),
        })
        if args.docker:
            entry["docker_image_mb"] = docker_size_mb(engine)
            entry["docker_platform"] = "linux/arm64, CPU-only, python:3.12-slim, models baked in"
        out[engine] = entry
        print(engine, entry, flush=True)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
