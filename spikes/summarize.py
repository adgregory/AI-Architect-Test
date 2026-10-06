"""Collect each spike's results into one committed file per spike: spikes/<spike>/summary.json.

Each run writes a full JSON per configuration (per-page and per-pair detail) into its spike's
results/ directory; those files are git-ignored. The summary keeps the aggregate metrics, run
settings and environment of every configuration and drops the per-item detail.
(Spike 04 writes a single results.json already and is left as is.)

    python3 spikes/summarize.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent
# Per-item detail: large, and only needed to debug a run.
DETAIL_KEYS = {
    "pages",  # 01: one entry per page × rep
    "gt_examples",  # 02: spans found on each ground-truth page
    "false_positive_examples",  # 02
    "pairs",  # 03: every scored sentence pair
    "misses_beyond_top3",  # 03
    "guard_blocks_true_paraphrases",  # 04: each blocked paraphrase
    "guard_lets_through_negatives",  # 04
}


def strip(value):
    if isinstance(value, dict):
        return {k: strip(v) for k, v in value.items() if k not in DETAIL_KEYS}
    if isinstance(value, list):
        return [strip(v) for v in value]
    return value


def load(path: Path) -> dict:
    return strip(json.loads(path.read_text()))


def configs(spike: str) -> dict:
    results = ROOT / spike / "results"
    return {p.stem: load(p) for p in sorted(results.glob("*.json")) if not p.stem.endswith("-smoke")}


def dumps(value, depth: int = 0, expand: int = 3) -> str:
    """Indented down to one line per metric group (configs → config → key), compact below,
    so the file diffs and reads by configuration rather than by number."""
    if depth >= expand or not isinstance(value, dict) or not value:
        return json.dumps(value, separators=(", ", ": "))
    pad = "  " * (depth + 1)
    items = [f"{pad}{json.dumps(k)}: {dumps(v, depth + 1, expand)}" for k, v in value.items()]
    return "{\n" + ",\n".join(items) + "\n" + "  " * depth + "}"


def main() -> None:
    summaries = {
        "01-ocr": {
            "configs": configs("01-ocr"),
            "gpu": {p.stem: load(p) for p in sorted((ROOT / "01-ocr/results/gpu").glob("*.json"))},
        },
        "02-ner": {"configs": configs("02-ner")},
        "03-embeddings": {"configs": configs("03-embeddings")},
    }
    for spike, summary in summaries.items():
        out = ROOT / spike / "summary.json"
        out.write_text(dumps(summary) + "\n")
        print(f"wrote {out.relative_to(ROOT.parent)} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
