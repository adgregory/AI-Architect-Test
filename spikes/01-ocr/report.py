"""Aggregate results/*.json (+ disk.json, gpu/*.json if present) into results/SUMMARY.md.

    python3 spikes/01-ocr/report.py      # stdlib only
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
ORDER = ["tesseract-cpu", "paddle-cpu", "paddle-onnx-cpu", "paddle-onnx-coreml", "docling-cpu", "docling-mps"]
PROFILES = ["clean", "noise", "skew", "blur", "jpeg", "low_dpi", "scan"]


def load() -> list[dict]:
    found = {p.stem: json.loads(p.read_text()) for p in RESULTS.glob("*.json")
             if not p.stem.endswith("-smoke") and p.stem not in ("disk",)}
    return [found[c] for c in ORDER if c in found]


def pct(x) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def best_iou(entry: dict, gran: str) -> str:
    v = entry.get(f"name_iou_{gran}")
    return "—" if not v else f"{v['mean']:.2f} / {pct(v['share_ge_0.5'])}"


def main() -> None:
    runs = load()
    if not runs:
        raise SystemExit("no results yet")
    disk = json.loads((RESULTS / "disk.json").read_text()) if (RESULTS / "disk.json").exists() else {}
    gpu_dir = RESULTS / "gpu"
    gpu = {p.stem: json.loads(p.read_text()) for p in gpu_dir.glob("*.json")} if gpu_dir.exists() else {}

    parts = ["# Spike 01 — OCR results\n",
             f"Host: {runs[0]['host']['system']}, Python {runs[0]['host']['python']}. "
             f"{runs[0]['run']['pages']} pages × {runs[0]['run']['reps']} reps per configuration "
             "(accuracy from rep 0; latency over all reps, after warm-up).\n"]

    parts.append("## Accuracy (all pages)\n")
    parts.append(table(
        ["Config", "CER", "WER", "Name recall", "Name recall (case-insens.)",
         "Name IoU word (mean / ≥0.5)", "Name IoU line (mean / ≥0.5)"],
        [[r["config"], pct(r["overall"]["cer"]), pct(r["overall"]["wer"]), pct(r["overall"]["name_recall"]),
          pct(r["overall"]["name_recall_ci"]), best_iou(r["overall"], "word"), best_iou(r["overall"], "line")]
         for r in runs]))

    parts.append("\n\n## CER by degradation profile\n")
    parts.append(table(["Config", *PROFILES],
                       [[r["config"], *(pct(r["by_profile"].get(p, {}).get("cer")) for p in PROFILES)] for r in runs]))
    parts.append("\n\n## Name recall by degradation profile\n")
    parts.append(table(["Config", *PROFILES],
                       [[r["config"], *(pct(r["by_profile"].get(p, {}).get("name_recall")) for p in PROFILES)]
                        for r in runs]))

    parts.append("\n\n## Latency per page (ms, warm)\n")
    parts.append(table(
        ["Config", "mean", "p50", "p90", "p95", "p99", "max", "pages/min", "cold start (s, median)"],
        [[r["config"], *(r["overall"]["latency_ms"][k] for k in ("mean", "p50", "p90", "p95", "p99", "max")),
          r["run"]["pages_per_minute"],
          sorted(c["cold_start_total_s"] for c in r["cold_start"])[len(r["cold_start"]) // 2] if r["cold_start"] else "—"]
         for r in runs]))

    parts.append("\n\n## Resources during the warm run\n")
    rows = []
    for r in runs:
        res = r["resources"]
        g = gpu.get(r["config"], {})
        rows.append([r["config"], res["cpu_pct_mean"], res["cpu_pct_peak"], res["rss_peak_mb"],
                     res["gpu_mem_peak_mb"] or "—", g.get("gpu_active_pct_mean", "—"), g.get("gpu_power_mw_mean", "—")])
    parts.append(table(["Config", "CPU % mean (100 = 1 core)", "CPU % peak", "RSS peak (MB)",
                        "GPU mem peak (MB)", "GPU active % (powermetrics)", "GPU power mW"], rows))

    if disk:
        parts.append("\n\n## Disk footprint\n")
        parts.append(table(["Engine", "Python env (MB)", "Models (MB)", "System deps (MB)", "Docker image (MB)"],
                           [[e, d.get("env_mb", "—"), d.get("models_mb", "—"), d.get("system_mb", "—"),
                             d.get("docker_image_mb", "—")] for e, d in disk.items()]))

    parts.append("\n\n## Engine details\n")
    for r in runs:
        i = r["info"]
        parts.append(f"### {r['config']}\n")
        parts.append(f"- **Version:** {i.get('engine_version')}\n- **Models:** {i.get('models')}\n"
                     f"- **ONNX:** {i.get('onnx')} — {i.get('onnx_note')}\n- **Device used:** {i.get('device_used')}"
                     + (f" (providers: {i['session_providers']})" if i.get("session_providers") else "")
                     + f"\n- **Positional output:** {i.get('positional_output')}\n")

    (RESULTS / "SUMMARY.md").write_text("\n".join(parts) + "\n")
    print((RESULTS / "SUMMARY.md").read_text())


if __name__ == "__main__":
    main()
