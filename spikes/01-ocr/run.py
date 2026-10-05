"""Benchmark one OCR configuration (engine × device) over the spike dataset.

Run inside the engine's own environment, e.g.:
    uv run --project spikes/01-ocr/engines/tesseract \
        python spikes/01-ocr/run.py --engine tesseract --device cpu

Writes results/<engine>-<device>.json (committed) and results/raw/<engine>-<device>.jsonl
(per-page OCR output, git-ignored).
"""

import time

T_PROCESS_START = time.perf_counter()  # before heavy imports: cold start includes them

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

SPIKE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SPIKE_DIR))

from common.metrics import cer_wer, name_localization, name_recall  # noqa: E402
from common.sampler import ResourceSampler  # noqa: E402

DATA_DIR = SPIKE_DIR / "data"
RESULTS_DIR = SPIKE_DIR / "results"


def load_engine(engine: str, device: str):
    path = SPIKE_DIR / "engines" / engine / "adapter.py"
    spec = importlib.util.spec_from_file_location(f"adapter_{engine.replace('-', '_')}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create(device)


def load_pages(profiles: list[str] | None, limit: int | None) -> list[dict]:
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    if profiles:
        manifest = [m for m in manifest if m["profile"] in profiles]
    if limit:
        manifest = manifest[:limit]
    pages = []
    for m in manifest:
        gt = json.loads((DATA_DIR / "ground_truth" / f"{m['page_id']}.json").read_text())
        gt["pil"] = Image.open(DATA_DIR / gt["image"]).convert("RGB")
        gt["pil"].load()
        pages.append(gt)
    return pages


def percentiles(values_ms: list[float]) -> dict:
    a = np.asarray(values_ms)
    return {
        "n": len(a),
        "mean": round(float(a.mean()), 1),
        **{f"p{p}": round(float(np.percentile(a, p)), 1) for p in (50, 90, 95, 99)},
        "max": round(float(a.max()), 1),
    }


def mean(xs):
    xs = list(xs)
    return round(sum(xs) / len(xs), 4) if xs else None


def cold_start_once(engine_name: str, device: str) -> None:
    """Fresh-process measurement: imports + model load + first page."""
    page = load_pages(["clean"], 1)[0]
    engine = load_engine(engine_name, device)
    t0 = time.perf_counter()
    engine.load()
    t1 = time.perf_counter()
    engine.ocr(page["pil"])
    t2 = time.perf_counter()
    print(json.dumps({
        "process_to_ready_s": round(t1 - T_PROCESS_START, 3),
        "model_load_s": round(t1 - t0, 3),
        "first_page_s": round(t2 - t1, 3),
        "cold_start_total_s": round(t2 - T_PROCESS_START, 3),
    }))


def cold_starts(engine_name: str, device: str, runs: int) -> list[dict]:
    out = []
    for i in range(runs):
        proc = subprocess.run(
            [sys.executable, __file__, "--engine", engine_name, "--device", device, "--cold-start-once"],
            capture_output=True, text=True, check=True,
        )
        out.append(json.loads(proc.stdout.strip().splitlines()[-1]))
        print(f"  cold start {i + 1}/{runs}: {out[-1]['cold_start_total_s']}s", flush=True)
    return out


def summarize(records: list[dict], key_fn) -> dict:
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(key_fn(r), []).append(r)
    out = {}
    for key, rs in sorted(groups.items()):
        acc = [r for r in rs if r["rep"] == 0]
        entry = {"latency_ms": percentiles([r["latency_ms"] for r in rs])}
        if acc:
            entry.update({
                "pages": len(acc),
                "cer": mean(r["cer"] for r in acc),
                "wer": mean(r["wer"] for r in acc),
                "name_recall": mean(r["name_recall"] for r in acc),
                "name_recall_ci": mean(r["name_recall_ci"] for r in acc),
            })
            for gran in acc[0]["name_iou"]:
                ious = [v for r in acc for v in r["name_iou"][gran]]
                entry[f"name_iou_{gran}"] = {
                    "mean": mean(ious),
                    "share_ge_0.5": mean(v >= 0.5 for v in ious),
                }
        out[key] = entry
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--cold-start-runs", type=int, default=3)
    ap.add_argument("--profiles", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tag", default="", help="suffix for result file names (smoke tests)")
    ap.add_argument("--cold-start-once", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.cold_start_once:
        cold_start_once(args.engine, args.device)
        return

    config = f"{args.engine}-{args.device}{args.tag}"
    pages = load_pages(args.profiles, args.limit)
    print(f"[{config}] {len(pages)} pages × {args.reps} reps", flush=True)

    engine = load_engine(args.engine, args.device)
    t0 = time.perf_counter()
    engine.load()
    load_s = time.perf_counter() - t0

    for page in pages[: args.warmup]:
        engine.ocr(page["pil"])

    RESULTS_DIR.joinpath("raw").mkdir(parents=True, exist_ok=True)
    records, raw = [], []
    with ResourceSampler(gpu_memory=engine.gpu_memory_bytes) as sampler:
        t_run = time.perf_counter()
        for rep in range(args.reps):
            for i, page in enumerate(pages):
                t = time.perf_counter()
                result = engine.ocr(page["pil"])
                latency_ms = (time.perf_counter() - t) * 1000
                rec = {"page_id": page["page_id"], "profile": page["profile"], "rep": rep,
                       "latency_ms": round(latency_ms, 2)}
                if rep == 0:
                    cer, wer = cer_wer(page["text"], result.text)
                    recall, recall_ci = name_recall(sorted({n["name"] for n in page["names"]}), result.text)
                    rec.update({
                        "cer": round(cer, 4), "wer": round(wer, 4),
                        "name_recall": round(recall, 4), "name_recall_ci": round(recall_ci, 4),
                        "name_iou": {g: [round(v, 3) for v in name_localization(page["names"], els, g)]
                                     for g, els in result.elements.items()},
                    })
                    raw.append({"page_id": page["page_id"], "text": result.text,
                                "elements": {g: [asdict(e) for e in els] for g, els in result.elements.items()}})
                records.append(rec)
                if (i + 1) % 25 == 0:
                    print(f"  rep {rep} page {i + 1}/{len(pages)}", flush=True)
        wall_s = time.perf_counter() - t_run

    print(f"[{config}] measuring cold start", flush=True)
    cold = cold_starts(args.engine, args.device, args.cold_start_runs) if args.cold_start_runs else []

    out = {
        "config": config,
        "engine": args.engine,
        "device": args.device,
        "info": engine.info(),
        "host": {"machine": platform.machine(), "system": platform.platform(), "python": platform.python_version()},
        "run": {"pages": len(pages), "reps": args.reps, "warmup": args.warmup,
                "in_process_model_load_s": round(load_s, 3), "wall_s": round(wall_s, 1),
                "pages_per_minute": round(len(records) / wall_s * 60, 1)},
        "cold_start": cold,
        "resources": sampler.summary(),
        "overall": summarize(records, lambda r: "all")["all"],
        "by_profile": summarize(records, lambda r: r["profile"]),
        "pages": records,
    }
    (RESULTS_DIR / f"{config}.json").write_text(json.dumps(out, indent=1))
    with (RESULTS_DIR / "raw" / f"{config}.jsonl").open("w") as f:
        for r in raw:
            f.write(json.dumps(r) + "\n")
    o = out["overall"]
    print(f"[{config}] CER {o['cer']} WER {o['wer']} names {o['name_recall']} "
          f"p50 {o['latency_ms']['p50']}ms p99 {o['latency_ms']['p99']}ms", flush=True)


if __name__ == "__main__":
    main()
