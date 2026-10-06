"""Benchmark one NER configuration on ground-truth text and on every OCR engine's output.

    uv run --project spikes/02-ner python spikes/02-ner/run.py --model gliner --backend torch --device cpu

Inputs come from spike 01: data/ground_truth/*.json (exact text + names) and
results/raw/<ocr-config>.jsonl (per-page OCR text). Writes results/<config>.json.
"""

import time

T_PROCESS_START = time.perf_counter()

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
OCR_SPIKE = HERE.parent / "01-ocr"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(OCR_SPIKE))

from common.sampler import ResourceSampler  # noqa: E402
from extractors import create  # noqa: E402
from metrics import aggregate, normalize_name, score_page, split_first_last  # noqa: E402

RESULTS = HERE / "results"
OCR_CONFIGS = ["tesseract-cpu", "paddle-cpu", "paddle-onnx-cpu", "paddle-onnx-coreml", "docling-cpu", "docling-mps"]


def load_inputs(sources: list[str]) -> dict[str, list[dict]]:
    gt_dir = OCR_SPIKE / "data" / "ground_truth"
    gt = {p.stem: json.loads(p.read_text()) for p in gt_dir.glob("*.json")}
    names = {pid: sorted({n["name"] for n in g["names"]}) for pid, g in gt.items()}
    inputs: dict[str, list[dict]] = {}
    if "gt" in sources:
        inputs["gt"] = [{"page_id": pid, "profile": "clean", "text": g["text"], "names": names[pid]}
                        for pid, g in sorted(gt.items()) if g["profile"] == "clean"]
    for cfg in OCR_CONFIGS:
        raw = OCR_SPIKE / "results" / "raw" / f"{cfg}.jsonl"
        if cfg in sources and raw.exists():
            rows = [json.loads(l) for l in raw.read_text().splitlines()]
            inputs[cfg] = [{"page_id": r["page_id"], "profile": gt[r["page_id"]]["profile"], "text": r["text"],
                            "names": names[r["page_id"]]} for r in rows]
    return inputs


def pct(values) -> dict:
    a = np.asarray(values)
    return {"n": len(a), "mean": round(float(a.mean()), 2),
            **{f"p{p}": round(float(np.percentile(a, p)), 2) for p in (50, 90, 95, 99)}}


def cold_start_once(args, out_path: str) -> None:
    t0 = time.perf_counter()
    ex = create(args.model, args.backend, args.device)
    t1 = time.perf_counter()
    ex.extract("Robert Chen has been promoted to Vice President of Engineering at Acme Corporation.")
    t2 = time.perf_counter()
    Path(out_path).write_text(json.dumps({
        "process_to_ready_s": round(t1 - T_PROCESS_START, 3), "model_load_s": round(t1 - t0, 3),
        "first_call_s": round(t2 - t1, 3), "cold_start_total_s": round(t2 - T_PROCESS_START, 3)}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["spacy-sm", "spacy-trf", "bert-ner", "gliner"])
    ap.add_argument("--backend", default="native", choices=["native", "torch", "onnx", "onnx-int8"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--sources", nargs="*", default=["gt", *OCR_CONFIGS])
    ap.add_argument("--latency-reps", type=int, default=20, help="extra timed passes over the clean GT pages")
    ap.add_argument("--cold-start-runs", type=int, default=3)
    ap.add_argument("--tag", default="")
    ap.add_argument("--cold-start-once", metavar="OUT_JSON", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.cold_start_once:
        cold_start_once(args, args.cold_start_once)
        return

    config = f"{args.model}-{args.backend}-{args.device}{args.tag}"
    ex = create(args.model, args.backend, args.device)
    inputs = load_inputs(args.sources)
    ex.extract(inputs["gt"][0]["text"])  # warm-up

    by_source, examples, latencies = {}, {}, []
    with ResourceSampler() as sampler:
        for source, pages in inputs.items():
            scored = []
            for page in pages:
                t = time.perf_counter()
                spans = ex.extract(page["text"])
                latencies.append((time.perf_counter() - t) * 1000)
                s = score_page(set(page["names"]), [sp.text for sp in spans])
                scored.append({**s, "profile": page["profile"]})
                if source == "gt":
                    examples[page["page_id"]] = {
                        "raw_spans": sorted({sp.text for sp in spans}),
                        "first_last": sorted({" | ".join(split_first_last(n)) for n in
                                              {normalize_name(sp.text) for sp in spans}}),
                        "missed": s["missed"], "false_positives": s["false_positives"], "partial": s["partial"],
                    }
            entry = {"overall": aggregate(scored)}
            if source != "gt":
                profiles = sorted({p["profile"] for p in scored})
                entry["by_profile"] = {pr: aggregate([p for p in scored if p["profile"] == pr]) for pr in profiles}
            entry["false_positive_examples"] = sorted({fp for p in scored for fp in p["false_positives"]})[:40]
            by_source[source] = entry
            o = entry["overall"]
            print(f"[{config}] {source}: exact F1 {o['exact']['f1']} fuzzy F1 {o['fuzzy']['f1']} "
                  f"detected F1 {o['detected']['f1']} FP {o['false_positives']}", flush=True)

        clean_latency = []
        for _ in range(args.latency_reps):
            for page in inputs["gt"]:
                t = time.perf_counter()
                ex.extract(page["text"])
                clean_latency.append((time.perf_counter() - t) * 1000)

    cold = []
    for _ in range(args.cold_start_runs):
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            subprocess.run([sys.executable, __file__, "--model", args.model, "--backend", args.backend,
                            "--device", args.device, "--cold-start-once", tmp.name],
                           capture_output=True, text=True, check=True)
            cold.append(json.loads(Path(tmp.name).read_text()))

    out = {
        "config": config, "model": args.model, "backend": args.backend, "device": args.device,
        "info": ex.info(),
        "latency_ms": {"clean_page": pct(clean_latency), "all_calls": pct(latencies)},
        "cold_start": cold, "resources": sampler.summary(),
        "by_source": by_source, "gt_examples": examples,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{config}.json").write_text(json.dumps(out, indent=1))
    print(f"[{config}] clean page p50 {out['latency_ms']['clean_page']['p50']}ms "
          f"p99 {out['latency_ms']['clean_page']['p99']}ms", flush=True)


if __name__ == "__main__":
    main()
