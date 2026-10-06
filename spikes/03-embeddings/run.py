"""Benchmark one embedding configuration (model × backend × device).

    uv run --project spikes/03-embeddings python spikes/03-embeddings/run.py \
        --model bge-small --backend torch --device cpu

backend: torch | onnx (ONNX Runtime, CPU).  device: cpu | mps (torch only).
Writes results/<model>-<backend>-<device>.json.
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
sys.path.insert(0, str(HERE))

from chunks import load_corpus, load_questions  # noqa: E402
from models import MODELS  # noqa: E402

RESULTS = HERE / "results"


class FastEmbedModel:
    """Adapter giving fastembed's TextEmbedding the subset of the SentenceTransformer API we use."""

    def __init__(self, name: str):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(name)
        self.max_seq_length = None

    def encode(self, texts, batch_size=32, normalize_embeddings=True, **_):
        vecs = np.array(list(self._model.embed(texts, batch_size=batch_size)), dtype=np.float32)
        if normalize_embeddings:
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs

    def get_sentence_embedding_dimension(self) -> int:
        return int(self.encode(["x"]).shape[1])


def load_model(spec, backend: str, device: str):
    if backend == "fastembed":
        return FastEmbedModel(spec.hf_name)

    from sentence_transformers import SentenceTransformer

    kwargs = {"device": device, "trust_remote_code": spec.trust_remote_code}
    if backend == "onnx":
        kwargs["backend"] = "onnx"
        kwargs["model_kwargs"] = {"provider": "CPUExecutionProvider"}
    return SentenceTransformer(spec.hf_name, **kwargs)


def encode(model, texts: list[str], prefix: str = "", batch_size: int = 32) -> np.ndarray:
    return model.encode([prefix + t for t in texts], batch_size=batch_size,
                        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)


# --------------------------------------------------------------------------- #
# Quality
# --------------------------------------------------------------------------- #
def load_pairs() -> tuple[list[dict], list[dict]]:
    triplets = [json.loads(l) for l in (HERE / "data/triplets.jsonl").read_text().splitlines()]
    pairs = [json.loads(l) for l in (HERE / "data/pairs.jsonl").read_text().splitlines()]
    graded = [{"a": p["a"], "b": p["b"], "grade": p["grade"], "kind": "graded"} for p in pairs]
    for t in triplets:
        graded.append({"a": t["anchor"], "b": t["positive"], "grade": 3, "kind": "triplet_pos", "category": t["category"]})
        graded.append({"a": t["anchor"], "b": t["negative"], "grade": 1, "kind": "hard_neg", "category": t["category"]})
    return triplets, graded


def quality(model, spec) -> dict:
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    triplets, graded = load_pairs()
    a = encode(model, [g["a"] for g in graded], spec.symmetric_prefix)
    b = encode(model, [g["b"] for g in graded], spec.symmetric_prefix)
    cos = (a * b).sum(axis=1)
    for g, c in zip(graded, cos):
        g["cos"] = float(c)

    grades = np.array([g["grade"] for g in graded])
    pos = np.array([g["cos"] for g in graded if g["grade"] == 3])
    hard = np.array([g["cos"] for g in graded if g["kind"] == "hard_neg"])
    unrelated = np.array([g["cos"] for g in graded if g["grade"] == 0])
    pooled_sd = float(np.sqrt((pos.var(ddof=1) + hard.var(ddof=1)) / 2))

    by_cat: dict[str, list[bool]] = {}
    pos_of = {(g["a"], g["category"]): g["cos"] for g in graded if g["kind"] == "triplet_pos"}
    for g in graded:
        if g["kind"] == "hard_neg":
            by_cat.setdefault(g["category"], []).append(pos_of[(g["a"], g["category"])] > g["cos"])

    mean_cos_by_grade = {str(k): round(float(cos[grades == k].mean()), 4) for k in (0, 1, 2, 3)}
    return {
        "spearman": round(float(spearmanr(cos, grades).statistic), 4),
        "auc_paraphrase_vs_hard_neg": round(float(roc_auc_score(
            np.r_[np.ones(len(pos)), np.zeros(len(hard))], np.r_[pos, hard])), 4),
        "separation_margin_sd": round(float((pos.mean() - hard.mean()) / pooled_sd), 3),
        "mean_cos": {"paraphrase": round(float(pos.mean()), 4), "hard_negative": round(float(hard.mean()), 4),
                     "unrelated": round(float(unrelated.mean()), 4), "by_grade": mean_cos_by_grade},
        "triplet_accuracy": {
            "all": round(float(np.mean([v for vs in by_cat.values() for v in vs])), 4),
            **{k: round(float(np.mean(v)), 4) for k, v in sorted(by_cat.items())},
        },
        "pairs": [{k: g[k] for k in ("a", "b", "grade", "kind", "cos") if k in g} | ({"category": g["category"]} if "category" in g else {})
                  for g in graded],
    }


def retrieval(model, spec) -> dict:
    corpus = load_corpus()
    questions = load_questions(corpus)
    docs = encode(model, [c["text"] for c in corpus], spec.passage_prefix)
    qs = encode(model, [q["question"] for q in questions], spec.query_prefix)
    sims = qs @ docs.T
    ranks, misses = [], []
    for q, row in zip(questions, sims):
        order = [corpus[i]["id"] for i in np.argsort(-row)]
        rank = order.index(q["gold"]) + 1
        ranks.append(rank)
        if rank > 3:
            misses.append({"id": q["id"], "question": q["question"], "gold": q["gold"], "rank": rank, "top1": order[0]})
    ranks = np.array(ranks)
    return {
        "chunks": len(corpus), "questions": len(questions),
        "recall@1": round(float((ranks <= 1).mean()), 4),
        "recall@3": round(float((ranks <= 3).mean()), 4),
        "mrr@10": round(float(np.where(ranks <= 10, 1 / ranks, 0).mean()), 4),
        "misses_beyond_top3": misses,
    }


# --------------------------------------------------------------------------- #
# Performance
# --------------------------------------------------------------------------- #
def pct(values_ms) -> dict:
    a = np.asarray(values_ms)
    return {"n": len(a), "mean": round(float(a.mean()), 2),
            **{f"p{p}": round(float(np.percentile(a, p)), 2) for p in (50, 90, 95, 99)}}


def performance(model, spec, reps: int) -> dict:
    import psutil

    corpus = [c["text"] for c in load_corpus()]
    questions = [q["question"] for q in load_questions(load_corpus())]
    for _ in range(3):  # warm-up
        encode(model, questions[:4], spec.query_prefix)
        encode(model, corpus, spec.passage_prefix)
    single, batch = [], []
    for _ in range(reps):
        for q in questions:
            t = time.perf_counter()
            encode(model, [q], spec.query_prefix)
            single.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        encode(model, corpus, spec.passage_prefix)
        batch.append((time.perf_counter() - t) * 1000)
    return {
        "query_latency_ms": pct(single),
        "chunk_batch_latency_ms": pct(batch),
        "chunks_per_second": round(len(corpus) / (np.median(batch) / 1000), 1),
        "rss_mb": round(psutil.Process().memory_info().rss / 2**20, 1),
    }


def cold_start_once(args, out_path: str) -> None:
    spec = MODELS[args.model]
    t0 = time.perf_counter()
    model = load_model(spec, args.backend, args.device)
    t1 = time.perf_counter()
    encode(model, ["Who is the new head of engineering?"], spec.query_prefix)
    t2 = time.perf_counter()
    Path(out_path).write_text(json.dumps({
        "process_to_ready_s": round(t1 - T_PROCESS_START, 3), "model_load_s": round(t1 - t0, 3),
        "first_query_s": round(t2 - t1, 3), "cold_start_total_s": round(t2 - T_PROCESS_START, 3)}))


def cold_starts(args, runs: int) -> list[dict]:
    out = []
    for _ in range(runs):
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            subprocess.run([sys.executable, __file__, "--model", args.model, "--backend", args.backend,
                            "--device", args.device, "--cold-start-once", tmp.name],
                           capture_output=True, text=True, check=True)
            out.append(json.loads(Path(tmp.name).read_text()))
    return out


def model_disk_mb(spec, backend: str) -> float | None:
    """Size of the weight file this backend loads (repos ship both torch and ONNX weights)."""
    from huggingface_hub import scan_cache_dir

    wanted = "onnx/model.onnx" if backend == "onnx" else "model.safetensors"
    for repo in scan_cache_dir().repos:
        if repo.repo_id == spec.hf_name:
            sizes = [f.size_on_disk for rev in repo.revisions for f in rev.files if f.file_name and
                     str(f.file_path).endswith(wanted)]
            return round(max(sizes) / 2**20, 1) if sizes else None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--backend", default="torch", choices=["torch", "onnx", "fastembed"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--cold-start-runs", type=int, default=3)
    ap.add_argument("--cold-start-once", metavar="OUT_JSON", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.backend in ("onnx", "fastembed") and args.device != "cpu":
        ap.error(f"{args.backend} backend is benchmarked on cpu only")

    if args.cold_start_once:
        cold_start_once(args, args.cold_start_once)
        return

    spec = MODELS[args.model]
    config = f"{args.model}-{args.backend}-{args.device}"
    print(f"[{config}] loading", flush=True)
    model = load_model(spec, args.backend, args.device)
    q = quality(model, spec)
    r = retrieval(model, spec)
    perf = performance(model, spec, args.reps)
    cold = cold_starts(args, args.cold_start_runs) if args.cold_start_runs else []
    out = {
        "config": config, "model": spec.id, "hf_name": spec.hf_name, "backend": args.backend, "device": args.device,
        "dim": model.get_sentence_embedding_dimension(), "max_seq_length": model.max_seq_length,
        "prefixes": {"query": spec.query_prefix, "passage": spec.passage_prefix, "symmetric": spec.symmetric_prefix},
        "weights_disk_mb": model_disk_mb(spec, args.backend),
        "quality": q, "retrieval": r, "performance": perf, "cold_start": cold,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{config}.json").write_text(json.dumps(out, indent=1))
    print(f"[{config}] spearman {q['spearman']} auc {q['auc_paraphrase_vs_hard_neg']} "
          f"margin {q['separation_margin_sd']} triplet {q['triplet_accuracy']['all']} "
          f"R@3 {r['recall@3']} MRR {r['mrr@10']} query p50 {perf['query_latency_ms']['p50']}ms", flush=True)


if __name__ == "__main__":
    main()
