"""Spike 04 — answer-cache threshold and question guard, on spike 03's labelled pairs.

Each pair is (cached question, new question, should_hit). Paraphrases should hit; hard
negatives (negation, antonym, entity swap, number change, role swap) and lower-graded pairs
must miss. Uses the app's own embedding service (bge-small via fastembed, query prefix) and
the app's QuestionGuard.

    uv run python spikes/04-answer-cache/evaluate.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.core.config import Settings  # noqa: E402
from app.core.factories import EmbeddingServiceFactory  # noqa: E402
from app.services.cache_service import QuestionGuard  # noqa: E402

DATA = ROOT / "spikes" / "03-embeddings" / "data"
THRESHOLDS = [0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96]


def load_cases() -> list[dict]:
    cases = []
    for line in (DATA / "triplets.jsonl").read_text().splitlines():
        t = json.loads(line)
        cases.append({"a": t["anchor"], "b": t["positive"], "hit": True, "kind": "paraphrase"})
        cases.append({"a": t["anchor"], "b": t["negative"], "hit": False, "kind": t["category"]})
    for line in (DATA / "pairs.jsonl").read_text().splitlines():
        p = json.loads(line)
        cases.append({"a": p["a"], "b": p["b"], "hit": p["grade"] == 3, "kind": f"grade{p['grade']}"})
    return cases


# How users actually repeat a question: same intent, small surface changes.
REPHRASINGS = [
    lambda q: q.lower(),
    lambda q: q.rstrip("?") + ", please?",
    lambda q: "Can you tell me " + q[0].lower() + q[1:],
    lambda q: q.replace("Who is", "Who's").replace("What is", "What's").replace("?", " ?"),
    lambda q: "  " + q.replace(" the ", " the  ") + "  ",
]


def load_repeated_questions() -> list[dict]:
    qs = [json.loads(line)["question"] for line in (DATA / "questions.jsonl").read_text().splitlines()]
    return [{"a": q, "b": f(q), "hit": True, "kind": "repeat"} for q in qs for f in REPHRASINGS if f(q) != q]


def main() -> None:
    settings = Settings(_env_file=None)
    embeddings = EmbeddingServiceFactory.create(settings)
    guard = QuestionGuard()
    cases = load_cases() + load_repeated_questions()
    for c in cases:
        va, vb = np.array(embeddings.embed_query(c["a"])), np.array(embeddings.embed_query(c["b"]))
        c["cos"] = float(va @ vb)
        c["guard_ok"] = guard.same_meaning(c["a"], c["b"])

    repeats = [c for c in cases if c["kind"] == "repeat"]
    should_hit = [c for c in cases if c["hit"] and c["kind"] != "repeat"]
    must_miss = [c for c in cases if not c["hit"]]
    rows = []
    for th in THRESHOLDS:
        for guarded in (False, True):
            hit = lambda c: c["cos"] >= th and (c["guard_ok"] or not guarded)  # noqa: E731
            false_hits = [c for c in must_miss if hit(c)]
            by_kind = defaultdict(int)
            for c in false_hits:
                by_kind[c["kind"]] += 1
            rows.append({
                "threshold": th, "guard": guarded,
                "hit_rate_paraphrase": round(sum(map(hit, should_hit)) / len(should_hit), 3),
                "hit_rate_repeated_question": round(sum(map(hit, repeats)) / len(repeats), 3),
                "false_hits": len(false_hits), "false_hits_by_kind": dict(by_kind),
            })

    out = {
        "model": settings.embedding_model, "cases": len(cases),
        "should_hit": len(should_hit), "repeated_questions": len(repeats), "must_miss": len(must_miss),
        "guard_blocks_true_paraphrases": [
            {"a": c["a"], "b": c["b"]} for c in should_hit if not c["guard_ok"]],
        "guard_lets_through_negatives": [
            {"a": c["a"], "b": c["b"], "kind": c["kind"], "cos": round(c["cos"], 3)}
            for c in must_miss if c["guard_ok"] and c["cos"] >= 0.85],
        "sweep": rows,
    }
    path = Path(__file__).parent / "results.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"{'threshold':>9} {'guard':>5} {'paraphrase hits':>15} {'repeat hits':>11} {'false hits':>10}  by kind")
    for r in rows:
        print(f"{r['threshold']:>9} {str(r['guard']):>5} {r['hit_rate_paraphrase']:>15} "
              f"{r['hit_rate_repeated_question']:>11} {r['false_hits']:>10}  {r['false_hits_by_kind']}")
    print(f"\nguard blocks {len(out['guard_blocks_true_paraphrases'])}/{len(should_hit)} true paraphrases; "
          f"lets through {len(out['guard_lets_through_negatives'])} negatives with cos >= 0.85")


if __name__ == "__main__":
    main()
