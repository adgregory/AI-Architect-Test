"""Name-level scoring. A name counts once per page however often it is mentioned."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

TITLES = {"dr", "prof", "mr", "mrs", "ms", "sir"}
# Organisations / places / events in the sample documents that must not be tagged as people.
TRACKED_NON_PERSONS = [
    "acme", "novatech", "eth zurich", "stanford", "nih", "nsf", "darpa", "aws", "london", "atlas",
    "forbes", "neurips", "icml", "national science foundation", "apac", "southeast asia",
    "human resources", "headquarters", "board",
]
FUZZY_THRESHOLD = 0.9  # same threshold the /extract fuzzy matching uses


def normalize_name(raw: str) -> str:
    """What the app would store: no titles, possessives, edge punctuation or line breaks."""
    s = re.sub(r"\s+", " ", raw).strip()
    s = re.sub(r"['’]s$", "", s)
    tokens = [t.strip(".,;:()\"'") for t in s.split()]
    tokens = [t for t in tokens if t and t.lower().rstrip(".") not in TITLES]
    return " ".join(tokens)


def needs_cleanup(raw: str) -> bool:
    return normalize_name(raw) != re.sub(r"\s+", " ", raw).strip()


def split_first_last(name: str) -> tuple[str, str]:
    parts = name.split()
    return (parts[0], parts[-1]) if len(parts) >= 2 else (name, "")


def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def score_page(gt_names: set[str], predicted_raw: list[str]) -> dict:
    predicted = {normalize_name(p) for p in predicted_raw} - {""}
    gt_l = {g.lower() for g in gt_names}

    exact_tp = {p for p in predicted if p.lower() in gt_l}
    fuzzy_tp = {p for p in predicted if any(similar(p, g) >= FUZZY_THRESHOLD for g in gt_names)}
    found_exact = {g for g in gt_names if g.lower() in {p.lower() for p in predicted}}
    found_fuzzy = {g for g in gt_names if any(similar(p, g) >= FUZZY_THRESHOLD for p in predicted)}
    partial = {p for p in predicted - fuzzy_tp
               if any(set(p.lower().split()) < set(g.lower().split()) for g in gt_names)}
    false_pos = predicted - fuzzy_tp - partial
    tracked_fp = {p for p in false_pos if any(t in p.lower() for t in TRACKED_NON_PERSONS)}
    single_token = {p for p in predicted if len(p.split()) < 2}

    return {
        "gt": len(gt_names), "predicted": len(predicted),
        "exact_tp": len(exact_tp), "fuzzy_tp": len(fuzzy_tp),
        "found_exact": len(found_exact), "found_fuzzy": len(found_fuzzy),
        "partial": sorted(partial), "false_positives": sorted(false_pos), "tracked_fp": sorted(tracked_fp),
        "single_token": len(single_token),
        "needs_cleanup": sum(needs_cleanup(p) for p in predicted_raw), "raw_spans": len(predicted_raw),
        "missed": sorted(gt_names - found_fuzzy),
    }


def aggregate(pages: list[dict]) -> dict:
    s = lambda k: sum(p[k] for p in pages)  # noqa: E731

    def prf(tp_pred: int, found: int) -> dict:
        precision = tp_pred / s("predicted") if s("predicted") else 0.0
        recall = found / s("gt") if s("gt") else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}

    return {
        "pages": len(pages),
        "exact": prf(s("exact_tp"), s("found_exact")),
        "fuzzy": prf(s("fuzzy_tp"), s("found_fuzzy")),
        "partial_names": sum(len(p["partial"]) for p in pages),
        "false_positives": sum(len(p["false_positives"]) for p in pages),
        "tracked_org_loc_as_person": sum(len(p["tracked_fp"]) for p in pages),
        "single_token_predictions": s("single_token"),
        "raw_spans_needing_cleanup_pct": round(s("needs_cleanup") / s("raw_spans"), 4) if s("raw_spans") else None,
    }
