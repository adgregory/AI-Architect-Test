# Spike 04 — Semantic answer cache: threshold and safety guard

**Status:** decided

## Question

When can a cached answer be reused for a new question? Pick the cosine threshold, and decide
whether similarity alone is safe.

## Method

[`evaluate.py`](evaluate.py) replays spike 03's labelled data through the app's own
embedding service (bge-small via fastembed, query prefix) and the app's `QuestionGuard`
([`app/services/cache_service.py`](../../app/services/cache_service.py)):

- **Must hit:** 50 paraphrases (35 triplet positives + 15 grade-3 pairs) and 145 realistic
  *repeated questions* (the 30 retrieval questions re-asked with case changes, filler words,
  contractions, extra whitespace).
- **Must miss:** 63 pairs — the 35 hard negatives (negation, antonym, entity swap, number
  change, role swap) and the grade 0–2 pairs.

The cost asymmetry drives the design: a **false miss** costs one LLM call; a **false hit**
returns a wrong answer. Target: zero false hits, then maximum hit rate.

## Results ([`results.json`](results.json))

| Threshold | Guard | Repeated-question hits | Paraphrase hits | **False hits** |
|-----------|-------|------------------------|-----------------|----------------|
| 0.80 | no | 100% | 90% | **30** |
| 0.90 | no | 100% | 48% | **18** (all 7 role swaps, 6 number changes, …) |
| 0.96 | no | 96% | 16% | **9** (still every role swap) |
| 0.80 | yes | 94% | 30% | **0** |
| **0.90** | **yes** | **94%** | 10% | **0** |
| 0.96 | yes | 90% | 0% | **0** |

### Findings

1. **No cosine threshold is safe on its own.** Role swaps ("A reports to B" / "B reports to A")
   score 0.993–0.998 — above almost every true paraphrase — and number changes reach 0.97.
2. **The guard removes every false hit at every threshold.** It rejects a candidate when the
   two questions differ in negation, numbers (number words normalised: "twelve percent" =
   "12%"), proper nouns, the *order* of proper nouns (role swaps), or a known antonym pair.
3. **Realistic repeats still hit (94%)**; heavily reworded paraphrases mostly miss (10% at
   0.90). A passive-voice paraphrase is lexically identical to a role swap, so blocking it is
   the price of safety — and a miss only costs a normal LLM call.

## Decision

**Accepted (2026-10-06): cosine ≥ 0.90 on bge-small query embeddings AND `QuestionGuard`**, in a
separate Qdrant collection with a TTL (`created_at` payload filter + periodic purge). The
threshold and TTL are configuration. A cross-encoder judging question equivalence is the
upgrade path if a higher paraphrase hit rate is ever worth the latency.
