"""Chunk the sample documents for the retrieval benchmark and resolve each question's gold chunk.

Chunks are the documents' natural blocks (separated by blank lines), with a
standalone heading merged into the block that follows it. Each question's gold
chunk is the one that contains its evidence text.

    uv run --project spikes/03-embeddings python spikes/03-embeddings/chunks.py   # prints a check
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import generate_test_pdfs as gen  # noqa: E402

GENERATORS = {
    "company_memo": gen.generate_company_memo,
    "meeting_minutes": gen.generate_meeting_minutes,
    "research_report": gen.generate_research_report,
}


def document_lines(generator) -> list[str]:
    captured: list[str] = []
    original = gen.create_page_image
    gen.create_page_image = lambda lines, title=None: captured.extend([*lines, ""])
    try:
        generator()
    finally:
        gen.create_page_image = original
    return captured


def chunk_document(lines: list[str]) -> list[str]:
    blocks, current = [], []
    for line in lines:
        if line.strip():
            current.append(line.strip())
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    chunks, pending = [], []
    for block in blocks:
        if len(block) == 1 and len(block[0]) < 45:  # standalone heading → prefix of next block
            pending.extend(block)
            continue
        chunks.append("\n".join(pending + block))
        pending = []
    if pending:
        chunks.append("\n".join(pending))
    return chunks


def load_corpus() -> list[dict]:
    corpus = []
    for doc, generator in GENERATORS.items():
        for i, text in enumerate(chunk_document(document_lines(generator))):
            corpus.append({"id": f"{doc}#{i}", "doc": doc, "text": text})
    return corpus


def load_questions(corpus: list[dict]) -> list[dict]:
    questions = []
    for line in (HERE / "data" / "questions.jsonl").read_text().splitlines():
        q = json.loads(line)
        gold = [c["id"] for c in corpus if q["evidence"] in c["text"]]
        if len(gold) != 1:
            raise ValueError(f"{q['id']}: evidence must be in exactly one chunk, found {gold}")
        questions.append({**q, "gold": gold[0]})
    return questions


if __name__ == "__main__":
    corpus = load_corpus()
    questions = load_questions(corpus)
    lengths = sorted(len(c["text"].split()) for c in corpus)
    print(f"{len(corpus)} chunks, words per chunk min/median/max: {lengths[0]}/{lengths[len(lengths) // 2]}/{lengths[-1]}")
    print(f"{len(questions)} questions, each resolved to exactly one gold chunk")
    for c in corpus:
        print(f"--- {c['id']}\n{c['text']}")
