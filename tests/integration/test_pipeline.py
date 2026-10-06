"""End-to-end: the configured (chosen) pipeline on the sample PDFs, through the real API and lifespan."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

pytestmark = pytest.mark.integration

SAMPLES = Path(__file__).resolve().parents[2] / "sample_pdfs"

# Expected people per document, as listed in scripts/generate_test_pdfs.py.
EXPECTED = {
    "company_memo.pdf": {"Margaret Thompson", "Robert Chen", "Sarah Williams", "James Anderson",
                         "Maria Garcia", "David Nakamura", "Patricia Okonkwo"},
    "meeting_minutes.pdf": {"Richard Hernandez", "Elizabeth Park", "Thomas Muller", "Aisha Patel",
                            "Kevin O'Brien", "Jennifer Liu", "Carlos Mendoza", "Yuki Tanaka",
                            "Alexander Popov", "Catherine Dubois"},
    "research_report.pdf": {"Olivia Chambers", "Benjamin Foster", "Priya Sharma", "Lucas Zimmermann",
                            "Fatima Al-Rashidi", "Christopher Wong", "Anna Kowalski", "Michael O'Sullivan",
                            "Elena Volkov", "Raj Krishnamurthy", "Hans Weber", "James Chen",
                            "Margaret Thompson"},
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs the lifespan: models load once for the module
        yield c


@pytest.mark.parametrize("pdf", sorted(EXPECTED))
def test_extracts_expected_people_with_boxes(client, pdf):
    expected = EXPECTED[pdf]
    query = [{"first_name": n.split()[0], "last_name": n.split()[-1]} for n in sorted(expected)]
    r = client.post("/api/extract",
                    files={"pdf_file": (pdf, (SAMPLES / pdf).read_bytes(), "application/pdf")},
                    data={"names": json.dumps(query)})
    assert r.status_code == 200
    body = r.json()

    found = {e["name"] for e in body["extracted_names"]}
    recall = len(found & expected) / len(expected)
    assert recall >= 0.9, f"recall {recall:.2f}; missing {sorted(expected - found)}"
    assert not found - expected, f"unexpected names: {sorted(found - expected)}"

    for e in body["extracted_names"]:
        box = e["bounding_box"]
        assert box["width"] > 0 and box["height"] > 0 and box["page_number"] >= 0

    matched = {m["matched_name"] for m in body["fuzzy_matches"]}
    assert len(matched) / len(expected) >= 0.9
    assert all(m["score"] >= 0.9 for m in body["fuzzy_matches"])


def test_typos_match_and_strangers_do_not(client):
    pdf = "company_memo.pdf"
    query = [{"first_name": "Margret", "last_name": "Thompson"}, {"first_name": "Zara", "last_name": "Xu"}]
    r = client.post("/api/extract",
                    files={"pdf_file": (pdf, (SAMPLES / pdf).read_bytes(), "application/pdf")},
                    data={"names": json.dumps(query)})
    assert [(m["matched_name"], m["extracted_name"]) for m in r.json()["fuzzy_matches"]] == [
        ("Margret Thompson", "Margaret Thompson"),
    ]
