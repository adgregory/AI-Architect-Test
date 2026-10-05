"""Build the OCR spike dataset: clean pages, degraded variants, and exact ground truth.

Ground truth comes from re-running the page generators in
scripts/generate_test_pdfs.py: we capture the lines each page draws and compute
every word's box from the same font metrics, so text and boxes are exact.

Usage (from repo root):
    uv run python spikes/01-ocr/dataset/build.py
"""

from __future__ import annotations

import io
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import generate_test_pdfs as gen  # noqa: E402

# Expected names per document, as listed in generate_test_pdfs.main().
EXPECTED_NAMES = {
    "company_memo": [
        "Margaret Thompson", "Robert Chen", "Sarah Williams", "James Anderson",
        "Maria Garcia", "David Nakamura", "Patricia Okonkwo",
    ],
    "meeting_minutes": [
        "Richard Hernandez", "Elizabeth Park", "Thomas Muller", "Aisha Patel",
        "Kevin O'Brien", "Jennifer Liu", "Carlos Mendoza", "Yuki Tanaka",
        "Alexander Popov", "Catherine Dubois",
    ],
    "research_report": [
        "Olivia Chambers", "Benjamin Foster", "Priya Sharma", "Lucas Zimmermann",
        "Fatima Al-Rashidi", "Christopher Wong", "Anna Kowalski",
        "Michael O'Sullivan", "Elena Volkov", "Raj Krishnamurthy", "Hans Weber",
        "James Chen", "Margaret Thompson",
    ],
}

GENERATORS = {
    "company_memo": gen.generate_company_memo,
    "meeting_minutes": gen.generate_meeting_minutes,
    "research_report": gen.generate_research_report,
}

PUNCT = ".,;:!?()[]'\""


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #
def capture_pages(generator) -> list[tuple[Image.Image, list[str]]]:
    """Run a page generator and capture the lines drawn on each page."""
    captured: list[list[str]] = []
    original = gen.create_page_image

    def spy(lines, title=None):
        assert title is None, "titles are not used by the generator"
        captured.append(list(lines))
        return original(lines, title)

    gen.create_page_image = spy
    try:
        images = generator()
    finally:
        gen.create_page_image = original
    return list(zip(images, captured))


def word_boxes(lines: list[str]) -> list[dict]:
    """Exact box of every word, mirroring create_page_image's drawing positions."""
    font = gen.get_font(20)
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    words = []
    y = gen.MARGIN
    for line_no, line in enumerate(lines):
        pos = 0
        for token in line.split():
            start = line.index(token, pos)
            pos = start + len(token)
            x = gen.MARGIN + font.getlength(line[:start])
            core = strip_punct(token) or token
            core_x = x + font.getlength(token[: token.index(core)])
            words.append({
                "text": token,
                "line": line_no,
                "box": list(draw.textbbox((x, y), token, font=font)),
                # Box of the token without surrounding punctuation, used for name boxes.
                "core_box": list(draw.textbbox((core_x, y), core, font=font)),
            })
        y += gen.LINE_HEIGHT
    return words


def strip_punct(token: str) -> str:
    """Drop surrounding punctuation; internal apostrophes/hyphens (O'Brien, Al-Rashidi) stay."""
    return token.strip(PUNCT)


def locate_names(words: list[dict], names: list[str]) -> list[dict]:
    """Find every occurrence of each expected name as a run of consecutive words."""
    found = []
    tokens = [strip_punct(w["text"]) for w in words]
    for name in names:
        parts = name.split()
        for i in range(len(tokens) - len(parts) + 1):
            if tokens[i : i + len(parts)] == parts:
                boxes = [words[i + k]["core_box"] for k in range(len(parts))]
                found.append({"name": name, "box": union(boxes), "word_index": i})
    return found


def union(boxes: list[list[float]]) -> list[float]:
    return [
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    ]


# --------------------------------------------------------------------------- #
# Degradations — each returns (image, affine) where affine maps clean pixel
# coordinates to degraded pixel coordinates: (x, y) -> (a*x + b*y + c, d*x + e*y + f)
# --------------------------------------------------------------------------- #
IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def add_noise(img: Image.Image, sigma: float, salt_pepper: float, seed: int):
    rng = np.random.default_rng(seed)
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    arr += rng.normal(0, sigma, arr.shape)
    mask = rng.random(arr.shape)
    arr[mask < salt_pepper / 2] = 0
    arr[mask > 1 - salt_pepper / 2] = 255
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGB"), IDENTITY


def rotate(img: Image.Image, degrees: float):
    w, h = img.size
    out = img.rotate(degrees, resample=Image.BICUBIC, fillcolor="white")
    # PIL rotates counter-clockwise around the centre (y axis points down).
    t = math.radians(degrees)
    cos, sin = math.cos(t), math.sin(t)
    cx, cy = w / 2, h / 2
    return out, (cos, sin, cx - cos * cx - sin * cy, -sin, cos, cy + sin * cx - cos * cy)


def blur(img: Image.Image, radius: float):
    return img.filter(ImageFilter.GaussianBlur(radius)), IDENTITY


def jpeg(img: Image.Image, quality: int):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), IDENTITY


def rescale(img: Image.Image, dpi: int):
    s = dpi / gen.DPI
    w, h = img.size
    return img.resize((round(w * s), round(h * s)), Image.LANCZOS), (s, 0.0, 0.0, 0.0, s, 0.0)


def compose(img: Image.Image, steps):
    affine = IDENTITY
    for fn, kwargs in steps:
        img, a = fn(img, **kwargs)
        affine = chain(affine, a)
    return img, affine


def chain(first, second):
    """Affine for applying `first` then `second`."""
    a1, b1, c1, d1, e1, f1 = first
    a2, b2, c2, d2, e2, f2 = second
    return (
        a2 * a1 + b2 * d1, a2 * b1 + b2 * e1, a2 * c1 + b2 * f1 + c2,
        d2 * a1 + e2 * d1, d2 * b1 + e2 * e1, d2 * c1 + e2 * f1 + f2,
    )


def transform_box(box, affine):
    a, b, c, d, e, f = affine
    x0, y0, x1, y1 = box
    pts = [(a * x + b * y + c, d * x + e * y + f) for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1))]
    xs, ys = zip(*pts)
    return [round(min(xs), 2), round(min(ys), 2), round(max(xs), 2), round(max(ys), 2)]


@dataclass(frozen=True)
class Variant:
    profile: str
    label: str
    steps: tuple
    dpi: int = gen.DPI


def variants() -> list[Variant]:
    out = []
    for sev, (sigma, sp) in enumerate([(10, 0.002), (25, 0.01), (45, 0.03)], 1):
        for seed in (1, 2):
            out.append(Variant("noise", f"s{sev}_seed{seed}", ((add_noise, {"sigma": sigma, "salt_pepper": sp, "seed": seed}),)))
    for deg in (-3, -1.5, -0.5, 0.5, 1.5, 3):
        out.append(Variant("skew", f"{deg:+g}deg", ((rotate, {"degrees": deg}),)))
    for r in (0.8, 1.4, 2.0):
        out.append(Variant("blur", f"r{r:g}", ((blur, {"radius": r}),)))
    for q in (30, 15, 5):
        out.append(Variant("jpeg", f"q{q}", ((jpeg, {"quality": q}),)))
    for dpi in (100, 72):
        out.append(Variant("low_dpi", f"{dpi}dpi", ((rescale, {"dpi": dpi}),), dpi=dpi))
    scan_levels = [
        (0.7, 8, 0.002, 0.6, 50),
        (1.5, 18, 0.006, 1.0, 30),
        (2.5, 30, 0.015, 1.4, 15),
    ]
    for sev, (deg, sigma, sp, r, q) in enumerate(scan_levels, 1):
        for seed in (1, 2, 3):
            sign = 1 if seed % 2 else -1
            out.append(Variant("scan", f"s{sev}_seed{seed}", (
                (rotate, {"degrees": sign * deg}),
                (blur, {"radius": r}),
                (add_noise, {"sigma": sigma, "salt_pepper": sp, "seed": seed}),
                (jpeg, {"quality": q}),
            )))
    return out


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def jpeg_like_pdf(img: Image.Image) -> Image.Image:
    """Apply the same JPEG encoding Pillow uses when writing the sample PDFs."""
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def write_page(image: Image.Image, page_id: str, subdir: Path, gt: dict, affine, dpi: int) -> dict:
    subdir.mkdir(parents=True, exist_ok=True)
    path = subdir / f"{page_id}.png"
    image.save(path)
    return {
        **gt,
        "page_id": page_id,
        "image": str(path.relative_to(DATA_DIR)),
        "width": image.width,
        "height": image.height,
        "dpi": dpi,
        "words": [
            {**w, "box": transform_box(w["box"], affine), "core_box": transform_box(w["core_box"], affine)}
            for w in gt["words"]
        ],
        "names": [{**n, "box": transform_box(n["box"], affine)} for n in gt["names"]],
    }


def main() -> None:
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    (DATA_DIR / "ground_truth").mkdir(parents=True)

    manifest = []
    for doc, generator in GENERATORS.items():
        for page_index, (image, lines) in enumerate(capture_pages(generator)):
            clean = jpeg_like_pdf(image)
            words = word_boxes(lines)
            base_gt = {
                "doc": doc,
                "page_index": page_index,
                "text": "\n".join(line for line in lines if line.strip()),
                "words": words,
                "names": locate_names(words, EXPECTED_NAMES[doc]),
            }
            base_id = f"{doc}_p{page_index}"
            entries = [write_page(clean, base_id, DATA_DIR / "clean", {**base_gt, "profile": "clean", "variant": "clean"}, IDENTITY, gen.DPI)]
            for v in variants():
                img, affine = compose(clean, v.steps)
                entries.append(write_page(
                    img, f"{base_id}__{v.profile}_{v.label}", DATA_DIR / "degraded" / v.profile,
                    {**base_gt, "profile": v.profile, "variant": v.label}, affine, v.dpi,
                ))
            for e in entries:
                (DATA_DIR / "ground_truth" / f"{e['page_id']}.json").write_text(json.dumps(e, indent=1))
                manifest.append({k: e[k] for k in ("page_id", "doc", "page_index", "profile", "variant", "image")})

    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))
    by_profile: dict[str, int] = {}
    for m in manifest:
        by_profile[m["profile"]] = by_profile.get(m["profile"], 0) + 1
    names_found = {doc: sorted({n["name"] for p in manifest if p["doc"] == doc
                                for n in json.loads((DATA_DIR / "ground_truth" / f"{p['page_id']}.json").read_text())["names"]})
                   for doc in GENERATORS}
    print(f"{len(manifest)} pages written to {DATA_DIR}")
    print("pages per profile:", by_profile)
    for doc, expected in EXPECTED_NAMES.items():
        missing = set(expected) - set(names_found[doc])
        print(f"{doc}: located {len(names_found[doc])}/{len(expected)} expected names", f"(missing: {sorted(missing)})" if missing else "")


if __name__ == "__main__":
    main()
