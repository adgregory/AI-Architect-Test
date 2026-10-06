"""Download / export the models the configured engines need into MODELS_DIR.

Builds the application container from Settings and warms it up — the same code path
the FastAPI lifespan runs — so a Docker build (or `task models:prepare`) leaves the
image or machine ready to serve offline.

    uv run python scripts/prepare_models.py                         # everything
    uv run python scripts/prepare_models.py --components embeddings # e.g. the agent image
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.core.container import Container  # noqa: E402
from app.core.logging import configure_logging, get_logger  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--components", default="ocr,ner,embeddings",
                    help="comma-separated subset of: ocr, ner, embeddings")
    components = set(ap.parse_args().components.split(","))
    settings = get_settings()
    configure_logging(settings)
    log = get_logger("prepare_models")
    started = time.perf_counter()
    container = Container(settings)
    if components >= {"ocr", "ner", "embeddings"}:
        container.warm_up()
    else:
        if "ocr" in components:
            container.ocr
        if "ner" in components:
            container.ner.extract_names("Warm-up text mentioning Jane Doe.")
        if "embeddings" in components:
            container.embeddings.embed_query("warm-up")
    log.info("models.ready", seconds=round(time.perf_counter() - started, 1), models_dir=str(settings.models_dir),
             components=sorted(components))


if __name__ == "__main__":
    main()
