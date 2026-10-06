"""Download / export every model the configured engines need into MODELS_DIR.

Builds the application container from Settings and warms it up — the same code path
the FastAPI lifespan runs — so a Docker build (or `task models:prepare`) leaves the
image or machine ready to serve offline.

    uv run python scripts/prepare_models.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.core.container import Container  # noqa: E402
from app.core.logging import configure_logging, get_logger  # noqa: E402


def main() -> None:
    settings = get_settings()
    configure_logging(settings)
    log = get_logger("prepare_models")
    started = time.perf_counter()
    Container(settings).warm_up()
    log.info("models.ready", seconds=round(time.perf_counter() - started, 1), models_dir=str(settings.models_dir),
             ocr=settings.ocr_engine, ner=settings.ner_engine, embeddings=settings.embedding_engine)


if __name__ == "__main__":
    main()
