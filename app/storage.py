"""Object storage for job artefacts (uploaded PDFs, per-page OCR results, extraction results).

Workflow activities exchange storage keys, never document contents: Temporal records every
activity input/output in workflow history (payloads are capped at ~2 MB).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ObjectStorage(Protocol):
    def put_bytes(self, key: str, data: bytes) -> None: ...

    def get_bytes(self, key: str) -> bytes: ...

    def exists(self, key: str) -> bool: ...

    def delete_prefix(self, prefix: str) -> None: ...

    def local_path(self, key: str) -> AbstractContextManager[Path]:
        """A filesystem path to the object for libraries that need one (S3: temp download)."""
        ...


class LocalFileStorage:
    """Objects as files under a root directory (a shared Docker volume across api/workers)."""

    def __init__(self, root: Path):
        self._root = Path(root).resolve()

    def _path(self, key: str) -> Path:
        path = (self._root / key).resolve()
        if self._root not in path.parents:
            raise ValueError(f"key escapes the storage root: {key!r}")
        return path

    def put_bytes(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)  # atomic: readers never see a partial object

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def delete_prefix(self, prefix: str) -> None:
        shutil.rmtree(self._path(prefix), ignore_errors=True)

    @contextmanager
    def local_path(self, key: str) -> Iterator[Path]:
        path = self._path(key)
        if not path.exists():
            raise FileNotFoundError(key)
        yield path


def document_id_for(content: bytes) -> str:
    """Content-addressed document ID: the same PDF always maps to the same ID, so indexing it
    again overwrites its chunks (point IDs derive from it) instead of duplicating them."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"sha256:{hashlib.sha256(content).hexdigest()}"))


class JobArtifacts:
    """Key layout for one job, plus JSON helpers."""

    def __init__(self, storage: ObjectStorage, job_id: str):
        self.storage = storage
        self.job_id = job_id
        self.prefix = f"jobs/{job_id}"

    @property
    def input_pdf(self) -> str:
        return f"{self.prefix}/input.pdf"

    def page(self, page_number: int) -> str:
        return f"{self.prefix}/pages/{page_number:04d}.json"

    @property
    def result(self) -> str:
        return f"{self.prefix}/result.json"

    @property
    def chunks(self) -> str:
        return f"{self.prefix}/chunks.json"

    def put_json(self, key: str, value: Any) -> str:
        self.storage.put_bytes(key, json.dumps(value).encode())
        return key

    def get_json(self, key: str) -> Any:
        return json.loads(self.storage.get_bytes(key))
