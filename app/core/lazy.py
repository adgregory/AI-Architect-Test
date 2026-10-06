"""A lazily-initialised proxy so modules can declare a heavy dependency without loading it at import."""

from __future__ import annotations

import threading
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Lazy(Generic[T]):
    """Builds its target on first use (thread-safe) and forwards calls/attributes to it."""

    def __init__(self, factory: Callable[[], T]):
        self._factory = factory
        self._value: T | None = None
        self._lock = threading.Lock()

    def get(self) -> T:
        if self._value is None:
            with self._lock:
                if self._value is None:
                    self._value = self._factory()
        return self._value

    def __call__(self, *args, **kwargs):
        return self.get()(*args, **kwargs)  # type: ignore[operator]

    def __getattr__(self, name: str):
        return getattr(self.get(), name)
