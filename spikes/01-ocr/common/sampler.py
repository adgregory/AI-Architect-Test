"""Background sampler for CPU, memory and (optionally) GPU memory of this process tree."""

from __future__ import annotations

import threading
import time
from typing import Callable

import psutil


class ResourceSampler:
    def __init__(self, interval_s: float = 0.05, gpu_memory: Callable[[], int | None] | None = None):
        self.interval_s = interval_s
        self.gpu_memory = gpu_memory
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._root = psutil.Process()

    def _procs(self) -> list[psutil.Process]:
        try:
            return [self._root, *self._root.children(recursive=True)]
        except psutil.Error:
            return [self._root]

    def _run(self) -> None:
        known: dict[int, psutil.Process] = {}
        psutil.cpu_percent(None)
        while not self._stop.is_set():
            cpu, rss = 0.0, 0
            for p in self._procs():
                proc = known.setdefault(p.pid, p)
                try:
                    cpu += proc.cpu_percent(None)
                    rss += proc.memory_info().rss
                except psutil.Error:
                    continue
            gpu = self.gpu_memory() if self.gpu_memory else None
            self.samples.append({"t": time.perf_counter(), "cpu_pct": cpu, "rss": rss,
                                 "sys_cpu_pct": psutil.cpu_percent(None), "gpu_mem": gpu})
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "ResourceSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()

    def summary(self) -> dict:
        s = self.samples[1:] or self.samples  # first cpu_percent reading is meaningless
        if not s:
            return {}
        cpu = [x["cpu_pct"] for x in s]
        gpu = [x["gpu_mem"] for x in s if x["gpu_mem"] is not None]
        return {
            "samples": len(s),
            "logical_cores": psutil.cpu_count(),
            "cpu_pct_mean": round(sum(cpu) / len(cpu), 1),  # 100 = one core fully busy
            "cpu_pct_peak": round(max(cpu), 1),
            "system_cpu_pct_mean": round(sum(x["sys_cpu_pct"] for x in s) / len(s), 1),
            "rss_peak_mb": round(max(x["rss"] for x in s) / 2**20, 1),
            "gpu_mem_peak_mb": round(max(gpu) / 2**20, 1) if gpu else None,
        }
