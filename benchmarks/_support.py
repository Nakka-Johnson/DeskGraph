"""Typed timing helper over pytest-benchmark (whose fixture methods are unannotated)."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable

from pytest_benchmark.fixture import BenchmarkFixture

Results = dict[str, dict[str, object]]
"""Metric name -> fields, collected by the ``results`` fixture into results.json."""


def timed[T](benchmark: BenchmarkFixture, fn: Callable[[], T], rounds: int = 5) -> tuple[T, float]:
    """Run ``fn`` under pytest-benchmark; return its result and median seconds per call.

    One warm-up call is made first and left out of the median. With benchmarks
    disabled (``--benchmark-disable``) ``fn`` runs once and that call is timed.
    """
    times: list[float] = []

    def run() -> T:
        start = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - start)
        return out

    out: T = benchmark.pedantic(run, rounds=rounds, warmup_rounds=1)  # type: ignore[no-untyped-call]
    return out, statistics.median(times[1:] or times)
