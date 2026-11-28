"""Collects benchmark metrics and writes them, with the environment, to results.json.

Run the whole suite to refresh the file: ``uv run pytest benchmarks --benchmark-only``.
A partial run rewrites the file with only the metrics it measured.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import scipy

import deskgraph
from benchmarks._support import Results

RESULTS_PATH = Path(__file__).with_name("results.json")


@pytest.fixture(scope="session")
def results() -> Iterator[Results]:
    collected: Results = {}
    yield collected
    if not collected:
        return
    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": _environment(),
        "metrics": dict(sorted(collected.items())),
    }
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    RESULTS_PATH.write_text(text, encoding="utf-8")


def _environment() -> dict[str, object]:
    status = _git("status", "--porcelain")
    return {
        "deskgraph": deskgraph.__version__,
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": None if status == "unknown" else bool(status),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()
