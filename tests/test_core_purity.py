"""Enforce CLAUDE.md's rule that src/deskgraph/core is a pure library.

The checker itself is tested first on small snippets, so a passing scan of the
real package means something.
"""

from pathlib import Path

import pytest

import deskgraph.core
from tests.purity import check_source

CORE_DIR = Path(deskgraph.core.__file__).parent


@pytest.mark.parametrize(
    ("snippet", "expected"),
    [
        ("import os", "import of 'os'"),
        ("import urllib.request", "import of 'urllib.request'"),
        ("from pathlib import Path", "import of 'pathlib'"),
        ("import random", "import of 'random'"),
        ("import logging", "import of 'logging'"),
        ("open('prices.csv')", "call to builtin 'open'"),
        ("print(1.0)", "call to builtin 'print'"),
        ("import numpy as np\nnp.random.seed(0)", "legacy numpy RNG 'np.random.seed'"),
        ("import numpy\nnumpy.random.normal()", "legacy numpy RNG 'numpy.random.normal'"),
        ("import numpy.random as npr\nnpr.rand()", "legacy numpy RNG 'npr.rand'"),
        ("from numpy import random\nrandom.normal()", "legacy numpy RNG 'random.normal'"),
        ("from numpy.random import normal", "legacy numpy RNG 'numpy.random.normal'"),
        ("from datetime import datetime\ndatetime.now()", "clock read '.now()'"),
        ("CACHE = {}", "mutable module-level state 'CACHE'"),
        ("SEEN: list[int] = []", "mutable module-level state 'SEEN'"),
        ("def f() -> None:\n    global X", "'global' statement"),
    ],
)
def test_checker_flags_impure_code(snippet: str, expected: str) -> None:
    messages = [v.message for v in check_source(snippet, "snippet.py")]
    assert any(expected in m for m in messages), messages


@pytest.mark.parametrize(
    "snippet",
    [
        "import math",
        "from dataclasses import dataclass",
        "import numpy as np\nfrom numpy.random import Generator",
        "import numpy as np\ndef f(seed: int) -> None:\n    rng = np.random.default_rng(seed)",
        "import numpy as np\ndef f(g: np.random.Generator) -> None: ...",
        "LEVELS: tuple[float, ...] = (0.6, 1.0)",
        "NAMES = frozenset({'call', 'put'})",
        "def f() -> list[int]:\n    out = []\n    return out",
    ],
)
def test_checker_allows_pure_code(snippet: str) -> None:
    assert check_source(snippet, "snippet.py") == []


def test_core_package_is_pure() -> None:
    files = sorted(CORE_DIR.rglob("*.py"))
    assert files, f"no modules found under {CORE_DIR}"
    violations = [v for f in files for v in check_source(f.read_text("utf-8"), str(f))]
    assert violations == [], "\n".join(str(v) for v in violations)
