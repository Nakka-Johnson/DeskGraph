"""Static checker for the pure-core rule (CLAUDE.md: no I/O, no network, no globals).

It walks a module's AST and flags:

- imports of modules that do I/O, networking, logging, concurrency, the clock or
  stdlib randomness;
- calls to I/O builtins such as ``open`` and ``print``;
- the legacy global numpy RNG (``np.random.seed``, ``np.random.normal``, ...),
  since every random number must come from a ``Generator`` passed in by the caller;
- clock reads such as ``datetime.now()``;
- mutable module-level state (list, dict or set values) and ``global`` statements.

This is a guard against accidents, not a sandbox: it cannot see through
``getattr`` tricks, and it doesn't try to.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

FORBIDDEN_MODULES = frozenset(
    {
        # file system and process
        "os", "sys", "io", "pathlib", "shutil", "glob", "tempfile", "subprocess", "pickle",
        # network
        "socket", "ssl", "select", "http", "urllib", "ftplib", "smtplib",
        "requests", "httpx", "aiohttp",
        # databases
        "sqlite3", "psycopg", "psycopg2", "sqlalchemy",
        # side channels, clock, concurrency, stdlib randomness
        "logging", "time", "threading", "multiprocessing", "asyncio", "random", "secrets",
    }
)  # fmt: skip
FORBIDDEN_BUILTINS = frozenset(
    {"open", "print", "input", "exec", "eval", "breakpoint", "__import__"}
)
CLOCK_METHODS = frozenset({"now", "today", "utcnow"})
# numpy.random names that build or describe a seeded Generator; everything else
# in numpy.random is the legacy global-state API.
ALLOWED_NUMPY_RANDOM = frozenset(
    {"Generator", "default_rng", "SeedSequence", "BitGenerator",
     "PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937"}
)  # fmt: skip
MUTABLE_FACTORIES = frozenset({"list", "dict", "set", "bytearray", "defaultdict", "deque"})


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.message}"


def check_source(source: str, path: str) -> list[Violation]:
    """Return every purity violation in ``source`` (empty list if clean)."""
    tree = ast.parse(source, filename=path)
    numpy_names, numpy_random_names = _numpy_aliases(tree)
    out: list[Violation] = []

    def flag(node: ast.AST, message: str) -> None:
        out.append(Violation(path, getattr(node, "lineno", 0), message))

    for node in ast.walk(tree):
        match node:
            case ast.Import(names=names):
                for alias in names:
                    if alias.name.split(".")[0] in FORBIDDEN_MODULES:
                        flag(node, f"import of '{alias.name}'")
            case ast.ImportFrom(module=str(module), names=names):
                if module.split(".")[0] in FORBIDDEN_MODULES:
                    flag(node, f"import of '{module}'")
                if module == "numpy.random":
                    for alias in names:
                        if alias.name not in ALLOWED_NUMPY_RANDOM:
                            flag(node, f"legacy numpy RNG 'numpy.random.{alias.name}'")
            case ast.Call(func=ast.Name(id=name)) if name in FORBIDDEN_BUILTINS:
                flag(node, f"call to builtin '{name}'")
            case ast.Call(func=ast.Attribute(attr=attr)) if attr in CLOCK_METHODS:
                flag(node, f"clock read '.{attr}()'")
            case ast.Attribute(value=value, attr=attr) if attr not in ALLOWED_NUMPY_RANDOM:
                if _is_numpy_random(value, numpy_names, numpy_random_names):
                    flag(node, f"legacy numpy RNG '{ast.unparse(node)}'")
            case ast.Global():
                flag(node, "'global' statement")

    for stmt in tree.body:  # module level only
        for name in _mutable_assignment_targets(stmt):
            flag(stmt, f"mutable module-level state '{name}'")
    return out


def _numpy_aliases(tree: ast.Module) -> tuple[set[str], set[str]]:
    """Names bound to ``numpy`` and to ``numpy.random`` anywhere in the module."""
    numpy_names: set[str] = set()
    numpy_random_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "numpy.random" and alias.asname:
                    numpy_random_names.add(alias.asname)
                elif alias.name.split(".")[0] == "numpy":
                    # "import numpy.random" binds the name "numpy"
                    numpy_names.add(alias.asname or "numpy")
        elif isinstance(node, ast.ImportFrom) and node.module == "numpy":
            for alias in node.names:
                if alias.name == "random":
                    numpy_random_names.add(alias.asname or "random")
    return numpy_names, numpy_random_names


def _is_numpy_random(expr: ast.expr, numpy_names: set[str], numpy_random_names: set[str]) -> bool:
    """True if ``expr`` is ``<numpy alias>.random`` or a bare ``numpy.random`` alias."""
    if isinstance(expr, ast.Name):
        return expr.id in numpy_random_names
    return (
        isinstance(expr, ast.Attribute)
        and expr.attr == "random"
        and isinstance(expr.value, ast.Name)
        and expr.value.id in numpy_names
    )


def _mutable_assignment_targets(stmt: ast.stmt) -> list[str]:
    match stmt:
        case ast.Assign(targets=targets, value=value):
            names = [t.id for t in targets if isinstance(t, ast.Name)]
        case ast.AnnAssign(target=ast.Name(id=name), value=value) if value is not None:
            names = [name]
        case _:
            return []
    if not _is_mutable_value(value):
        return []
    return [n for n in names if not (n.startswith("__") and n.endswith("__"))]  # allow __all__


def _is_mutable_value(value: ast.expr) -> bool:
    if isinstance(value, ast.List | ast.Dict | ast.Set | ast.ListComp | ast.DictComp | ast.SetComp):
        return True
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id in MUTABLE_FACTORIES
    )
