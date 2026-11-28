"""Field validation shared by the frozen core types.

Every helper names the offending field in its error, so a bad RFQ or API payload
fails with a message that says exactly which input is wrong.
"""

from __future__ import annotations

import math
from enum import Enum
from numbers import Real


def as_float(name: str, value: object) -> float:
    """Return ``value`` as a finite Python float.

    Accepts any real number (int, float, numpy scalars) but not bool, which is an
    int subclass and almost always a mistake here. Negative zero is normalised to
    zero so that equal inputs serialise, and therefore hash, identically.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite, got {x}")
    return x + 0.0  # -0.0 + 0.0 == +0.0


def positive(name: str, value: object) -> float:
    x = as_float(name, value)
    if x <= 0.0:
        raise ValueError(f"{name} must be > 0, got {x}")
    return x


def non_negative(name: str, value: object) -> float:
    x = as_float(name, value)
    if x < 0.0:
        raise ValueError(f"{name} must be >= 0, got {x}")
    return x


def require_enum[E: Enum](name: str, value: object, enum_type: type[E]) -> E:
    if not isinstance(value, enum_type):
        raise TypeError(f"{name} must be a {enum_type.__name__}, got {value!r}")
    return value
