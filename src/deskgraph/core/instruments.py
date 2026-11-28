"""Product definitions: European, barrier and autocallable.

Instruments are frozen, validated value objects with no pricing logic, so the
same instrument can be priced by Black-Scholes, Monte Carlo or a bumped
revaluation, and used as a cache key in the dependency graph.

Validation here checks that the payoff is *well defined*. Desk business rules
that are stricter (for example "knock-in strictly below the autocall level")
belong to RFQ validation in the AI layer, so edge cases such as an autocall
level of 0 stay priceable for tests.

Conventions (docs/WHEN_IN_DOUBT.md): times are year fractions; vanilla and
barrier strikes and barrier levels are absolute prices; autocallable levels are
fractions of the initial spot.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise

from deskgraph.core._validation import non_negative, positive, require_enum

# A last observation time within this many years of expiry is treated as expiry
# itself (absorbs rounding when dates are built by summing period lengths).
_EXPIRY_SNAP_TOLERANCE = 1e-9


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


class BarrierType(Enum):
    DOWN_IN = "down_in"
    DOWN_OUT = "down_out"
    UP_IN = "up_in"
    UP_OUT = "up_out"

    @property
    def is_down(self) -> bool:
        """True if the barrier is hit by spot falling to or below it."""
        return self in (BarrierType.DOWN_IN, BarrierType.DOWN_OUT)

    @property
    def is_knock_in(self) -> bool:
        """True if hitting the barrier switches the option on (knock-in)."""
        return self in (BarrierType.DOWN_IN, BarrierType.UP_IN)


class Monitoring(Enum):
    """How often a barrier is checked."""

    DAILY = "daily"  # every trading day, time step 1/252 year
    AT_MATURITY = "at_maturity"  # only at expiry


@dataclass(frozen=True, slots=True)
class European:
    """European call or put on one unit of the underlying.

    Attributes:
        option_type: call or put.
        strike: strike price K, absolute, > 0.
        expiry: time to expiry T in years, > 0.
    """

    option_type: OptionType
    strike: float
    expiry: float

    def __post_init__(self) -> None:
        require_enum("option_type", self.option_type, OptionType)
        object.__setattr__(self, "strike", positive("strike", self.strike))
        object.__setattr__(self, "expiry", positive("expiry", self.expiry))


@dataclass(frozen=True, slots=True)
class Barrier:
    """Single-barrier European call or put, barrier monitored daily.

    Knock-in options pay the vanilla payoff only if spot touched the barrier
    during the life of the option; knock-out options pay it only if spot never
    did. Monitoring is discrete (daily), not continuous.

    Attributes:
        option_type: call or put.
        strike: strike price K, absolute, > 0.
        expiry: time to expiry T in years, > 0.
        barrier: barrier level H, absolute price, > 0.
        barrier_type: direction (down/up) and effect (in/out).
    """

    option_type: OptionType
    strike: float
    expiry: float
    barrier: float
    barrier_type: BarrierType

    def __post_init__(self) -> None:
        require_enum("option_type", self.option_type, OptionType)
        object.__setattr__(self, "strike", positive("strike", self.strike))
        object.__setattr__(self, "expiry", positive("expiry", self.expiry))
        object.__setattr__(self, "barrier", positive("barrier", self.barrier))
        require_enum("barrier_type", self.barrier_type, BarrierType)


@dataclass(frozen=True, slots=True)
class Autocallable:
    """Autocallable note, priced per unit notional (1.0 = par).

    With observation times t₁ < … < tₙ = T, spot ratio Xᵢ = S(tᵢ)/S₀ and coupon c
    per period (ARCHITECTURE decision D-10):

    - at the first tᵢ with Xᵢ ≥ autocall_level the note redeems early and pays
      1 + i·c (coupons accumulate: memory);
    - if never called, it pays 1 at T when not knocked in, or X(T) when knocked
      in, with no coupon. Knocked in means spot fell below knock_in_level on any
      trading day (``Monitoring.DAILY``) or at T only (``Monitoring.AT_MATURITY``).

    S₀ is the spot at pricing time: M1 prices notes at inception.

    Attributes:
        expiry: maturity T in years, > 0.
        observation_times: autocall dates in years, strictly increasing, > 0; the
            last one must equal ``expiry`` (within 1e-9, then set to it exactly).
        autocall_level: early redemption level as a fraction of S₀, >= 0.
        coupon: coupon per observation period as a fraction of notional, >= 0.
        knock_in_level: knock-in level as a fraction of S₀, between 0 and
            ``autocall_level`` inclusive.
        knock_in_monitoring: daily or at maturity only.
    """

    expiry: float
    observation_times: tuple[float, ...]
    autocall_level: float
    coupon: float
    knock_in_level: float
    knock_in_monitoring: Monitoring

    def __post_init__(self) -> None:
        expiry = positive("expiry", self.expiry)
        object.__setattr__(self, "expiry", expiry)
        object.__setattr__(
            self, "observation_times", _observation_times(self.observation_times, expiry)
        )
        autocall = non_negative("autocall_level", self.autocall_level)
        knock_in = non_negative("knock_in_level", self.knock_in_level)
        if knock_in > autocall:
            raise ValueError(
                f"knock_in_level ({knock_in}) must not exceed autocall_level ({autocall})"
            )
        object.__setattr__(self, "autocall_level", autocall)
        object.__setattr__(self, "knock_in_level", knock_in)
        object.__setattr__(self, "coupon", non_negative("coupon", self.coupon))
        require_enum("knock_in_monitoring", self.knock_in_monitoring, Monitoring)


Instrument = European | Barrier | Autocallable
"""Any product the pricers accept."""


def _observation_times(values: object, expiry: float) -> tuple[float, ...]:
    """Validate and copy observation times into a tuple of floats ending at expiry."""
    if not isinstance(values, Iterable):
        raise TypeError(
            f"observation_times must be a sequence of times, got {type(values).__name__}"
        )
    raw: tuple[object, ...] = tuple(values)
    if not raw:
        raise ValueError("observation_times must not be empty")
    times = [positive(f"observation_times[{i}]", t) for i, t in enumerate(raw)]
    if any(later <= earlier for earlier, later in pairwise(times)):
        raise ValueError(f"observation_times must be strictly increasing, got {times}")
    if not math.isclose(times[-1], expiry, rel_tol=0.0, abs_tol=_EXPIRY_SNAP_TOLERANCE):
        raise ValueError(f"last observation time ({times[-1]}) must equal expiry ({expiry})")
    times[-1] = expiry
    return tuple(times)
