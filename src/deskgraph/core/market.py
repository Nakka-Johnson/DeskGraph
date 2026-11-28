"""Market inputs for pricing.

M1 uses flat parameters: one spot, one continuously compounded rate, one
dividend yield and one volatility (ARCHITECTURE decision D-11). The full
``MarketSnapshot`` with a rate curve and an SVI surface replaces this in M3; the
``discount_factor`` and ``forward`` methods are the interface pricers rely on, so
they keep working when the curve arrives.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from deskgraph.core._validation import as_float, non_negative, positive

_SNAPSHOT_KIND = "MarketInputs"
_SNAPSHOT_VERSION = 1  # bump if the canonical form below ever changes


@dataclass(frozen=True, slots=True)
class MarketInputs:
    """Flat market inputs for one underlying at one moment.

    Attributes:
        spot: underlying price, > 0, in currency units.
        rate: risk-free rate r, continuously compounded, decimal per year (0.05 = 5%).
        div_yield: dividend yield q, continuously compounded, decimal per year.
        vol: Black-Scholes volatility σ, decimal per year (0.20 = 20%), > 0.

    Negative rates and dividend yields are allowed; all values must be finite.
    """

    spot: float
    rate: float
    div_yield: float
    vol: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "spot", positive("spot", self.spot))
        object.__setattr__(self, "rate", as_float("rate", self.rate))
        object.__setattr__(self, "div_yield", as_float("div_yield", self.div_yield))
        object.__setattr__(self, "vol", positive("vol", self.vol))

    def discount_factor(self, t: float) -> float:
        """Price today of 1 unit of currency paid at time t: DF(t) = exp(−r·t).

        Args:
            t: time in years from now, >= 0.
        """
        return math.exp(-self.rate * non_negative("t", t))

    def forward(self, t: float) -> float:
        """Forward price for delivery at time t: F(t) = S·exp((r − q)·t).

        Args:
            t: time in years from now, >= 0.
        """
        return self.spot * math.exp((self.rate - self.div_yield) * non_negative("t", t))

    @property
    def snapshot_id(self) -> str:
        """Content hash identifying these inputs: ``"sha256:<64 hex chars>"``.

        The hash covers a canonical JSON form: sorted keys, no whitespace, a kind
        and schema version, and each float written with ``repr`` (the shortest
        string that round-trips exactly). Equal inputs therefore always give the
        same id, on any machine, and any change to any field gives a new one.
        """
        canonical = json.dumps(
            {
                "kind": _SNAPSHOT_KIND,
                "version": _SNAPSHOT_VERSION,
                "spot": repr(self.spot),
                "rate": repr(self.rate),
                "div_yield": repr(self.div_yield),
                "vol": repr(self.vol),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
