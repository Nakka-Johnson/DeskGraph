"""Result types shared by the pricers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Greeks:
    """Price sensitivities of one instrument (units: docs/WHEN_IN_DOUBT.md).

    Attributes:
        delta: ∂V/∂S, change in price per 1.00 move in spot.
        gamma: ∂²V/∂S², change in delta per 1.00 move in spot.
        vega: ∂V/∂σ, per 1.00 change in vol (multiply by 0.01 for one vol point).
        theta: ∂V/∂t as calendar time passes, per year (divide by 365 for one
            calendar day). Equal to −∂V/∂T, where T is time to expiry.
        rho: ∂V/∂r, per 1.00 change in the rate (multiply by 0.0001 for one bp).
    """

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
