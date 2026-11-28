r"""Bump-and-revalue greeks by Monte Carlo with common random numbers.

Each greek is a finite difference of MC prices (units: docs/WHEN_IN_DOUBT.md):

    delta = [V(S + h) − V(S − h)] / 2h                 h = 1% · S
    gamma = [V(S + h) − 2·V(S) + V(S − h)] / h²
    vega  = [V(σ + k) − V(σ − k)] / 2k                 k = min(1 vol point, σ/2)
    rho   = [V(r + b) − V(r − b)] / 2b                 b = 1 bp
    theta = [V(dates − τ) − V(dates)] / τ              τ = min(1 day, t₁/2)

Theta lets calendar time pass with the market unchanged: expiry and every
observation date move τ years closer. t₁ is the first date ahead (expiry, or
the first observation of an autocallable), so no date is moved into the past.
The vol bump shrinks the same way so that σ − k stays positive.

Common random numbers: every revaluation uses the same seed, so sample k of
each revaluation comes from the same normal draws (the time grid only changes
for theta, where it shifts by at most one step). Each greek is therefore
computed sample by sample, as a fixed combination of revaluation samples; its
value is the mean and its standard error is s/√n over those per-sample greeks.
The shared noise cancels inside every sample, and the standard error accounts
for whatever is left, including the large noise of gamma near a barrier or
knock-in level (decision D-22: every greek comes with its standard error, and
the reporting layers decide what is too noisy to show).

An autocallable struck today (``initial_spot`` None) has its initial spot
pinned to today's spot before bumping, so spot bumps move spot, not the levels
(decision D-21).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import assert_never

from deskgraph.core import monte_carlo as mc
from deskgraph.core.instruments import Autocallable, Barrier, European, Instrument
from deskgraph.core.market import MarketInputs
from deskgraph.core.monte_carlo import FloatArray, MCConfig, MCResult
from deskgraph.core.results import Greeks

SPOT_BUMP = 0.01  # relative: 1% of spot
VOL_BUMP = 0.01  # absolute: 1 vol point
RATE_BUMP = 0.0001  # absolute: 1 bp
TIME_BUMP = 1 / 365  # one calendar day, in years


@dataclass(frozen=True, slots=True)
class BumpGreeks:
    """Monte Carlo greeks with their standard errors.

    Attributes:
        value: the greeks, in the units of ``Greeks``.
        std_error: standard error of each greek, same units.
        price: the unbumped MC price (with its seed and path count, for lineage).
    """

    value: Greeks
    std_error: Greeks
    price: MCResult


def bump_greeks(instrument: Instrument, market: MarketInputs, config: MCConfig) -> BumpGreeks:
    """Delta, gamma, vega, theta and rho by central bumps (theta: one-sided) on shared draws."""
    inst = _pin_initial_spot(instrument, market)

    def samples(i: Instrument = inst, m: MarketInputs = market) -> FloatArray:
        return mc.price_samples(i, m, config)

    base = samples()

    h = SPOT_BUMP * market.spot
    spot_up = samples(m=replace(market, spot=market.spot + h))
    spot_down = samples(m=replace(market, spot=market.spot - h))

    k = min(VOL_BUMP, 0.5 * market.vol)
    vol_up = samples(m=replace(market, vol=market.vol + k))
    vol_down = samples(m=replace(market, vol=market.vol - k))

    rate_up = samples(m=replace(market, rate=market.rate + RATE_BUMP))
    rate_down = samples(m=replace(market, rate=market.rate - RATE_BUMP))

    tau = min(TIME_BUMP, 0.5 * _first_date(inst))
    aged = samples(i=_age(inst, tau))

    per_sample = {
        "delta": (spot_up - spot_down) / (2 * h),
        "gamma": (spot_up - 2 * base + spot_down) / (h * h),
        "vega": (vol_up - vol_down) / (2 * k),
        "theta": (aged - base) / tau,
        "rho": (rate_up - rate_down) / (2 * RATE_BUMP),
    }
    summaries = {name: mc.summarise(x, config) for name, x in per_sample.items()}
    return BumpGreeks(
        value=Greeks(**{name: s.price for name, s in summaries.items()}),
        std_error=Greeks(**{name: s.std_error for name, s in summaries.items()}),
        price=mc.summarise(base, config),
    )


def _pin_initial_spot(instrument: Instrument, market: MarketInputs) -> Instrument:
    if isinstance(instrument, Autocallable) and instrument.initial_spot is None:
        return replace(instrument, initial_spot=market.spot)
    return instrument


def _first_date(instrument: Instrument) -> float:
    """Years to the first date the payoff depends on (an observation or expiry)."""
    match instrument:
        case European() | Barrier():
            return instrument.expiry
        case Autocallable():
            return instrument.observation_times[0]
        case _:
            assert_never(instrument)


def _age(instrument: Instrument, tau: float) -> Instrument:
    """The same contract ``tau`` years later: every date is ``tau`` years closer."""
    match instrument:
        case European() | Barrier():
            return replace(instrument, expiry=instrument.expiry - tau)
        case Autocallable():
            return replace(
                instrument,
                expiry=instrument.expiry - tau,
                observation_times=tuple(t - tau for t in instrument.observation_times),
            )
        case _:
            assert_never(instrument)
