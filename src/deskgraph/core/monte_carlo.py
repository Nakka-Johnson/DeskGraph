r"""Monte Carlo pricing under geometric Brownian motion.

Paths follow the exact GBM solution in log space, so there is no discretisation
bias in the distribution of spot at any grid time:

    ln S(t+Δt) = ln S(t) + (r − q − σ²/2)·Δt + σ·√Δt·Z,      Z ~ N(0, 1)

Reproducibility: every price is a function of (instrument, market, config). The
only randomness is a ``numpy.random.Generator`` built from ``config.seed``, and
normals are drawn in a fixed order, so the same inputs always give the same
number and bumped revaluations with the same seed reuse the same draws.

Variance reduction:

- Antithetic variates: each draw Z is paired with −Z. The two payoffs of a pair
  are negatively correlated, not independent, so the estimator averages each
  pair first and computes the standard error over the n/2 pair means.
- Control variate (Europeans only): the discounted terminal spot X = e^(−rT)·S_T
  has known mean S₀·e^(−qT). The estimate Y − β·(X − E[X]) uses β = Cov(Y, X) /
  Var(X) estimated from the same sample (bias O(1/n), negligible here).

Time grids: barriers step daily (``steps_per_year`` steps per year, rounded up
so the last step lands on expiry); autocallables step from one observation date
to the next, daily in between when knock-in is monitored daily. Only the current
spot of each path is kept, so memory is O(paths), not O(paths × steps).

Barrier and knock-in levels are also checked at t = 0, so a barrier that spot
has already crossed today counts as hit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import assert_never

import numpy as np
import numpy.typing as npt

from deskgraph.core._validation import as_int
from deskgraph.core.instruments import (
    Autocallable,
    Barrier,
    European,
    Instrument,
    Monitoring,
    OptionType,
)
from deskgraph.core.market import MarketInputs

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

# Absorbs float noise when converting an interval to a whole number of steps,
# so 0.25 years at 252 steps per year is 63 steps, not 64.
_STEP_ROUNDING = 1e-9


@dataclass(frozen=True, slots=True)
class MCConfig:
    """Monte Carlo settings; together with the inputs they fully determine the price.

    Attributes:
        n_paths: number of simulated paths. Even when ``antithetic`` (each draw
            and its mirror are two paths); at least 2 independent samples.
        seed: seed for ``numpy.random.default_rng``, >= 0.
        antithetic: pair every draw with its negative.
        control_variate: use the discounted terminal spot as a control variate
            (Europeans only; ignored for path-dependent products).
        steps_per_year: monitoring frequency for daily barriers and knock-ins.
    """

    n_paths: int
    seed: int
    antithetic: bool = True
    control_variate: bool = True
    steps_per_year: int = 252

    def __post_init__(self) -> None:
        n_paths = as_int("n_paths", self.n_paths)
        seed = as_int("seed", self.seed)
        steps = as_int("steps_per_year", self.steps_per_year)
        if self.antithetic and n_paths % 2:
            raise ValueError(f"n_paths must be even with antithetic variates, got {n_paths}")
        samples = n_paths // 2 if self.antithetic else n_paths
        if samples < 2:
            raise ValueError(
                f"n_paths={n_paths} gives {samples} independent sample(s); "
                "at least 2 are needed for a standard error"
            )
        if seed < 0:
            raise ValueError(f"seed must be >= 0, got {seed}")
        if steps < 1:
            raise ValueError(f"steps_per_year must be >= 1, got {steps}")
        object.__setattr__(self, "n_paths", n_paths)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "steps_per_year", steps)


@dataclass(frozen=True, slots=True)
class MCResult:
    """Monte Carlo estimate.

    Attributes:
        price: estimated present value (per unit underlying, or per unit
            notional for autocallables).
        std_error: standard error of ``price``, over independent samples (pair
            means when antithetic).
        n_paths: paths simulated, for lineage.
        seed: seed used, for lineage.
    """

    price: float
    std_error: float
    n_paths: int
    seed: int


def price(instrument: Instrument, market: MarketInputs, config: MCConfig) -> MCResult:
    """Price any supported instrument by Monte Carlo."""
    return summarise(price_samples(instrument, market, config), config)


def summarise(samples: FloatArray, config: MCConfig) -> MCResult:
    """Mean and standard error of independent samples: SE = s / √n, s with ddof = 1."""
    std_error = float(np.std(samples, ddof=1)) / math.sqrt(samples.size)
    return MCResult(float(np.mean(samples)), std_error, config.n_paths, config.seed)


def price_samples(instrument: Instrument, market: MarketInputs, config: MCConfig) -> FloatArray:
    """Independent, identically distributed samples whose mean is the MC price.

    One sample per path, or per antithetic pair (the pair mean). Sample k comes
    from the same draws for any market and any instrument with the same time
    grid, which is what lets bump greeks difference samples path by path.
    """
    rng = np.random.default_rng(config.seed)
    match instrument:
        case European():
            return _european(instrument, market, config, rng)
        case Barrier():
            return _barrier(instrument, market, config, rng)
        case Autocallable():
            return _autocallable(instrument, market, config, rng)
        case _:
            assert_never(instrument)


# --- products -------------------------------------------------------------------


def _european(
    inst: European, mkt: MarketInputs, cfg: MCConfig, rng: np.random.Generator
) -> FloatArray:
    """Discounted payoff samples from one exact step to expiry."""
    t = inst.expiry
    z = _normals(rng, cfg)
    s_t = mkt.spot * np.exp(_drift(mkt) * t + mkt.vol * math.sqrt(t) * z)
    df = mkt.discount_factor(t)
    y = _pair_means(df * _intrinsic(inst.option_type, s_t, inst.strike), cfg)
    if not cfg.control_variate:
        return y
    x = _pair_means(df * s_t, cfg)
    return _apply_control_variate(y, x, known_mean=mkt.spot * math.exp(-mkt.div_yield * t))


def _barrier(
    inst: Barrier, mkt: MarketInputs, cfg: MCConfig, rng: np.random.Generator
) -> FloatArray:
    """Daily-monitored single barrier with a running hit flag per path."""
    log_barrier = math.log(inst.barrier)
    is_down = inst.barrier_type.is_down

    def crossed(log_s: FloatArray) -> BoolArray:
        out: BoolArray = log_s <= log_barrier if is_down else log_s >= log_barrier
        return out

    log_s = np.full(cfg.n_paths, math.log(mkt.spot))
    hit = crossed(log_s)  # already through the barrier today
    n_steps = _steps(inst.expiry, cfg.steps_per_year)
    dt = inst.expiry / n_steps
    for _ in range(n_steps):
        log_s += _drift(mkt) * dt + mkt.vol * math.sqrt(dt) * _normals(rng, cfg)
        hit |= crossed(log_s)

    vanilla = _intrinsic(inst.option_type, np.exp(log_s), inst.strike)
    alive = hit if inst.barrier_type.is_knock_in else ~hit
    payoff = np.where(alive, vanilla, 0.0)
    return _pair_means(mkt.discount_factor(inst.expiry) * payoff, cfg)


def _autocallable(
    inst: Autocallable, mkt: MarketInputs, cfg: MCConfig, rng: np.random.Generator
) -> FloatArray:
    """Autocallable per unit notional; payoff rules in ``Autocallable``'s docstring.

    Works with log X, where X = S/S₀ starts at S/initial_spot (0 for a note
    struck today). A level of 0 maps to −∞, so an autocall level of 0 calls
    every path and a knock-in level of 0 never knocks in. Daily knock-in also
    checks today's X (D-20); early redemption is only checked on observation dates.
    """
    log_autocall = _log_level(inst.autocall_level)
    log_knock_in = _log_level(inst.knock_in_level)
    daily = inst.knock_in_monitoring is Monitoring.DAILY
    log_x0 = 0.0 if inst.initial_spot is None else math.log(mkt.spot / inst.initial_spot)

    log_x = np.full(cfg.n_paths, log_x0)
    alive = np.ones(cfg.n_paths, dtype=bool)  # not yet called
    payoff = np.zeros(cfg.n_paths)
    knocked = np.full(cfg.n_paths, daily and log_x0 < log_knock_in)

    t_prev = 0.0
    for i, t_obs in enumerate(inst.observation_times, start=1):
        n_steps = _steps(t_obs - t_prev, cfg.steps_per_year) if daily else 1
        dt = (t_obs - t_prev) / n_steps
        for _ in range(n_steps):
            log_x += _drift(mkt) * dt + mkt.vol * math.sqrt(dt) * _normals(rng, cfg)
            if daily:
                knocked |= log_x < log_knock_in
        called = alive & (log_x >= log_autocall)
        payoff[called] = (1.0 + i * inst.coupon) * mkt.discount_factor(t_obs)
        alive &= ~called
        t_prev = t_obs

    if not daily:
        knocked = log_x < log_knock_in
    redemption = np.where(knocked, np.exp(log_x), 1.0)
    payoff[alive] = mkt.discount_factor(inst.expiry) * redemption[alive]
    return _pair_means(payoff, cfg)


# --- building blocks ---------------------------------------------------------


def _drift(mkt: MarketInputs) -> float:
    """Risk-neutral log drift per year: r − q − σ²/2."""
    return mkt.rate - mkt.div_yield - 0.5 * mkt.vol * mkt.vol


def _normals(rng: np.random.Generator, cfg: MCConfig) -> FloatArray:
    """One standard normal per path; with antithetic, the second half mirrors the first."""
    if not cfg.antithetic:
        return rng.standard_normal(cfg.n_paths)
    z = rng.standard_normal(cfg.n_paths // 2)
    return np.concatenate([z, -z])


def _pair_means(values: FloatArray, cfg: MCConfig) -> FloatArray:
    """Independent samples: antithetic pair means, or the values themselves."""
    if not cfg.antithetic:
        return values
    half = cfg.n_paths // 2
    out: FloatArray = 0.5 * (values[:half] + values[half:])
    return out


def _apply_control_variate(y: FloatArray, x: FloatArray, known_mean: float) -> FloatArray:
    x_centred = x - x.mean()
    var_x = float(np.mean(x_centred * x_centred))
    if var_x == 0.0:
        return y
    beta = float(np.mean((y - y.mean()) * x_centred)) / var_x
    out: FloatArray = y - beta * (x - known_mean)
    return out


def _intrinsic(option_type: OptionType, spot: FloatArray, strike: float) -> FloatArray:
    out: FloatArray
    if option_type is OptionType.CALL:
        out = np.maximum(spot - strike, 0.0)
    else:
        out = np.maximum(strike - spot, 0.0)
    return out


def _steps(interval: float, steps_per_year: int) -> int:
    """Whole number of equal steps covering ``interval`` years at the given frequency."""
    return max(1, math.ceil(interval * steps_per_year - _STEP_ROUNDING))


def _log_level(level: float) -> float:
    return math.log(level) if level > 0.0 else -math.inf
