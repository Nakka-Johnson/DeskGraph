r"""Black-Scholes-Merton prices and analytic greeks for European options.

Model: the underlying follows geometric Brownian motion with constant rate r,
continuous dividend yield q and volatility σ. With time to expiry T,

    d₁ = [ln(S/K) + (r − q + σ²/2)·T] / (σ·√T),        d₂ = d₁ − σ·√T

    call = S·e^(−qT)·N(d₁) − K·e^(−rT)·N(d₂)
    put  = K·e^(−rT)·N(−d₂) − S·e^(−qT)·N(−d₁)

where N is the standard normal CDF and n its density. Puts use N(−d) rather than
1 − N(d) so deep in-the-money and out-of-the-money values keep full precision.

Units follow docs/WHEN_IN_DOUBT.md: prices per one unit of underlying; vega per
1.00 of vol; theta per year of calendar time; rho per 1.00 of rate.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from scipy.special import ndtr

from deskgraph.core.instruments import European, OptionType
from deskgraph.core.market import MarketInputs
from deskgraph.core.results import Greeks

FloatArray = npt.NDArray[np.float64]

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def price(instrument: European, market: MarketInputs) -> float:
    """Black-Scholes-Merton price of a European call or put, per unit of underlying."""
    out = price_vec(
        instrument.option_type is OptionType.CALL,
        market.spot,
        instrument.strike,
        instrument.expiry,
        market.rate,
        market.div_yield,
        market.vol,
    )
    return float(out)


def price_vec(
    is_call: npt.ArrayLike,
    spot: npt.ArrayLike,
    strike: npt.ArrayLike,
    expiry: npt.ArrayLike,
    rate: npt.ArrayLike,
    div_yield: npt.ArrayLike,
    vol: npt.ArrayLike,
) -> FloatArray:
    """Vectorised Black-Scholes-Merton prices.

    All arguments broadcast against each other with numpy rules, so one call can
    price a whole strike grid or a book of mixed calls and puts.

    Args:
        is_call: True for calls, False for puts.
        spot: S > 0.
        strike: K > 0.
        expiry: T > 0, years.
        rate: r, continuously compounded, decimal per year.
        div_yield: q, continuously compounded, decimal per year.
        vol: σ > 0, decimal per year.

    Returns:
        Prices as a float64 array of the broadcast shape (0-d for scalar inputs).
    """
    c = np.asarray(is_call, dtype=bool)
    s = _checked("spot", spot, positive=True)
    k = _checked("strike", strike, positive=True)
    t = _checked("expiry", expiry, positive=True)
    r = _checked("rate", rate)
    q = _checked("div_yield", div_yield)
    v = _checked("vol", vol, positive=True)

    d1, d2 = _d1_d2(s, k, t, r, q, v)
    disc_spot = s * np.exp(-q * t)
    disc_strike = k * np.exp(-r * t)
    call = disc_spot * ndtr(d1) - disc_strike * ndtr(d2)
    put = disc_strike * ndtr(-d2) - disc_spot * ndtr(-d1)
    out: FloatArray = np.where(c, call, put)
    return out


def greeks(instrument: European, market: MarketInputs) -> Greeks:
    """Analytic Black-Scholes-Merton greeks.

    With δq = e^(−qT), δr = e^(−rT) and ± meaning + for calls and − for puts:

        delta = ±δq·N(±d₁)
        gamma = δq·n(d₁) / (S·σ·√T)
        vega  = S·δq·n(d₁)·√T
        theta = −S·δq·n(d₁)·σ / (2√T) ∓ r·K·δr·N(±d₂) ± q·S·δq·N(±d₁)
        rho   = ±K·T·δr·N(±d₂)
    """
    s, k, t = market.spot, instrument.strike, instrument.expiry
    r, q, v = market.rate, market.div_yield, market.vol
    sign = 1.0 if instrument.option_type is OptionType.CALL else -1.0

    d1_arr, d2_arr = _d1_d2(*(np.float64(x) for x in (s, k, t, r, q, v)))
    d1, d2 = float(d1_arr), float(d2_arr)
    dq, dr = math.exp(-q * t), math.exp(-r * t)
    pdf_d1 = _INV_SQRT_2PI * math.exp(-0.5 * d1 * d1)
    n_d1 = float(ndtr(sign * d1))
    n_d2 = float(ndtr(sign * d2))
    sqrt_t = math.sqrt(t)

    return Greeks(
        delta=sign * dq * n_d1,
        gamma=dq * pdf_d1 / (s * v * sqrt_t),
        vega=s * dq * pdf_d1 * sqrt_t,
        theta=(
            -s * dq * pdf_d1 * v / (2.0 * sqrt_t)
            - sign * r * k * dr * n_d2
            + sign * q * s * dq * n_d1
        ),
        rho=sign * k * t * dr * n_d2,
    )


def _d1_d2(
    s: npt.ArrayLike,
    k: npt.ArrayLike,
    t: npt.ArrayLike,
    r: npt.ArrayLike,
    q: npt.ArrayLike,
    v: npt.ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    s_, k_, t_, r_, q_, v_ = (np.asarray(x, dtype=np.float64) for x in (s, k, t, r, q, v))
    vol_sqrt_t = v_ * np.sqrt(t_)
    d1: FloatArray = (np.log(s_ / k_) + (r_ - q_ + 0.5 * v_ * v_) * t_) / vol_sqrt_t
    d2: FloatArray = d1 - vol_sqrt_t
    return d1, d2


def _checked(name: str, value: npt.ArrayLike, *, positive: bool = False) -> FloatArray:
    arr = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    if positive and not np.all(arr > 0.0):
        raise ValueError(f"{name} must be > 0")
    return arr
