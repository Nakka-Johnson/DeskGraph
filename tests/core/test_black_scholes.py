import math
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest

from deskgraph.core import black_scholes as bs
from deskgraph.core.instruments import European, OptionType
from deskgraph.core.market import MarketInputs

CALL, PUT = OptionType.CALL, OptionType.PUT

# (spot, strike, expiry, rate, div_yield, vol): ATM, ITM, OTM, with and without
# dividends, short and long dated, low and high vol, negative rate.
GRID = [
    (42.0, 40.0, 0.5, 0.10, 0.00, 0.20),
    (100.0, 100.0, 1.0, 0.05, 0.02, 0.25),
    (100.0, 130.0, 0.25, 0.03, 0.01, 0.40),
    (100.0, 70.0, 2.0, 0.01, 0.03, 0.15),
    (930.0, 900.0, 2 / 12, 0.08, 0.03, 0.20),
    (50.0, 55.0, 3.0, -0.005, 0.00, 0.60),
]


def setup(case: tuple[float, ...], option_type: OptionType) -> tuple[European, MarketInputs]:
    spot, strike, expiry, rate, div_yield, vol = case
    return (
        European(option_type, strike=strike, expiry=expiry),
        MarketInputs(spot=spot, rate=rate, div_yield=div_yield, vol=vol),
    )


# --- textbook values -----------------------------------------------------------


def test_hull_example_call_and_put() -> None:
    # Hull, Options, Futures, and Other Derivatives: S=42, K=40, r=10%, σ=20%, T=0.5.
    mkt = MarketInputs(spot=42.0, rate=0.10, div_yield=0.0, vol=0.20)
    assert bs.price(European(CALL, strike=40.0, expiry=0.5), mkt) == pytest.approx(4.76, abs=5e-3)
    assert bs.price(European(PUT, strike=40.0, expiry=0.5), mkt) == pytest.approx(0.81, abs=5e-3)


def test_hull_index_option_with_dividend_yield() -> None:
    # Hull, index option example: S=930, K=900, r=8%, q=3%, σ=20%, T=2 months -> 51.83.
    mkt = MarketInputs(spot=930.0, rate=0.08, div_yield=0.03, vol=0.20)
    call = European(CALL, strike=900.0, expiry=2 / 12)
    assert bs.price(call, mkt) == pytest.approx(51.83, abs=1e-2)


# --- identities ----------------------------------------------------------------


@pytest.mark.parametrize("case", GRID)
def test_put_call_parity(case: tuple[float, ...]) -> None:
    call, mkt = setup(case, CALL)
    put, _ = setup(case, PUT)
    t = call.expiry
    lhs = bs.price(call, mkt) - bs.price(put, mkt)
    rhs = mkt.spot * math.exp(-mkt.div_yield * t) - call.strike * mkt.discount_factor(t)
    assert lhs == pytest.approx(rhs, abs=1e-10 * mkt.spot)


@pytest.mark.parametrize("case", GRID)
def test_call_put_greek_identities(case: tuple[float, ...]) -> None:
    call, mkt = setup(case, CALL)
    put, _ = setup(case, PUT)
    gc, gp = bs.greeks(call, mkt), bs.greeks(put, mkt)
    assert gc.delta - gp.delta == pytest.approx(math.exp(-mkt.div_yield * call.expiry), abs=1e-12)
    assert gc.gamma == pytest.approx(gp.gamma, rel=1e-12)
    assert gc.vega == pytest.approx(gp.vega, rel=1e-12)


# --- analytic greeks against central finite differences -----------------------


def central(f: Callable[[float], float], x: float, h: float) -> float:
    return (f(x + h) - f(x - h)) / (2 * h)


@pytest.mark.parametrize("option_type", [CALL, PUT])
@pytest.mark.parametrize("case", GRID)
def test_greeks_match_finite_differences(case: tuple[float, ...], option_type: OptionType) -> None:
    inst, mkt = setup(case, option_type)
    g = bs.greeks(inst, mkt)

    def at_spot(s: float) -> float:
        return bs.price(inst, replace(mkt, spot=s))

    def at_vol(v: float) -> float:
        return bs.price(inst, replace(mkt, vol=v))

    def at_rate(r: float) -> float:
        return bs.price(inst, replace(mkt, rate=r))

    def at_expiry(t: float) -> float:
        return bs.price(replace(inst, expiry=t), mkt)

    hs = 1e-4 * mkt.spot
    hg = 1e-3 * mkt.spot
    fd_gamma = (at_spot(mkt.spot + hg) - 2 * at_spot(mkt.spot) + at_spot(mkt.spot - hg)) / hg**2

    assert g.delta == pytest.approx(central(at_spot, mkt.spot, hs), rel=1e-6, abs=1e-9)
    assert g.gamma == pytest.approx(fd_gamma, rel=1e-5, abs=1e-9)
    assert g.vega == pytest.approx(central(at_vol, mkt.vol, 1e-5), rel=1e-6, abs=1e-8)
    assert g.rho == pytest.approx(central(at_rate, mkt.rate, 1e-5), rel=1e-6, abs=1e-8)
    # theta is ∂V/∂t as calendar time passes, i.e. minus the derivative in expiry.
    assert g.theta == pytest.approx(-central(at_expiry, inst.expiry, 1e-6), rel=1e-5, abs=1e-8)


# --- limits ----------------------------------------------------------------------


@pytest.mark.parametrize(("strike", "intrinsic_call"), [(90.0, 10.0), (110.0, 0.0)])
def test_price_tends_to_intrinsic_at_expiry(strike: float, intrinsic_call: float) -> None:
    mkt = MarketInputs(spot=100.0, rate=0.05, div_yield=0.02, vol=0.3)
    call = European(CALL, strike=strike, expiry=1e-10)
    put = European(PUT, strike=strike, expiry=1e-10)
    assert bs.price(call, mkt) == pytest.approx(intrinsic_call, abs=1e-6)
    assert bs.price(put, mkt) == pytest.approx(max(strike - 100.0, 0.0), abs=1e-6)


@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
def test_price_tends_to_discounted_forward_intrinsic_at_zero_vol(strike: float) -> None:
    mkt = MarketInputs(spot=100.0, rate=0.05, div_yield=0.02, vol=1e-9)
    t = 1.0
    fwd_value = 100.0 * math.exp(-0.02 * t) - strike * math.exp(-0.05 * t)
    assert bs.price(European(CALL, strike, t), mkt) == pytest.approx(max(fwd_value, 0.0), abs=1e-8)
    assert bs.price(European(PUT, strike, t), mkt) == pytest.approx(max(-fwd_value, 0.0), abs=1e-8)


def test_deep_out_of_the_money_prices_are_tiny_and_non_negative() -> None:
    mkt = MarketInputs(spot=100.0, rate=0.05, div_yield=0.0, vol=0.1)
    far_call = bs.price(European(CALL, strike=1000.0, expiry=0.1), mkt)
    far_put = bs.price(European(PUT, strike=10.0, expiry=0.1), mkt)
    assert 0.0 <= far_call < 1e-12
    assert 0.0 <= far_put < 1e-12


# --- vectorised pricing --------------------------------------------------------


def test_price_vec_matches_scalar_price_and_broadcasts() -> None:
    mkt = MarketInputs(spot=100.0, rate=0.05, div_yield=0.02, vol=0.25)
    strikes = np.array([80.0, 100.0, 120.0])
    is_call = np.array([True, False, True])
    out = bs.price_vec(is_call, mkt.spot, strikes, 1.0, mkt.rate, mkt.div_yield, mkt.vol)
    expected = [
        bs.price(European(CALL if c else PUT, strike=k, expiry=1.0), mkt)
        for c, k in zip(is_call, strikes, strict=True)
    ]
    assert out.shape == (3,)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, expected, rtol=1e-14)


@pytest.mark.parametrize(
    "bad",
    [
        {"spot": -1.0},
        {"strike": 0.0},
        {"expiry": 0.0},
        {"vol": -0.1},
        {"rate": math.nan},
    ],
)
def test_price_vec_rejects_invalid_inputs(bad: dict[str, float]) -> None:
    args: dict[str, float] = {
        "spot": 100.0, "strike": 100.0, "expiry": 1.0, "rate": 0.05, "div_yield": 0.0, "vol": 0.2,
    }  # fmt: skip
    args.update(bad)
    with pytest.raises(ValueError, match=next(iter(bad))):
        bs.price_vec(True, **args)
