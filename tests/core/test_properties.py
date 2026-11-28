"""Pricing laws that must hold for any valid input (Hypothesis).

Black-Scholes laws are checked on a wide input box. Monte Carlo laws are the
ones that hold path by path on shared draws (so exactly, not within standard
errors): they need few paths and catch payoff bugs the closed-form checks miss.

Tolerances only absorb float rounding: TOL is relative to the size of the
numbers involved (spot plus strike), far below anything a pricing bug produces.
"""

from __future__ import annotations

import math
from dataclasses import replace

from hypothesis import given, settings
from hypothesis import strategies as st

from deskgraph.core import black_scholes as bs
from deskgraph.core import monte_carlo as mc
from deskgraph.core.greeks import bump_greeks
from deskgraph.core.instruments import (
    Autocallable,
    Barrier,
    BarrierType,
    European,
    Monitoring,
    OptionType,
)
from deskgraph.core.market import MarketInputs
from deskgraph.core.monte_carlo import MCConfig

TOL = 1e-9

option_types = st.sampled_from(OptionType)
expiries = st.floats(0.01, 5.0)
vols = st.floats(0.05, 1.0)
seeds = st.integers(0, 2**32 - 1)


@st.composite
def markets(draw: st.DrawFn) -> MarketInputs:
    return MarketInputs(
        spot=draw(st.floats(1.0, 1000.0)),
        rate=draw(st.floats(-0.02, 0.10)),
        div_yield=draw(st.floats(0.0, 0.06)),
        vol=draw(vols),
    )


@st.composite
def europeans(draw: st.DrawFn, mkt: MarketInputs) -> European:
    strike = mkt.spot * draw(st.floats(0.5, 2.0))
    return European(draw(option_types), strike, draw(expiries))


@st.composite
def market_and_option(draw: st.DrawFn) -> tuple[MarketInputs, European]:
    mkt = draw(markets())
    return mkt, draw(europeans(mkt))


def scale(mkt: MarketInputs, inst: European) -> float:
    return TOL * (mkt.spot + inst.strike)


# --- Black-Scholes ----------------------------------------------------------------


@given(market_and_option())
def test_price_within_no_arbitrage_bounds(case: tuple[MarketInputs, European]) -> None:
    # Call: max(S·e^(−qT) − K·e^(−rT), 0) <= C <= S·e^(−qT); put mirrored.
    mkt, inst = case
    fwd_spot = mkt.spot * math.exp(-mkt.div_yield * inst.expiry)
    pv_strike = inst.strike * mkt.discount_factor(inst.expiry)
    upper, intrinsic = (
        (fwd_spot, fwd_spot - pv_strike)
        if inst.option_type is OptionType.CALL
        else (pv_strike, pv_strike - fwd_spot)
    )
    value = bs.price(inst, mkt)
    assert max(intrinsic, 0.0) - scale(mkt, inst) <= value <= upper + scale(mkt, inst)


@given(market_and_option())
def test_put_call_parity(case: tuple[MarketInputs, European]) -> None:
    mkt, inst = case
    call = bs.price(replace(inst, option_type=OptionType.CALL), mkt)
    put = bs.price(replace(inst, option_type=OptionType.PUT), mkt)
    forward_value = mkt.spot * math.exp(
        -mkt.div_yield * inst.expiry
    ) - inst.strike * mkt.discount_factor(inst.expiry)
    assert abs(call - put - forward_value) <= scale(mkt, inst)


@given(market_and_option(), vols, vols)
def test_price_rises_with_vol(case: tuple[MarketInputs, European], v1: float, v2: float) -> None:
    mkt, inst = case
    low, high = sorted((v1, v2))
    lower = bs.price(inst, replace(mkt, vol=low))
    higher = bs.price(inst, replace(mkt, vol=high))
    assert higher >= lower - scale(mkt, inst)


@given(market_and_option(), st.floats(1.0, 1000.0))
def test_call_rises_and_put_falls_with_spot(
    case: tuple[MarketInputs, European], other_spot: float
) -> None:
    mkt, inst = case
    low, high = sorted((mkt.spot, other_spot))
    at_low = bs.price(inst, replace(mkt, spot=low))
    at_high = bs.price(inst, replace(mkt, spot=high))
    tol = TOL * (high + inst.strike)
    if inst.option_type is OptionType.CALL:
        assert at_high >= at_low - tol
    else:
        assert at_high <= at_low + tol


@given(market_and_option(), st.floats(0.5, 2.0))
def test_call_falls_and_put_rises_with_strike(
    case: tuple[MarketInputs, European], ratio: float
) -> None:
    mkt, inst = case
    low, high = sorted((inst.strike, mkt.spot * ratio))
    at_low = bs.price(replace(inst, strike=low), mkt)
    at_high = bs.price(replace(inst, strike=high), mkt)
    tol = TOL * (mkt.spot + high)
    if inst.option_type is OptionType.CALL:
        assert at_high <= at_low + tol
    else:
        assert at_high >= at_low - tol


@given(market_and_option(), st.floats(0.01, 100.0))
def test_price_scales_with_spot_and_strike(
    case: tuple[MarketInputs, European], factor: float
) -> None:
    # Price is homogeneous of degree 1 in (S, K): V(λS, λK) = λ·V(S, K).
    mkt, inst = case
    scaled = bs.price(
        replace(inst, strike=factor * inst.strike), replace(mkt, spot=factor * mkt.spot)
    )
    assert abs(scaled - factor * bs.price(inst, mkt)) <= factor * scale(mkt, inst)


@given(market_and_option())
def test_greek_signs_and_bounds(case: tuple[MarketInputs, European]) -> None:
    mkt, inst = case
    g = bs.greeks(inst, mkt)
    carry = math.exp(-mkt.div_yield * inst.expiry)
    assert g.gamma >= 0.0
    assert g.vega >= 0.0
    if inst.option_type is OptionType.CALL:
        assert 0.0 <= g.delta <= carry
    else:
        assert -carry <= g.delta <= 0.0


# --- Monte Carlo: laws that hold path by path ---------------------------------------

MC_EXAMPLES = settings(max_examples=40)
SMALL = 256  # paths per run: enough for the laws below, which hold on every path
UNREACHABLE = {True: 1e-6, False: 1e12}  # down / up barrier spot never crosses


@st.composite
def barrier_cases(draw: st.DrawFn) -> tuple[MarketInputs, Barrier, Barrier, Barrier]:
    """(market, knock-in, knock-out, same option with an unreachable barrier = vanilla)."""
    mkt = draw(markets())
    is_down = draw(st.booleans())
    level = mkt.spot * draw(st.floats(0.6, 0.99) if is_down else st.floats(1.01, 1.4))
    knock_in, knock_out = (
        (BarrierType.DOWN_IN, BarrierType.DOWN_OUT)
        if is_down
        else (BarrierType.UP_IN, BarrierType.UP_OUT)
    )
    option_type = draw(option_types)
    strike = mkt.spot * draw(st.floats(0.7, 1.3))
    expiry = draw(st.floats(0.02, 0.5))

    def make(lvl: float, kind: BarrierType) -> Barrier:
        return Barrier(option_type, strike, expiry, lvl, kind)

    return (
        mkt,
        make(level, knock_in),
        make(level, knock_out),
        make(UNREACHABLE[is_down], knock_out),
    )


@MC_EXAMPLES
@given(barrier_cases(), seeds)
def test_knock_in_plus_knock_out_is_the_vanilla_on_every_path(
    case: tuple[MarketInputs, Barrier, Barrier, Barrier], seed: int
) -> None:
    mkt, knock_in, knock_out, vanilla = case
    cfg = MCConfig(n_paths=SMALL, seed=seed)
    total = mc.price_samples(knock_in, mkt, cfg) + mc.price_samples(knock_out, mkt, cfg)
    expected = mc.price_samples(vanilla, mkt, cfg)
    assert all(abs(total - expected) <= TOL * (mkt.spot + knock_in.strike))


@MC_EXAMPLES
@given(barrier_cases(), seeds)
def test_knock_out_is_worth_between_zero_and_the_vanilla(
    case: tuple[MarketInputs, Barrier, Barrier, Barrier], seed: int
) -> None:
    mkt, _, knock_out, vanilla = case
    cfg = MCConfig(n_paths=SMALL, seed=seed)
    out, plain = mc.price_samples(knock_out, mkt, cfg), mc.price_samples(vanilla, mkt, cfg)
    assert all(out >= 0.0)
    assert all(out <= plain)


@st.composite
def notes(draw: st.DrawFn) -> Autocallable:
    n_obs = draw(st.integers(1, 6))
    expiry = draw(st.floats(0.1, 3.0))
    autocall = draw(st.floats(0.0, 1.3))
    return Autocallable(
        expiry=expiry,
        observation_times=tuple(expiry * (i + 1) / n_obs for i in range(n_obs)),
        autocall_level=autocall,
        coupon=draw(st.floats(0.0, 0.2)),
        knock_in_level=autocall * draw(st.floats(0.0, 1.0)),
        knock_in_monitoring=draw(st.sampled_from(Monitoring)),
        initial_spot=draw(st.none() | st.floats(1.0, 1000.0)),
    )


@MC_EXAMPLES
@given(markets(), notes(), seeds)
def test_note_pays_between_zero_and_its_largest_possible_redemption(
    mkt: MarketInputs, note: Autocallable, seed: int
) -> None:
    # A called note pays at most 1 + n·c; an uncalled note pays 1, or X(T) < autocall
    # level (uncalled at maturity means X(T) is below it). Discounting at a negative
    # rate can lift a payment by up to 1/DF(T).
    n = len(note.observation_times)
    cap = max(1.0 + n * note.coupon, note.autocall_level) * max(
        1.0, mkt.discount_factor(note.expiry)
    )
    samples = mc.price_samples(note, mkt, MCConfig(n_paths=SMALL, seed=seed))
    assert all(samples >= 0.0)
    assert all(samples <= cap * (1 + TOL))


@MC_EXAMPLES
@given(markets(), notes(), st.floats(0.0, 0.2), seeds)
def test_note_is_worth_more_with_a_higher_coupon_on_every_path(
    mkt: MarketInputs, note: Autocallable, other_coupon: float, seed: int
) -> None:
    cfg = MCConfig(n_paths=SMALL, seed=seed)
    low, high = sorted((note.coupon, other_coupon))
    at_low = mc.price_samples(replace(note, coupon=low), mkt, cfg)
    at_high = mc.price_samples(replace(note, coupon=high), mkt, cfg)
    assert all(at_high >= at_low)


@MC_EXAMPLES
@given(market_and_option(), seeds)
def test_bump_gamma_of_a_vanilla_is_never_negative(
    case: tuple[MarketInputs, European], seed: int
) -> None:
    # The payoff is convex in S_T and S_T = S·g(Z) is linear in S, so on shared
    # draws each path's second difference is >= 0. Independent draws per bump
    # would make it negative about half the time at this path count.
    mkt, inst = case
    cfg = MCConfig(n_paths=SMALL, seed=seed, control_variate=False)
    gamma = bump_greeks(inst, mkt, cfg).value.gamma
    assert gamma >= -TOL / mkt.spot
