"""Bump-and-revalue greeks with common random numbers.

Under common random numbers every revaluation reuses the same draws, so each MC
bump greek is an unbiased estimate of the *finite difference* of the true price
with the same bump sizes. The checks therefore compare against Black-Scholes
finite differences built with the same bumps, within 4 standard errors, which
tests the bump convention exactly instead of hiding it inside a tolerance.
"""

from __future__ import annotations

import math
from dataclasses import fields, replace

import pytest

from deskgraph.core import black_scholes as bs
from deskgraph.core import monte_carlo as mc
from deskgraph.core.greeks import (
    RATE_BUMP,
    SPOT_BUMP,
    TIME_BUMP,
    VOL_BUMP,
    BumpGreeks,
    bump_greeks,
)
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
from deskgraph.core.results import Greeks

MKT = MarketInputs(spot=100.0, rate=0.05, div_yield=0.02, vol=0.25)
GREEK_NAMES = tuple(f.name for f in fields(Greeks))


def test_bump_sizes_follow_the_conventions() -> None:
    # docs/WHEN_IN_DOUBT.md: spot 1% relative, vol 1 point, rate 1 bp, time 1 day.
    assert (SPOT_BUMP, VOL_BUMP, RATE_BUMP, TIME_BUMP) == (0.01, 0.01, 0.0001, 1 / 365)


def bs_finite_differences(inst: European, mkt: MarketInputs) -> Greeks:
    """Black-Scholes greeks by the same bumps bump_greeks uses."""
    h_s = SPOT_BUMP * mkt.spot
    h_v = min(VOL_BUMP, 0.5 * mkt.vol)
    h_t = min(TIME_BUMP, 0.5 * inst.expiry)

    def p(m: MarketInputs = mkt, i: European = inst) -> float:
        return bs.price(i, m)

    up, down = p(replace(mkt, spot=mkt.spot + h_s)), p(replace(mkt, spot=mkt.spot - h_s))
    vol_up, vol_down = p(replace(mkt, vol=mkt.vol + h_v)), p(replace(mkt, vol=mkt.vol - h_v))
    rate_up = p(replace(mkt, rate=mkt.rate + RATE_BUMP))
    rate_down = p(replace(mkt, rate=mkt.rate - RATE_BUMP))
    return Greeks(
        delta=(up - down) / (2 * h_s),
        gamma=(up - 2 * p() + down) / (h_s * h_s),
        vega=(vol_up - vol_down) / (2 * h_v),
        theta=(p(i=replace(inst, expiry=inst.expiry - h_t)) - p()) / h_t,
        rho=(rate_up - rate_down) / (2 * RATE_BUMP),
    )


def assert_within_4se(result: BumpGreeks, target: Greeks, extra_se: Greeks | None = None) -> None:
    for name in GREEK_NAMES:
        est, se = getattr(result.value, name), getattr(result.std_error, name)
        if extra_se is not None:
            se += getattr(extra_se, name)
        expected = getattr(target, name)
        # The absolute floor covers greeks that are zero up to rounding (SE ~ 1e-15).
        assert abs(est - expected) <= 4 * se + 1e-9, f"{name}: {est} vs {expected} (se {se})"


# --- Europeans ------------------------------------------------------------------

EUROPEAN_CFG = MCConfig(n_paths=200_000, seed=41)


@pytest.mark.parametrize(
    ("inst", "mkt"),
    [
        (European(OptionType.CALL, 100.0, 1.0), MKT),
        (European(OptionType.PUT, 90.0, 0.5), MKT),
        (European(OptionType.CALL, 120.0, 2.0), MKT),
        # Expiry under one day: the time bump shrinks to half the expiry.
        (European(OptionType.PUT, 100.0, 0.002), MKT),
        # Vol under 2 points: the vol bump shrinks to half the vol. Strike near the
        # forward (103.05); far from it the option is a forward up to ~1e-9 that no
        # path can see.
        (European(OptionType.CALL, 103.0, 1.0), replace(MKT, vol=0.005)),
    ],
)
def test_european_bump_greeks_within_4se_of_black_scholes(
    inst: European, mkt: MarketInputs
) -> None:
    assert_within_4se(bump_greeks(inst, mkt, EUROPEAN_CFG), bs_finite_differences(inst, mkt))


def test_bump_greeks_are_close_to_analytic_greeks() -> None:
    # The bump convention itself is a good approximation of the true derivatives.
    inst = European(OptionType.CALL, 100.0, 1.0)
    value, exact = bump_greeks(inst, MKT, EUROPEAN_CFG).value, bs.greeks(inst, MKT)
    assert value.delta == pytest.approx(exact.delta, abs=2e-3)
    assert value.gamma == pytest.approx(exact.gamma, rel=0.05)
    assert value.vega == pytest.approx(exact.vega, rel=0.01)
    assert value.theta == pytest.approx(exact.theta, rel=0.01)
    assert value.rho == pytest.approx(exact.rho, rel=0.01)


def test_common_random_numbers_shrink_the_delta_standard_error() -> None:
    # Two independent prices would give a delta SE of √2·SE(price)/(2h); reusing
    # the same draws cancels most of that noise.
    cfg = MCConfig(n_paths=100_000, seed=5, control_variate=False)
    result = bump_greeks(European(OptionType.CALL, 100.0, 1.0), MKT, cfg)
    independent_se = math.sqrt(2) * result.price.std_error / (2 * SPOT_BUMP * MKT.spot)
    assert result.std_error.delta < 0.2 * independent_se


def test_same_inputs_give_identical_greeks() -> None:
    inst = European(OptionType.PUT, 95.0, 0.75)
    assert bump_greeks(inst, MKT, EUROPEAN_CFG) == bump_greeks(inst, MKT, EUROPEAN_CFG)


def test_result_carries_the_base_price_for_lineage() -> None:
    inst = European(OptionType.CALL, 105.0, 1.0)
    assert bump_greeks(inst, MKT, EUROPEAN_CFG).price == mc.price(inst, MKT, EUROPEAN_CFG)


# --- barriers -------------------------------------------------------------------

BARRIER_CFG = MCConfig(n_paths=20_000, seed=43)


@pytest.mark.parametrize(
    ("option_type", "level", "knock_in", "knock_out"),
    [
        (OptionType.CALL, 85.0, BarrierType.DOWN_IN, BarrierType.DOWN_OUT),
        (OptionType.PUT, 115.0, BarrierType.UP_IN, BarrierType.UP_OUT),
    ],
)
def test_knock_in_plus_knock_out_greeks_match_the_vanilla(
    option_type: OptionType, level: float, knock_in: BarrierType, knock_out: BarrierType
) -> None:
    # Per path, in + out = vanilla, and every bump reuses the same paths, so the
    # greeks add up too. SE(in + out) <= SE(in) + SE(out) bounds the noise.
    def greeks_of(kind: BarrierType) -> BumpGreeks:
        return bump_greeks(Barrier(option_type, 100.0, 0.5, level, kind), MKT, BARRIER_CFG)

    ins, outs = greeks_of(knock_in), greeks_of(knock_out)
    total = BumpGreeks(
        value=Greeks(*(getattr(ins.value, n) + getattr(outs.value, n) for n in GREEK_NAMES)),
        std_error=ins.std_error,
        price=ins.price,
    )
    vanilla = bs_finite_differences(European(option_type, 100.0, 0.5), MKT)
    assert_within_4se(total, vanilla, extra_se=outs.std_error)


def test_every_greek_comes_with_a_finite_non_negative_standard_error() -> None:
    # Near a barrier gamma is noisy (D-22): it is reported with its SE,
    # never hidden; the API and AI layers decide what is too noisy to show.
    result = bump_greeks(Barrier(OptionType.CALL, 100.0, 0.5, 97.0, BarrierType.DOWN_OUT),
                         MKT, BARRIER_CFG)  # fmt: skip
    for name in GREEK_NAMES:
        se = getattr(result.std_error, name)
        assert math.isfinite(se)
        assert se >= 0.0
    assert result.std_error.gamma > 0.0


# --- autocallables --------------------------------------------------------------

NOTE_CFG = MCConfig(n_paths=20_000, seed=47)


def note(**overrides: object) -> Autocallable:
    terms: dict[str, object] = {
        "expiry": 1.0,
        "observation_times": (0.25, 0.5, 0.75, 1.0),
        "autocall_level": 1.0,
        "coupon": 0.025,
        "knock_in_level": 0.6,
        "knock_in_monitoring": Monitoring.DAILY,
    }
    terms.update(overrides)
    return Autocallable(**terms)  # type: ignore[arg-type]


def test_note_struck_today_has_its_initial_spot_pinned_before_bumping() -> None:
    assert bump_greeks(note(), MKT, NOTE_CFG) == bump_greeks(
        note(initial_spot=MKT.spot), MKT, NOTE_CFG
    )


def test_note_delta_is_positive() -> None:
    # Higher spot: more early calls and fewer knock-ins, so the note is worth more.
    # (Before initial_spot existed the levels moved with spot and delta was 0.)
    result = bump_greeks(note(), MKT, NOTE_CFG)
    assert result.value.delta > 5 * result.std_error.delta


def test_note_theta_moves_every_observation_date_one_day_closer() -> None:
    pinned = note(initial_spot=MKT.spot)
    aged = replace(
        pinned,
        expiry=pinned.expiry - TIME_BUMP,
        observation_times=tuple(t - TIME_BUMP for t in pinned.observation_times),
    )
    aged_price, price = mc.price(aged, MKT, NOTE_CFG).price, mc.price(pinned, MKT, NOTE_CFG).price
    expected = (aged_price - price) / TIME_BUMP
    assert bump_greeks(note(), MKT, NOTE_CFG).value.theta == pytest.approx(expected, rel=1e-12)


def test_note_with_an_observation_under_a_day_away_still_has_a_theta() -> None:
    # The time bump shrinks to half the time to the first observation, so no
    # observation date is moved into the past.
    result = bump_greeks(note(observation_times=(0.001, 0.5, 1.0)), MKT, NOTE_CFG)
    assert math.isfinite(result.value.theta)
