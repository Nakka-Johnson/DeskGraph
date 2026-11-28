"""Monte Carlo engine tests.

Seeds are fixed, so every test is deterministic. The 3-standard-error checks were
written before the engine; if one fails, find out why rather than changing the seed.
"""

import math

import numpy as np
import pytest
from scipy.special import ndtr

from deskgraph.core import black_scholes as bs
from deskgraph.core import monte_carlo as mc
from deskgraph.core.instruments import (
    Autocallable,
    Barrier,
    BarrierType,
    European,
    Monitoring,
    OptionType,
)
from deskgraph.core.market import MarketInputs
from deskgraph.core.monte_carlo import MCConfig, MCResult

CALL, PUT = OptionType.CALL, OptionType.PUT
MKT = MarketInputs(spot=100.0, rate=0.05, div_yield=0.02, vol=0.25)
QUARTERLY = (0.25, 0.5, 0.75, 1.0)


def assert_within_3se(result: MCResult, target: float) -> None:
    gap = abs(result.price - target)
    assert gap <= 3 * result.std_error, (
        f"MC {result.price:.6f} vs {target:.6f}: gap {gap:.2e} > 3 SE ({3 * result.std_error:.2e})"
    )


# --- configuration -----------------------------------------------------------


def test_config_defaults() -> None:
    cfg = MCConfig(n_paths=1000, seed=1)
    assert cfg.antithetic is True
    assert cfg.control_variate is True
    assert cfg.steps_per_year == 252


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"n_paths": 0}, ValueError, "n_paths"),
        ({"n_paths": -10}, ValueError, "n_paths"),
        ({"n_paths": 1001}, ValueError, "even"),
        ({"n_paths": 2}, ValueError, "n_paths"),  # one antithetic pair: no standard error
        ({"n_paths": 1000.0}, TypeError, "n_paths"),
        ({"n_paths": True}, TypeError, "n_paths"),
        ({"seed": -1}, ValueError, "seed"),
        ({"seed": 1.5}, TypeError, "seed"),
        ({"steps_per_year": 0}, ValueError, "steps_per_year"),
    ],
)
def test_config_validation(kwargs: dict[str, object], error: type[Exception], match: str) -> None:
    args: dict[str, object] = {"n_paths": 1000, "seed": 1}
    args.update(kwargs)
    with pytest.raises(error, match=match):
        MCConfig(**args)  # type: ignore[arg-type]


def test_odd_path_count_is_fine_without_antithetic() -> None:
    assert MCConfig(n_paths=1001, seed=1, antithetic=False).n_paths == 1001


# --- Europeans against Black-Scholes -----------------------------------------


@pytest.mark.parametrize("option_type", [CALL, PUT])
@pytest.mark.parametrize(("strike", "expiry"), [(100.0, 1.0), (80.0, 0.5), (120.0, 2.0)])
def test_european_within_3se_of_black_scholes(
    option_type: OptionType, strike: float, expiry: float
) -> None:
    inst = European(option_type, strike=strike, expiry=expiry)
    result = mc.price(inst, MKT, MCConfig(n_paths=100_000, seed=11))
    assert_within_3se(result, bs.price(inst, MKT))


def test_plain_mc_without_variance_reduction_is_also_unbiased() -> None:
    inst = European(CALL, strike=105.0, expiry=1.0)
    cfg = MCConfig(n_paths=200_000, seed=7, antithetic=False, control_variate=False)
    assert_within_3se(mc.price(inst, MKT, cfg), bs.price(inst, MKT))


def test_same_seed_same_result_different_seed_different_result() -> None:
    inst = European(CALL, strike=100.0, expiry=1.0)
    a = mc.price(inst, MKT, MCConfig(n_paths=10_000, seed=3))
    b = mc.price(inst, MKT, MCConfig(n_paths=10_000, seed=3))
    c = mc.price(inst, MKT, MCConfig(n_paths=10_000, seed=4))
    assert a == b
    assert a.price != c.price


def test_result_echoes_paths_and_seed_for_lineage() -> None:
    result = mc.price(European(CALL, 100.0, 1.0), MKT, MCConfig(n_paths=1000, seed=99))
    assert (result.n_paths, result.seed) == (1000, 99)


def test_antithetic_standard_error_is_computed_on_pair_means() -> None:
    # Reproduce the engine by hand on 4 pairs: one standard_normal(n/2) draw, the
    # mirrored draws appended, and the standard error taken over the 4 pair means
    # (the two paths of a pair are not independent).
    inst = European(CALL, strike=100.0, expiry=1.0)
    cfg = MCConfig(n_paths=8, seed=2024, control_variate=False)
    z = np.random.default_rng(2024).standard_normal(4)
    t, s, r, q, v = 1.0, MKT.spot, MKT.rate, MKT.div_yield, MKT.vol

    def payoff(zz: np.ndarray) -> np.ndarray:
        st = s * np.exp((r - q - 0.5 * v * v) * t + v * math.sqrt(t) * zz)
        out: np.ndarray = math.exp(-r * t) * np.maximum(st - 100.0, 0.0)
        return out

    pairs = 0.5 * (payoff(z) + payoff(-z))
    result = mc.price(inst, MKT, cfg)
    assert result.price == pytest.approx(pairs.mean(), rel=1e-14)
    assert result.std_error == pytest.approx(pairs.std(ddof=1) / math.sqrt(4), rel=1e-12)


@pytest.mark.parametrize(("strike", "max_ratio"), [(80.0, 0.5), (100.0, 0.8)])
def test_control_variate_cuts_the_standard_error(strike: float, max_ratio: float) -> None:
    inst = European(CALL, strike=strike, expiry=1.0)
    plain = mc.price(inst, MKT, MCConfig(n_paths=50_000, seed=5, antithetic=False,
                                         control_variate=False))  # fmt: skip
    with_cv = mc.price(inst, MKT, MCConfig(n_paths=50_000, seed=5, antithetic=False))
    assert with_cv.std_error < max_ratio * plain.std_error


def test_antithetic_cuts_the_standard_error_for_a_monotone_payoff() -> None:
    inst = European(PUT, strike=100.0, expiry=1.0)
    plain = mc.price(inst, MKT, MCConfig(n_paths=50_000, seed=5, antithetic=False,
                                         control_variate=False))  # fmt: skip
    anti = mc.price(inst, MKT, MCConfig(n_paths=50_000, seed=5, control_variate=False))
    assert anti.std_error < plain.std_error


# --- barriers --------------------------------------------------------------------

BARRIER_CFG = MCConfig(n_paths=40_000, seed=21)
UNREACHABLE_DOWN = 1e-6  # a down barrier spot can't reach: the option is a vanilla


def barrier(option_type: OptionType, level: float, kind: BarrierType) -> Barrier:
    return Barrier(option_type, strike=100.0, expiry=1.0, barrier=level, barrier_type=kind)


@pytest.mark.parametrize("option_type", [CALL, PUT])
@pytest.mark.parametrize(
    ("level", "knock_in", "knock_out"),
    [
        (85.0, BarrierType.DOWN_IN, BarrierType.DOWN_OUT),
        (120.0, BarrierType.UP_IN, BarrierType.UP_OUT),
    ],
)
def test_knock_in_plus_knock_out_equals_vanilla_on_the_same_paths(
    option_type: OptionType, level: float, knock_in: BarrierType, knock_out: BarrierType
) -> None:
    ki = mc.price(barrier(option_type, level, knock_in), MKT, BARRIER_CFG)
    ko = mc.price(barrier(option_type, level, knock_out), MKT, BARRIER_CFG)
    vanilla = mc.price(barrier(option_type, UNREACHABLE_DOWN, BarrierType.DOWN_OUT), MKT,
                       BARRIER_CFG)  # fmt: skip
    assert ki.price + ko.price == pytest.approx(vanilla.price, abs=1e-12)
    assert 0.0 < ki.price < vanilla.price
    assert 0.0 < ko.price < vanilla.price


@pytest.mark.parametrize("option_type", [CALL, PUT])
def test_barrier_engine_prices_a_vanilla_within_3se_of_black_scholes(
    option_type: OptionType,
) -> None:
    result = mc.price(barrier(option_type, UNREACHABLE_DOWN, BarrierType.DOWN_OUT), MKT,
                      BARRIER_CFG)  # fmt: skip
    assert_within_3se(result, bs.price(European(option_type, 100.0, 1.0), MKT))


def test_barrier_already_breached_today() -> None:
    # Down barrier above spot: knocked in or out at t = 0, before any step.
    vanilla = mc.price(barrier(CALL, UNREACHABLE_DOWN, BarrierType.DOWN_OUT), MKT, BARRIER_CFG)
    ki = mc.price(barrier(CALL, 110.0, BarrierType.DOWN_IN), MKT, BARRIER_CFG)
    ko = mc.price(barrier(CALL, 110.0, BarrierType.DOWN_OUT), MKT, BARRIER_CFG)
    assert ki.price == vanilla.price
    assert ko.price == 0.0
    assert ko.std_error == 0.0


# --- autocallables -----------------------------------------------------------------

NOTE_CFG = MCConfig(n_paths=100_000, seed=31)


def note(**overrides: object) -> Autocallable:
    fields: dict[str, object] = {
        "expiry": 1.0,
        "observation_times": QUARTERLY,
        "autocall_level": 1.0,
        "coupon": 0.025,
        "knock_in_level": 0.6,
        "knock_in_monitoring": Monitoring.AT_MATURITY,
    }
    fields.update(overrides)
    return Autocallable(**fields)  # type: ignore[arg-type]


@pytest.mark.parametrize("monitoring", [Monitoring.AT_MATURITY, Monitoring.DAILY])
def test_autocall_level_zero_redeems_on_the_first_date(monitoring: Monitoring) -> None:
    result = mc.price(note(autocall_level=0.0, knock_in_level=0.0, knock_in_monitoring=monitoring),
                      MKT, NOTE_CFG)  # fmt: skip
    # Every path pays the same amount; only summation rounding (~1e-16) can differ.
    assert result.price == pytest.approx(1.025 * MKT.discount_factor(0.25), rel=1e-13)
    assert result.std_error < 1e-13


@pytest.mark.parametrize("monitoring", [Monitoring.AT_MATURITY, Monitoring.DAILY])
def test_never_called_with_knock_in_zero_repays_the_discounted_notional(
    monitoring: Monitoring,
) -> None:
    result = mc.price(note(autocall_level=1e9, knock_in_level=0.0, knock_in_monitoring=monitoring),
                      MKT, NOTE_CFG)  # fmt: skip
    assert result.price == pytest.approx(MKT.discount_factor(1.0), rel=1e-13)
    assert result.std_error < 1e-13


def test_coupons_accumulate_until_the_note_is_called() -> None:
    # At (almost) zero vol the path is the forward: X(t) = exp((r - q)·t), which is
    # 1.0075 at t=0.25 and 1.0151 at t=0.5. With an autocall level of 1.01 the note
    # is first called on the 2nd date and must pay two coupons (memory): (1 + 2c)·DF.
    mkt = MarketInputs(spot=100.0, rate=0.05, div_yield=0.02, vol=1e-9)
    result = mc.price(note(autocall_level=1.01, coupon=0.025), mkt, NOTE_CFG)
    assert result.price == pytest.approx(1.05 * mkt.discount_factor(0.5), rel=1e-9)


def single_observation_note_closed_form(
    mkt: MarketInputs, expiry: float, autocall: float, knock_in: float, coupon: float
) -> float:
    """Black-Scholes value of a note observed only at maturity, knock-in at maturity.

    Payoff per unit notional with X = S_T/S₀:  1 + c·1{X ≥ A} − (1 − X)·1{X < KI}.
    So value = DF·(1 + c·N(d₂(A·S₀))) − [DF·N(−d₂(KI·S₀)) − e^(−qT)·N(−d₁(KI·S₀))].
    """
    s, r, q, v, t = mkt.spot, mkt.rate, mkt.div_yield, mkt.vol, expiry

    def d1(k: float) -> float:
        return (math.log(s / k) + (r - q + 0.5 * v * v) * t) / (v * math.sqrt(t))

    def d2(k: float) -> float:
        return d1(k) - v * math.sqrt(t)

    df, dq = math.exp(-r * t), math.exp(-q * t)
    digital = df * (1.0 + coupon * float(ndtr(d2(autocall * s))))
    knock_in_put = df * float(ndtr(-d2(knock_in * s))) - dq * float(ndtr(-d1(knock_in * s)))
    return digital - knock_in_put


@pytest.mark.parametrize(
    ("autocall", "knock_in", "coupon", "expiry"),
    [(1.0, 0.6, 0.08, 1.0), (1.05, 0.7, 0.12, 2.0), (0.9, 0.5, 0.05, 0.5)],
)
def test_single_observation_note_within_3se_of_closed_form(
    autocall: float, knock_in: float, coupon: float, expiry: float
) -> None:
    inst = note(expiry=expiry, observation_times=(expiry,), autocall_level=autocall,
                knock_in_level=knock_in, coupon=coupon)  # fmt: skip
    expected = single_observation_note_closed_form(MKT, expiry, autocall, knock_in, coupon)
    assert_within_3se(mc.price(inst, MKT, NOTE_CFG), expected)


def test_daily_knock_in_is_worth_less_than_knock_in_at_maturity() -> None:
    # Daily monitoring knocks in on every path that at-maturity monitoring does, and more.
    mkt = MarketInputs(spot=100.0, rate=0.05, div_yield=0.02, vol=0.35)
    daily = mc.price(note(knock_in_level=0.7, knock_in_monitoring=Monitoring.DAILY), mkt, NOTE_CFG)
    at_maturity = mc.price(note(knock_in_level=0.7), mkt, NOTE_CFG)
    combined_se = math.hypot(daily.std_error, at_maturity.std_error)
    assert at_maturity.price - daily.price > 3 * combined_se


def test_higher_coupon_is_worth_more() -> None:
    low = mc.price(note(coupon=0.01), MKT, NOTE_CFG)
    high = mc.price(note(coupon=0.05), MKT, NOTE_CFG)
    assert high.price > low.price  # same seed and paths: only the coupon differs
