import dataclasses
import math
from itertools import accumulate

import pytest

from deskgraph.core.instruments import (
    Autocallable,
    Barrier,
    BarrierType,
    European,
    Monitoring,
    OptionType,
)

QUARTERLY = (0.25, 0.5, 0.75, 1.0)


def autocallable(**overrides: object) -> Autocallable:
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


# --- European -------------------------------------------------------------------


def test_european_valid() -> None:
    call = European(OptionType.CALL, strike=40, expiry=0.5)
    assert call.strike == 40.0
    assert type(call.strike) is float
    assert call.option_type is OptionType.CALL


@pytest.mark.parametrize(
    ("field", "value"),
    [("strike", 0.0), ("strike", -1.0), ("strike", math.nan), ("expiry", 0.0),
     ("expiry", -0.5), ("expiry", math.inf)],
)  # fmt: skip
def test_european_rejects_bad_numbers(field: str, value: float) -> None:
    fields: dict[str, object] = {"option_type": OptionType.PUT, "strike": 40.0, "expiry": 0.5}
    fields[field] = value
    with pytest.raises(ValueError, match=field):
        European(**fields)  # type: ignore[arg-type]


def test_european_rejects_option_type_given_as_string() -> None:
    with pytest.raises(TypeError, match="option_type"):
        European("call", strike=40.0, expiry=0.5)  # type: ignore[arg-type]


def test_instruments_are_frozen_hashable_and_equal_by_value() -> None:
    a = European(OptionType.CALL, strike=40.0, expiry=0.5)
    b = European(OptionType.CALL, strike=40, expiry=0.5)
    assert a == b
    assert hash(a) == hash(b)  # usable as cache keys in the dependency graph
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.strike = 41.0  # type: ignore[misc]


# --- Barrier --------------------------------------------------------------------


def test_barrier_valid() -> None:
    b = Barrier(OptionType.CALL, strike=100.0, expiry=1.0, barrier=80.0,
                barrier_type=BarrierType.DOWN_OUT)  # fmt: skip
    assert b.barrier == 80.0
    assert b.barrier_type is BarrierType.DOWN_OUT


@pytest.mark.parametrize("value", [0.0, -5.0, math.nan])
def test_barrier_rejects_bad_level(value: float) -> None:
    with pytest.raises(ValueError, match="barrier"):
        Barrier(OptionType.CALL, strike=100.0, expiry=1.0, barrier=value,
                barrier_type=BarrierType.DOWN_IN)  # fmt: skip


def test_barrier_rejects_barrier_type_given_as_string() -> None:
    with pytest.raises(TypeError, match="barrier_type"):
        Barrier(OptionType.CALL, strike=100.0, expiry=1.0, barrier=80.0,
                barrier_type="down_in")  # type: ignore[arg-type]  # fmt: skip


@pytest.mark.parametrize(
    ("barrier_type", "is_down", "is_knock_in"),
    [
        (BarrierType.DOWN_IN, True, True),
        (BarrierType.DOWN_OUT, True, False),
        (BarrierType.UP_IN, False, True),
        (BarrierType.UP_OUT, False, False),
    ],
)
def test_barrier_type_properties(
    barrier_type: BarrierType, is_down: bool, is_knock_in: bool
) -> None:
    assert barrier_type.is_down is is_down
    assert barrier_type.is_knock_in is is_knock_in


# --- Autocallable ---------------------------------------------------------------


def test_autocallable_valid() -> None:
    note = autocallable()
    assert note.observation_times == QUARTERLY
    assert note.knock_in_monitoring is Monitoring.AT_MATURITY


def test_observation_times_are_stored_as_an_immutable_tuple_of_floats() -> None:
    times = [0.5, 1]
    note = autocallable(observation_times=times)
    times.append(2.0)  # mutating the caller's list must not reach the instrument
    assert note.observation_times == (0.5, 1.0)
    assert all(type(t) is float for t in note.observation_times)


def test_last_observation_within_tolerance_is_snapped_to_expiry() -> None:
    # Adding 0.1 step by step drifts below 1.0 (sum() on 3.12+ compensates, so avoid it).
    tenths = tuple(accumulate([0.1] * 10))
    assert tenths[-1] != 1.0
    note = autocallable(observation_times=tenths)
    assert note.observation_times[-1] == 1.0


@pytest.mark.parametrize(
    ("times", "message"),
    [
        ((), "observation_times"),
        ((0.5, 0.25, 1.0), "increasing"),
        ((0.5, 0.5, 1.0), "increasing"),
        ((0.0, 0.5, 1.0), "observation_times"),
        ((0.25, 0.5, 0.75), "expiry"),
        ((0.5, 1.0, 1.5), "expiry"),
        ((0.5, math.nan, 1.0), "observation_times"),
    ],
)
def test_bad_observation_times_are_rejected(times: tuple[float, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        autocallable(observation_times=times)


@pytest.mark.parametrize(
    ("field", "value"),
    [("autocall_level", -0.1), ("coupon", -0.01), ("knock_in_level", -0.1),
     ("autocall_level", math.inf), ("expiry", 0.0)],
)  # fmt: skip
def test_bad_levels_are_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        autocallable(**{field: value})


def test_knock_in_above_autocall_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="knock_in_level"):
        autocallable(autocall_level=0.9, knock_in_level=0.95)


def test_spec_edge_cases_are_constructible() -> None:
    # SPEC component 2: autocall level 0 (called on the first date) and a note that
    # is never called with knock-in 0. Core only checks the payoff is well defined;
    # the desk rule "knock-in strictly below autocall" belongs to RFQ validation.
    always_called = autocallable(autocall_level=0.0, knock_in_level=0.0)
    never_called = autocallable(autocall_level=1e9, knock_in_level=0.0)
    assert always_called.autocall_level == 0.0
    assert never_called.knock_in_level == 0.0


def test_knock_in_monitoring_must_be_the_enum() -> None:
    with pytest.raises(TypeError, match="knock_in_monitoring"):
        autocallable(knock_in_monitoring="daily")
