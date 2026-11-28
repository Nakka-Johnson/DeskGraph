import dataclasses
import hashlib
import json
import math
import re

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from deskgraph.core.market import MarketInputs


def make(**overrides: object) -> MarketInputs:
    fields: dict[str, object] = {"spot": 42.0, "rate": 0.10, "div_yield": 0.0, "vol": 0.20}
    fields.update(overrides)
    return MarketInputs(**fields)  # type: ignore[arg-type]


# --- construction and validation ---------------------------------------------


def test_valid_inputs_are_stored_as_floats() -> None:
    mkt = make(spot=42, rate=np.float64(0.1))
    assert mkt.spot == 42.0
    assert type(mkt.spot) is float
    assert type(mkt.rate) is float


def test_negative_rate_and_dividend_yield_are_allowed() -> None:
    mkt = make(rate=-0.005, div_yield=-0.01)
    assert mkt.rate == -0.005
    assert mkt.div_yield == -0.01


@pytest.mark.parametrize(
    ("field", "value"),
    [("spot", 0.0), ("spot", -1.0), ("vol", 0.0), ("vol", -0.2)],
)
def test_non_positive_spot_or_vol_is_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        make(**{field: value})


@pytest.mark.parametrize("field", ["spot", "rate", "div_yield", "vol"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_values_are_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        make(**{field: value})


@pytest.mark.parametrize("value", [True, "42", None])
def test_non_numbers_are_rejected(value: object) -> None:
    with pytest.raises(TypeError, match="spot"):
        make(spot=value)


def test_market_inputs_are_frozen() -> None:
    mkt = make()
    with pytest.raises(dataclasses.FrozenInstanceError):
        mkt.spot = 50.0  # type: ignore[misc]


# --- discounting and forwards ------------------------------------------------


def test_discount_factor_is_continuous_compounding() -> None:
    mkt = make(rate=0.10)
    assert mkt.discount_factor(0.0) == 1.0
    assert mkt.discount_factor(0.5) == pytest.approx(math.exp(-0.05), rel=1e-15)


def test_forward_includes_dividend_yield() -> None:
    mkt = make(spot=100.0, rate=0.05, div_yield=0.02)
    assert mkt.forward(2.0) == pytest.approx(100.0 * math.exp(0.03 * 2.0), rel=1e-15)


@pytest.mark.parametrize("t", [-0.1, math.nan, math.inf])
def test_discount_factor_rejects_bad_times(t: float) -> None:
    with pytest.raises(ValueError, match="t"):
        make().discount_factor(t)


# --- snapshot_id ---------------------------------------------------------------


def test_snapshot_id_is_sha256_of_the_documented_canonical_form() -> None:
    # This pins the canonical form. Changing it changes every stored snapshot_id and
    # breaks reproduction of past results: bump "version" instead of editing in place.
    canonical = json.dumps(
        {"kind": "MarketInputs", "version": 1, "spot": "42.0", "rate": "0.1",
         "div_yield": "0.0", "vol": "0.2"},
        sort_keys=True,
        separators=(",", ":"),
    )  # fmt: skip
    expected = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert make().snapshot_id == expected


def test_snapshot_id_format() -> None:
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", make().snapshot_id)


def test_equal_inputs_give_equal_ids_regardless_of_how_they_were_written() -> None:
    assert make(spot=42).snapshot_id == make(spot=42.0).snapshot_id
    assert make(div_yield=-0.0).snapshot_id == make(div_yield=0.0).snapshot_id


@pytest.mark.parametrize("field", ["spot", "rate", "div_yield", "vol"])
def test_changing_any_field_changes_the_id(field: str) -> None:
    base = make()
    nudged = make(**{field: math.nextafter(getattr(base, field), math.inf)})
    assert nudged.snapshot_id != base.snapshot_id


@given(
    spot=st.floats(min_value=1e-6, max_value=1e6),
    rate=st.floats(min_value=-0.1, max_value=0.5),
    div_yield=st.floats(min_value=-0.1, max_value=0.5),
    vol=st.floats(min_value=1e-4, max_value=5.0),
)
def test_snapshot_id_is_deterministic(
    spot: float, rate: float, div_yield: float, vol: float
) -> None:
    a = MarketInputs(spot=spot, rate=rate, div_yield=div_yield, vol=vol)
    b = MarketInputs(spot=spot, rate=rate, div_yield=div_yield, vol=vol)
    assert a.snapshot_id == b.snapshot_id
    assert a == b
    assert hash(a) == hash(b)
