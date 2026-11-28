"""Pricing speed and accuracy, written to benchmarks/results.json.

Run: ``uv run pytest benchmarks --benchmark-only``. Timings depend on the
machine (recorded under "environment" in results.json); accuracy figures don't.
"""

from __future__ import annotations

import math
import statistics
import time

import numpy as np
from pytest_benchmark.fixture import BenchmarkFixture

from benchmarks._support import Results, timed
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

MKT = MarketInputs(spot=100.0, rate=0.05, div_yield=0.02, vol=0.25)
NOTE = Autocallable(
    expiry=1.0,
    observation_times=(0.25, 0.5, 0.75, 1.0),
    autocall_level=1.0,
    coupon=0.025,
    knock_in_level=0.6,
    knock_in_monitoring=Monitoring.DAILY,
)


def test_mc_european_one_million_paths(benchmark: BenchmarkFixture, results: Results) -> None:
    inst = European(OptionType.CALL, 100.0, 1.0)
    cfg = MCConfig(n_paths=1_000_000, seed=1)
    result, seconds = timed(benchmark, lambda: mc.price(inst, MKT, cfg))
    exact = bs.price(inst, MKT)
    z = (result.price - exact) / result.std_error
    results["mc_european_1m_paths"] = {
        "seed": cfg.seed,
        "seconds": seconds,
        "paths_per_second": cfg.n_paths / seconds,
        "price": result.price,
        "std_error": result.std_error,
        "black_scholes": exact,
        "z": z,
    }
    assert abs(z) <= 3


def test_mc_standard_error_is_calibrated_across_seeds(
    benchmark: BenchmarkFixture, results: Results
) -> None:
    # If the estimator is unbiased and its standard error honest, z over many
    # seeds is ~N(0, 1): mean within 3/√n of 0, sd within about 3/√(2(n−1)) of 1.
    inst = European(OptionType.CALL, 100.0, 1.0)
    exact = bs.price(inst, MKT)
    seeds = range(30)

    def z_scores() -> list[float]:
        out = []
        for seed in seeds:
            r = mc.price(inst, MKT, MCConfig(n_paths=1_000_000, seed=seed))
            out.append((r.price - exact) / r.std_error)
        return out

    zs, _ = timed(benchmark, z_scores, rounds=1)
    mean_z, sd_z = statistics.mean(zs), statistics.stdev(zs)
    results["mc_european_seed_sweep"] = {
        "seeds": len(seeds),
        "n_paths": 1_000_000,
        "mean_z": mean_z,
        "sd_z": sd_z,
        "max_abs_z": max(abs(z) for z in zs),
    }
    assert abs(mean_z) < 3 / math.sqrt(len(seeds))
    assert abs(sd_z - 1) < 3 / math.sqrt(2 * (len(seeds) - 1))


def test_mc_accuracy_against_black_scholes(benchmark: BenchmarkFixture, results: Results) -> None:
    cases = [
        European(option_type, strike, expiry)
        for option_type in OptionType
        for strike, expiry in [(100.0, 1.0), (80.0, 0.5), (120.0, 2.0)]
    ]
    cfg = MCConfig(n_paths=100_000, seed=11)

    def z_scores() -> list[float]:
        out = []
        for inst in cases:
            r = mc.price(inst, MKT, cfg)
            out.append((r.price - bs.price(inst, MKT)) / r.std_error)
        return out

    zs, _ = timed(benchmark, z_scores, rounds=1)
    max_abs_z = max(abs(z) for z in zs)
    results["mc_vs_black_scholes"] = {
        "cases": len(cases),
        "n_paths": cfg.n_paths,
        "z_scores": zs,
        "max_abs_z": max_abs_z,
        "within_3se": max_abs_z <= 3,
    }
    assert max_abs_z <= 3


def test_black_scholes_vectorised_vs_python_loop(
    benchmark: BenchmarkFixture, results: Results
) -> None:
    n = 10_000
    rng = np.random.default_rng(0)
    is_call = rng.random(n) < 0.5
    strikes = MKT.spot * rng.uniform(0.5, 2.0, n)
    expiries = rng.uniform(0.05, 3.0, n)
    book = [
        European(OptionType.CALL if c else OptionType.PUT, float(k), float(t))
        for c, k, t in zip(is_call, strikes, expiries, strict=True)
    ]

    def vectorised() -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return bs.price_vec(is_call, MKT.spot, strikes, expiries, MKT.rate, MKT.div_yield, MKT.vol)

    vec_prices, vec_seconds = timed(benchmark, vectorised, rounds=20)

    loop_times = []
    for _ in range(3):
        start = time.perf_counter()
        loop_prices = [bs.price(inst, MKT) for inst in book]
        loop_times.append(time.perf_counter() - start)
    loop_seconds = statistics.median(loop_times)

    max_diff = float(np.max(np.abs(vec_prices - np.array(loop_prices))))
    results["black_scholes_vectorised_vs_loop"] = {
        "n_options": n,
        "loop_seconds": loop_seconds,
        "vectorised_seconds": vec_seconds,
        "speedup": loop_seconds / vec_seconds,
        "max_abs_price_difference": max_diff,
    }
    assert max_diff < 1e-10


def test_mc_barrier_daily_monitoring(benchmark: BenchmarkFixture, results: Results) -> None:
    inst = Barrier(OptionType.CALL, 100.0, 1.0, 85.0, BarrierType.DOWN_OUT)
    cfg = MCConfig(n_paths=100_000, seed=2)
    result, seconds = timed(benchmark, lambda: mc.price(inst, MKT, cfg), rounds=3)
    results["mc_barrier_daily_100k_paths_1y"] = {
        "seconds": seconds,
        "path_steps_per_second": cfg.n_paths * 252 / seconds,
        "price": result.price,
        "std_error": result.std_error,
    }


def test_mc_autocallable_daily_knock_in(benchmark: BenchmarkFixture, results: Results) -> None:
    cfg = MCConfig(n_paths=100_000, seed=3)
    result, seconds = timed(benchmark, lambda: mc.price(NOTE, MKT, cfg), rounds=3)
    results["mc_autocallable_daily_ki_100k_paths_1y"] = {
        "seconds": seconds,
        "price": result.price,
        "std_error": result.std_error,
    }


def test_bump_greeks_autocallable(benchmark: BenchmarkFixture, results: Results) -> None:
    cfg = MCConfig(n_paths=100_000, seed=4)
    result, seconds = timed(benchmark, lambda: bump_greeks(NOTE, MKT, cfg), rounds=3)
    results["bump_greeks_autocallable_100k_paths"] = {
        "seconds": seconds,
        "revaluations": 8,
        "delta": result.value.delta,
        "delta_std_error": result.std_error.delta,
        "vega": result.value.vega,
        "vega_std_error": result.std_error.vega,
    }
