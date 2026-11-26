# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "yfinance>=0.2.50",
#   "pandas>=2.2",
#   "pyarrow>=17",
#   "tzdata",
# ]
# ///
"""Record one daily snapshot of a listed option chain from Yahoo Finance.

Why this exists: yfinance only serves the *current* chain, so the history that
P&L explain needs (spec component 5) has to be recorded day by day, starting
now. The script deliberately lives outside ``src/deskgraph``: it does network
and file I/O, and its output is raw, uncleaned data. The snapshot loader
(spec component 1) will clean it and content-hash it later.

Run it with ``uv run scripts/capture_chain.py`` (uv reads the dependency block
above, so the project environment is not touched).

Output, one folder per ticker and trading date, never overwritten::

    data/raw/chains/<TICKER>/<YYYY-MM-DD>/chain.parquet   all expiries, calls and puts
    data/raw/chains/<TICKER>/<YYYY-MM-DD>/meta.json       spot, timestamps, row counts, sha256

The trading date is the date of the latest daily bar Yahoo returns, so running
the script on a weekend or holiday records the last session once and then
skips. Spot is read just before the chain, so the two are as close in time as
the delayed feed allows; it is *not* the official close.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "chains"
REQUIRED_COLUMNS = {"contractSymbol", "strike", "bid", "ask", "lastPrice", "lastTradeDate"}
log = logging.getLogger("capture_chain")


def with_retries[T](fn: Any, what: str, attempts: int = 3) -> T:
    """Call ``fn()``; on failure wait 2s, 4s, ... and try again."""
    for i in range(1, attempts + 1):
        try:
            result: T = fn()
            return result
        except Exception as exc:  # yfinance raises many unrelated types
            if i == attempts:
                raise
            wait = 2**i
            log.warning("%s failed (%s), retry %d/%d in %ds", what, exc, i, attempts - 1, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


def last_trading_date_and_close(ticker: yf.Ticker) -> tuple[date, float | None]:
    """Date and close of the latest daily bar.

    Today's bar is partial intraday, and Yahoo sometimes publishes a finished
    session's bar with volume but a NaN open/close for a while; the date is
    still right, so the close is returned as None rather than dropping the bar.
    """
    hist: pd.DataFrame = with_retries(lambda: ticker.history(period="5d", interval="1d"), "history")
    if hist.empty:
        raise RuntimeError("no price history returned")
    last = hist.index[-1]
    close = hist["Close"].iloc[-1]
    return last.date(), None if pd.isna(close) else float(close)


def fetch_chain(ticker: yf.Ticker, pause_s: float) -> tuple[pd.DataFrame, list[str]]:
    """All expiries stacked into one frame with ``expiry`` and ``option_type`` columns."""
    expiries: list[str] = list(with_retries(lambda: ticker.options, "expiry list"))
    if not expiries:
        raise RuntimeError("no option expiries returned")
    frames: list[pd.DataFrame] = []
    for exp in expiries:
        chain = with_retries(lambda e=exp: ticker.option_chain(e), f"chain {exp}")
        for side, df in (("C", chain.calls), ("P", chain.puts)):
            if df.empty:
                continue
            df = df.copy()
            df["expiry"] = pd.Timestamp(exp).date()
            df["option_type"] = side
            frames.append(df)
        time.sleep(pause_s)  # be polite to an unofficial endpoint
    if not frames:
        raise RuntimeError("every expiry came back empty")
    out = pd.concat(frames, ignore_index=True)
    missing = REQUIRED_COLUMNS - set(out.columns)
    if missing:
        raise RuntimeError(f"chain is missing columns: {sorted(missing)}")
    return out, expiries


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(symbol: str, out_root: Path, pause_s: float) -> Path | None:
    """Capture one ticker; return the folder written, or None if already captured."""
    ticker = yf.Ticker(symbol)
    trade_date, bar_close = last_trading_date_and_close(ticker)
    final_dir = out_root / symbol / trade_date.isoformat()
    if final_dir.exists():
        log.info("%s %s already captured, skipping", symbol, trade_date)
        return None

    captured_at = datetime.now(UTC)
    spot_last = float(with_retries(lambda: ticker.fast_info["lastPrice"], "spot"))
    chain, expiries = fetch_chain(ticker, pause_s)
    chain_done_at = datetime.now(UTC)

    # Write into a temp folder and rename at the end, so a crash never leaves a
    # half-written snapshot that a later run would mistake for a complete one.
    tmp_dir = out_root / symbol / f".tmp-{trade_date.isoformat()}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    parquet_path = tmp_dir / "chain.parquet"
    chain.to_parquet(parquet_path, index=False)

    meta = {
        "ticker": symbol,
        "trade_date": trade_date.isoformat(),
        "captured_at_utc": captured_at.isoformat(),
        "chain_done_at_utc": chain_done_at.isoformat(),
        "spot_last": spot_last,
        "daily_bar_close": bar_close,
        "n_expiries": len(expiries),
        "n_rows": len(chain),
        "n_calls": int((chain["option_type"] == "C").sum()),
        "n_puts": int((chain["option_type"] == "P").sum()),
        "columns": list(chain.columns),
        "chain_sha256": sha256_of(parquet_path),
        "source": "Yahoo Finance via yfinance (unofficial, delayed)",
        "yfinance_version": yf.__version__,
    }
    (tmp_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    tmp_dir.rename(final_dir)
    log.info(
        "%s %s: %d rows over %d expiries, spot %.2f -> %s",
        symbol,
        trade_date,
        len(chain),
        len(expiries),
        spot_last,
        final_dir,
    )
    return final_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tickers", nargs="*", default=["SPY"], help="default: SPY")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"default: {DEFAULT_OUT}")
    parser.add_argument("--pause", type=float, default=0.3, help="seconds between expiry requests")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.out / "capture.log", encoding="utf-8"),
        ],
    )
    failures = 0
    for symbol in args.tickers:
        try:
            capture(symbol.upper(), args.out, args.pause)
        except Exception:
            failures += 1
            log.exception("%s capture failed", symbol)
    return 1 if failures else 0  # non-zero so Task Scheduler shows the failure


if __name__ == "__main__":
    raise SystemExit(main())
