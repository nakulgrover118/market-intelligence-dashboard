"""Fetches raw daily OHLCV history for the pilot universe and writes it,
untouched, to Parquet.

This module deliberately does no cleaning, filling, or adjustment beyond
what the data vendor (Yahoo Finance, via yfinance) already applies. Keeping
a raw, immutable layer means every later stage (features, labels, models)
can be rebuilt from here without re-hitting the network, and any bug found
downstream can be traced back to a known-good starting point.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf

from app.data.universe import UNIVERSE, Instrument

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"

# Columns we expect back from yfinance for a daily-bar history request.
EXPECTED_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


@dataclass
class IngestResult:
    ticker: str
    rows: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    warnings: list[str]
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def fetch_ticker_history(
    ticker: str, start: str = "2005-01-01", max_retries: int = 3, retry_delay_s: float = 2.0
) -> pd.DataFrame:
    """Fetch full daily OHLCV history for one ticker, retrying on transient failures."""
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.Ticker(ticker).history(start=start, interval="1d", auto_adjust=False)
            if df.empty:
                raise ValueError(f"yfinance returned no rows for {ticker}")
            return df
        except Exception as exc:  # yfinance can raise a variety of network/HTTP errors
            last_error = exc
            logger.warning("Fetch attempt %d/%d failed for %s: %s", attempt, max_retries, ticker, exc)
            if attempt < max_retries:
                time.sleep(retry_delay_s)
    raise RuntimeError(f"Failed to fetch {ticker} after {max_retries} attempts") from last_error


def _validate(ticker: str, df: pd.DataFrame) -> list[str]:
    """Flag data quality issues without silently fixing them."""
    warnings: list[str] = []

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        warnings.append(f"missing expected columns: {missing_cols}")

    if df.index.duplicated().any():
        warnings.append(f"{df.index.duplicated().sum()} duplicate date rows")

    if not df.index.is_monotonic_increasing:
        warnings.append("date index is not monotonically increasing")

    price_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close"] if c in df.columns]
    if price_cols and (df[price_cols] <= 0).any().any():
        warnings.append("non-positive price values present")

    if "Volume" in df.columns and (df["Volume"] < 0).any():
        warnings.append("negative volume values present")

    return warnings


def ingest_instrument(instrument: Instrument, start: str = "2005-01-01") -> IngestResult:
    try:
        df = fetch_ticker_history(instrument.ticker, start=start)
    except Exception as exc:
        return IngestResult(instrument.ticker, rows=0, start=None, end=None, warnings=[], error=str(exc))

    warnings = _validate(instrument.ticker, df)
    for w in warnings:
        logger.warning("%s: %s", instrument.ticker, w)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DATA_DIR / f"{instrument.ticker.replace('^', 'IDX_')}.parquet"
    df.to_parquet(out_path)

    return IngestResult(
        ticker=instrument.ticker,
        rows=len(df),
        start=df.index.min(),
        end=df.index.max(),
        warnings=warnings,
    )


def ingest_universe(
    instruments: list[Instrument] = UNIVERSE, start: str = "2005-01-01", delay_s: float = 0.5
) -> list[IngestResult]:
    results = []
    for i, instrument in enumerate(instruments):
        result = ingest_instrument(instrument, start=start)
        results.append(result)
        if i < len(instruments) - 1:
            time.sleep(delay_s)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = ingest_universe()

    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]

    print(f"\nIngested {len(ok)}/{len(results)} instruments successfully.")
    for r in ok:
        flag = f" [{len(r.warnings)} warning(s)]" if r.warnings else ""
        print(f"  {r.ticker:16s} {r.rows:6d} rows  {r.start.date()} -> {r.end.date()}{flag}")

    if failed:
        print(f"\n{len(failed)} instrument(s) FAILED:")
        for r in failed:
            print(f"  {r.ticker:16s} {r.error}")
