"""Raw -> processed data cleaning.

This is the one place we're allowed to *change* values rather than just flag
them (contrast with `ingest.py`, which never touches the raw layer). Every
transformation here is a deliberate, documented decision, not a silent fix.
"""

import logging
from dataclasses import dataclass

import pandas as pd

from app.data.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR, ticker_filename
from app.data.universe import UNIVERSE, Instrument

logger = logging.getLogger(__name__)

_RENAME = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
    "Dividends": "dividends",
    "Stock Splits": "stock_splits",
}

# Columns we actually consume downstream. `open` is deliberately excluded:
# it found real vendor data (GOLDBEES.NS has Open == 0 for ~247 scattered
# rows in its first ~13 months) that doesn't affect any feature we compute,
# since none of our indicators (returns, SMA/EMA, RSI, MACD, ATR, Bollinger,
# volume ratio) use raw Open. Truncating or dropping rows over a field we
# never read would destroy real Close/Volume signal for no benefit. If a
# future feature needs Open (e.g. an overnight-gap feature), this column
# list must be revisited.
_CORE_PRICE_COLS = ["high", "low", "close", "adj_close"]


@dataclass
class CleanResult:
    ticker: str
    rows_in: int
    rows_out: int
    rows_dropped_leading_unusable: int
    rows_dropped_duplicate: int


def _to_snake_case_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=_RENAME)


def _drop_tz(df: pd.DataFrame) -> pd.DataFrame:
    """Instruments can have different exchange timezones; normalize to naive
    dates so joins across instruments (Phase 2c) line up on calendar date."""
    df = df.copy()
    df.index = df.index.tz_localize(None)
    df.index.name = "date"
    return df


def _drop_duplicate_dates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    n_before = len(df)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df, n_before - len(df)


def _leading_unusable_mask(df: pd.DataFrame) -> pd.Series:
    """A row is unusable if it has a non-positive core price, or zero
    volume. Zero volume alone is the signature of vendor placeholder data,
    not real market activity: NESTLEIND.NS in this dataset has a completely
    flat price and zero volume for its first ~5 years (2005-2010, 1238
    rows) — no real instrument trades at an unchanged price for five years
    straight. A real, occasional zero-volume day is expected to occur later
    in a series and is handled as a warning, not a drop (see below)."""
    present_price_cols = [c for c in _CORE_PRICE_COLS if c in df.columns]
    bad_price = (df[present_price_cols] <= 0).any(axis=1)
    bad_volume = (df["volume"] <= 0) if "volume" in df.columns else False
    return bad_price | bad_volume


def _truncate_leading_unusable_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop leading rows that don't represent real trading activity (see
    `_leading_unusable_mask`). We only truncate from the *start* of the
    series, not anywhere in the middle: a bad row in the middle of an
    otherwise-trading series is a different kind of problem (a vendor data
    error) that we want to keep visible, not silently drop, since dropping
    rows out of the middle of a time series creates a gap that a naive
    .shift()-based feature would silently paper over."""
    is_bad = _leading_unusable_mask(df)
    if not is_bad.any():
        return df, 0
    first_good = is_bad[~is_bad].index.min()
    if pd.isna(first_good):
        raise ValueError("every row is unusable (bad price or zero volume); refusing to clean")
    n_dropped = int((df.index < first_good).sum())
    return df.loc[df.index >= first_good], n_dropped


def clean_instrument(instrument: Instrument) -> CleanResult:
    raw_path = RAW_DATA_DIR / ticker_filename(instrument.ticker)
    df = pd.read_parquet(raw_path)
    rows_in = len(df)

    df = _to_snake_case_columns(df)
    df = _drop_tz(df)
    df, n_dup = _drop_duplicate_dates(df)
    df, n_leading = _truncate_leading_unusable_rows(df)

    remaining_bad = _leading_unusable_mask(df)
    if remaining_bad.any():
        logger.warning(
            "%s: %d non-leading rows still have a non-positive core price or zero volume "
            "after cleaning (left as-is, not dropped, since it's mid-series)",
            instrument.ticker,
            int(remaining_bad.sum()),
        )

    if "open" in df.columns:
        bad_open = (df["open"] <= 0).sum()
        if bad_open:
            logger.warning(
                "%s: %d rows have Open <= 0 (not used by any current feature, left as-is)",
                instrument.ticker,
                int(bad_open),
            )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / ticker_filename(instrument.ticker)
    df.to_parquet(out_path)

    return CleanResult(
        ticker=instrument.ticker,
        rows_in=rows_in,
        rows_out=len(df),
        rows_dropped_leading_unusable=n_leading,
        rows_dropped_duplicate=n_dup,
    )


def clean_universe(instruments: list[Instrument] = UNIVERSE) -> list[CleanResult]:
    return [clean_instrument(instrument) for instrument in instruments]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = clean_universe()
    print(f"\nCleaned {len(results)} instruments.")
    for r in results:
        note = []
        if r.rows_dropped_leading_unusable:
            note.append(f"dropped {r.rows_dropped_leading_unusable} leading unusable rows")
        if r.rows_dropped_duplicate:
            note.append(f"dropped {r.rows_dropped_duplicate} duplicate-date rows")
        note_str = f" ({'; '.join(note)})" if note else ""
        print(f"  {r.ticker:16s} {r.rows_in:6d} -> {r.rows_out:6d} rows{note_str}")
