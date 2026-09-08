"""Fixed-horizon, volatility-scaled binary labels.

Target: P(forward N-day log return >= threshold), where threshold scales
with the instrument's own recent volatility rather than being a fixed %
(see docs/roadmap.md's methodology log for why: fixed thresholds make class
balance swing wildly across instruments with very different volatility).

threshold(t, N) = k * daily_vol(t) * sqrt(N)   [standard sqrt-time scaling]

Everything here operates on a single instrument's own row sequence — "N
trading days ahead" is `close.shift(-N)`, not a calendar-day offset, so it's
automatically correct regardless of holidays or gaps in that instrument's
calendar.
"""

import logging

import numpy as np
import pandas as pd

from app.data.paths import LABELS_DATA_DIR, PROCESSED_DATA_DIR, ticker_filename
from app.data.universe import UNIVERSE, Instrument
from app.features import technical as ta

logger = logging.getLogger(__name__)

HORIZONS = (5, 20)
DAILY_VOL_WINDOW = 20  # matches app.features.build's realized_vol_20d
DEFAULT_K = 0.5  # empirically selected — see docs/roadmap.md's methodology log


def forward_log_return(close: pd.Series, horizon: int) -> pd.Series:
    """The label-defining quantity. NaN for the last `horizon` rows of the
    series (no future data yet) — that's correct right-censoring, not a bug
    to be filled in."""
    return np.log(close.shift(-horizon) / close)


def volatility_threshold(daily_vol: pd.Series, horizon: int, k: float) -> pd.Series:
    """theta(t, N) = k * daily_vol(t) * sqrt(N). Uses the trailing (already
    no-lookahead) daily_vol estimate as of day t — the threshold itself
    never peeks at information not available at prediction time."""
    return k * daily_vol * np.sqrt(horizon)


def label_end_date(index: pd.DatetimeIndex, horizon: int) -> pd.Series:
    """The date each row's forward-return window actually ends on (i.e. the
    date of the close used as the label's numerator). Needed by Phase 5's
    walk-forward CV to purge training samples whose label window overlaps
    a test period — a label computed today is not usable for training until
    its own end date has passed. NaT for rows with no future data."""
    end_dates = pd.Series(pd.NaT, index=index, dtype="datetime64[ns]")
    if horizon < len(index):
        end_dates.iloc[: len(index) - horizon] = index[horizon:]
    return end_dates


def build_labels_for_instrument(
    close: pd.Series, daily_vol: pd.Series, k: float, horizons: tuple[int, ...] = HORIZONS
) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for n in horizons:
        fwd_ret = forward_log_return(close, n)
        threshold = volatility_threshold(daily_vol, n, k)
        columns[f"forward_return_{n}d"] = fwd_ret
        columns[f"label_threshold_{n}d"] = threshold
        columns[f"label_{n}d"] = (fwd_ret >= threshold).astype("Int64")
        # A label needs both the forward return AND the threshold to be
        # defined; mask out rows where either is NaN rather than letting a
        # NaN threshold silently compare as False.
        undefined = fwd_ret.isna() | threshold.isna()
        columns[f"label_{n}d"] = columns[f"label_{n}d"].mask(undefined, pd.NA)
        columns[f"label_end_date_{n}d"] = label_end_date(close.index, n)
    return pd.DataFrame(columns, index=close.index)


def build_labels_for_universe(instruments: list[Instrument] = UNIVERSE, k: float = DEFAULT_K) -> None:
    LABELS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for instrument in instruments:
        df = pd.read_parquet(PROCESSED_DATA_DIR / ticker_filename(instrument.ticker))
        close = df["adj_close"]
        daily_vol = ta.realized_volatility(close, DAILY_VOL_WINDOW)
        label_df = build_labels_for_instrument(close, daily_vol, k)
        label_df.to_parquet(LABELS_DATA_DIR / ticker_filename(instrument.ticker))

        rates = []
        for n in HORIZONS:
            valid = label_df[f"label_{n}d"].dropna()
            rate = valid.astype(bool).mean() if len(valid) else float("nan")
            rates.append(f"{n}d: {rate:.1%} of {len(valid)}")
        logger.info("%s: %s", instrument.ticker, "; ".join(rates))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_labels_for_universe()
