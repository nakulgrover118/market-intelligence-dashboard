"""Assembles the per-instrument feature matrix from cleaned OHLCV data.

Every feature here is trailing-only (see app/features/technical.py). The
first ~200 rows of the output will have NaNs from warm-up windows (the
slowest is the 200-day SMA) — that's expected and correct: there is no
valid 200-day trend feature until 200 days have actually happened. Rows
with NaN features are dropped at the modeling stage (Phase 4/5), not here,
so this module's output always reflects the true availability of each
feature on each date.
"""

import logging

import pandas as pd

from app.data.paths import FEATURES_DATA_DIR, PROCESSED_DATA_DIR, ticker_filename
from app.data.universe import UNIVERSE, Instrument
from app.features import technical as ta

logger = logging.getLogger(__name__)

SMA_WINDOWS = (10, 20, 50, 200)
RETURN_WINDOWS = (1, 5, 10, 20)
VOL_WINDOWS = (20, 60)


def build_features_for_instrument(df: pd.DataFrame) -> pd.DataFrame:
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    features: dict[str, pd.Series] = {}

    for w in RETURN_WINDOWS:
        features[f"log_return_{w}d"] = ta.log_return(close, w)

    for w in SMA_WINDOWS:
        sma_w = ta.sma(close, w)
        features[f"sma_{w}"] = sma_w
        features[f"close_over_sma_{w}"] = close / sma_w - 1

    features["ema_12"] = ta.ema(close, 12)
    features["ema_26"] = ta.ema(close, 26)

    for w in VOL_WINDOWS:
        features[f"realized_vol_{w}d"] = ta.realized_volatility(close, w)

    features["rsi_14"] = ta.rsi(close, window=14)
    features["roc_10"] = ta.rate_of_change(close, window=10)
    features["atr_14"] = ta.average_true_range(high, low, close, window=14)
    features["volume_ratio_20"] = ta.volume_ratio(volume, window=20)

    macd_df = ta.macd(close)
    bb_df = ta.bollinger_bands(close)

    feature_df = pd.concat([pd.DataFrame(features), macd_df, bb_df], axis=1)
    feature_df.index = df.index
    return feature_df


def build_features_for_universe(instruments: list[Instrument] = UNIVERSE) -> None:
    FEATURES_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for instrument in instruments:
        in_path = PROCESSED_DATA_DIR / ticker_filename(instrument.ticker)
        df = pd.read_parquet(in_path)
        feature_df = build_features_for_instrument(df)
        out_path = FEATURES_DATA_DIR / ticker_filename(instrument.ticker)
        feature_df.to_parquet(out_path)
        n_valid = feature_df.dropna().shape[0]
        logger.info(
            "%s: %d feature rows (%d fully warmed up, %.0f%%)",
            instrument.ticker,
            len(feature_df),
            n_valid,
            100 * n_valid / len(feature_df),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_features_for_universe()
