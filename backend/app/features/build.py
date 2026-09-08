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
from app.features import cross_asset as xa
from app.features import technical as ta

logger = logging.getLogger(__name__)

SMA_WINDOWS = (10, 20, 50, 200)
RETURN_WINDOWS = (1, 5, 10, 20)
VOL_WINDOWS = (20, 60)
BETA_WINDOWS = (20, 60)
EXCESS_RETURN_WINDOWS = (5, 20)
MACRO_RETURN_WINDOWS = (5, 20)

REFERENCE_INDEX_TICKER = "^NSEI"
GOLD_TICKER = "GOLDBEES.NS"
SILVER_TICKER = "SILVERBEES.NS"


def build_features_for_instrument(
    df: pd.DataFrame,
    index_close: pd.Series | None = None,
    gold_close: pd.Series | None = None,
    silver_close: pd.Series | None = None,
) -> pd.DataFrame:
    """`index_close`/`gold_close`/`silver_close` are the *other* instruments'
    cleaned adj_close series, used to compute cross-asset features (Phase 2c).
    Pass None for whichever reference doesn't apply — e.g. the Nifty
    instrument itself doesn't get a beta-vs-itself feature (see
    build_features_for_universe, which decides this per-ticker).

    Uses adj_close/adj_high/adj_low (split & dividend back-adjusted, derived
    in clean.py), not raw close/high/low — using unadjusted prices would
    inject a fake return spike at every split/dividend date into every
    price-based feature."""
    close, high, low, volume = df["adj_close"], df["adj_high"], df["adj_low"], df["volume"]
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

    if index_close is not None:
        for w in BETA_WINDOWS:
            features[f"beta_{w}d_vs_nifty"] = xa.rolling_beta(close, index_close, w).reindex(df.index)
            features[f"corr_{w}d_vs_nifty"] = xa.rolling_correlation(close, index_close, w).reindex(df.index)
        for w in EXCESS_RETURN_WINDOWS:
            features[f"excess_return_{w}d_vs_nifty"] = xa.excess_log_return(close, index_close, w).reindex(
                df.index
            )

    if gold_close is not None:
        for w in MACRO_RETURN_WINDOWS:
            features[f"gold_return_{w}d"] = ta.log_return(gold_close, w).reindex(df.index)

    if silver_close is not None:
        for w in MACRO_RETURN_WINDOWS:
            features[f"silver_return_{w}d"] = ta.log_return(silver_close, w).reindex(df.index)

    if gold_close is not None and silver_close is not None:
        features["gold_silver_ratio"] = xa.price_ratio(gold_close, silver_close).reindex(df.index)

    macd_df = ta.macd(close)
    bb_df = ta.bollinger_bands(close)

    feature_df = pd.concat([pd.DataFrame(features), macd_df, bb_df], axis=1)
    feature_df.index = df.index
    return feature_df


def build_features_for_universe(instruments: list[Instrument] = UNIVERSE) -> None:
    FEATURES_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _load_close(ticker: str) -> pd.Series:
        return pd.read_parquet(PROCESSED_DATA_DIR / ticker_filename(ticker))["adj_close"]

    reference_index = _load_close(REFERENCE_INDEX_TICKER)
    gold = _load_close(GOLD_TICKER)
    silver = _load_close(SILVER_TICKER)

    for instrument in instruments:
        in_path = PROCESSED_DATA_DIR / ticker_filename(instrument.ticker)
        df = pd.read_parquet(in_path)
        # Cross-asset features are computed uniformly for every instrument,
        # including the reference instruments themselves (beta(x,x)=1,
        # corr(x,x)=1, excess_return(x,x)=0 are the mathematically exact
        # self-comparison values, not fabricated ones). Earlier this
        # special-cased the reference tickers to skip these columns as
        # "degenerate" — but that made those instruments' feature files
        # have a different column set than everyone else's, which silently
        # introduced NaN for their rows after Phase 4's pooled-panel
        # concat (columns present elsewhere but absent here get filled
        # with NaN by pandas). Uniform computation removes that class of
        # bug entirely and is simpler than the special-casing it replaced.
        feature_df = build_features_for_instrument(
            df, index_close=reference_index, gold_close=gold, silver_close=silver
        )
        out_path = FEATURES_DATA_DIR / ticker_filename(instrument.ticker)
        feature_df.to_parquet(out_path)

        # Every silver-derived column (silver_return_*, gold_silver_ratio) is
        # only defined from SILVERBEES.NS's 2022 listing onward (see
        # docs/roadmap.md), so an all-columns dropna() collapses warm-up rate
        # to whatever that instrument allows. Reporting both figures makes
        # that visible instead of it looking like a regression.
        n_valid_all = feature_df.dropna().shape[0]
        cols_excl_silver = [c for c in feature_df.columns if "silver" not in c]
        n_valid_excl_silver = feature_df[cols_excl_silver].dropna().shape[0]
        logger.info(
            "%s: %d feature rows (%d fully warmed up incl. silver features [%.0f%%]; "
            "%d excl. them [%.0f%%])",
            instrument.ticker,
            len(feature_df),
            n_valid_all,
            100 * n_valid_all / len(feature_df),
            n_valid_excl_silver,
            100 * n_valid_excl_silver / len(feature_df),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_features_for_universe()
