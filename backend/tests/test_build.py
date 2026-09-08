import numpy as np
import pandas as pd
import pytest

from app.data.universe import AssetClass, Instrument
from app.features import build
from app.features.build import build_features_for_instrument, build_features_for_universe


@pytest.fixture
def clean_df() -> pd.DataFrame:
    # Plain numpy arrays, not pd.Series: a Series carries its own RangeIndex,
    # which would silently misalign against the DatetimeIndex given to the
    # DataFrame below and NaN out every value (found by this fixture's own
    # test failing until this was fixed).
    rng = np.random.default_rng(7)
    n = 300
    close = 100 + np.cumsum(rng.normal(0.05, 1.0, n))
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "adj_close": close,
            "adj_high": close + 1.0,
            "adj_low": close - 1.0,
            "volume": rng.integers(1000, 5000, n).astype(float),
        },
        index=pd.date_range("2023-01-01", periods=n, freq="B", name="date"),
    )


@pytest.fixture
def reference_close(clean_df) -> pd.Series:
    rng = np.random.default_rng(11)
    n = len(clean_df)
    prices = 1000 + np.cumsum(rng.normal(0.02, 0.8, n))
    return pd.Series(prices, index=clean_df.index)


@pytest.fixture
def gold_close(clean_df) -> pd.Series:
    rng = np.random.default_rng(13)
    n = len(clean_df)
    prices = 50 + np.cumsum(rng.normal(0.01, 0.3, n))
    return pd.Series(prices, index=clean_df.index)


@pytest.fixture
def silver_close(clean_df) -> pd.Series:
    rng = np.random.default_rng(17)
    n = len(clean_df)
    prices = 60 + np.cumsum(rng.normal(0.01, 0.5, n))
    return pd.Series(prices, index=clean_df.index)


def test_output_is_aligned_with_input_index(clean_df):
    features = build_features_for_instrument(clean_df)
    pd.testing.assert_index_equal(features.index, clean_df.index)


def test_expected_columns_present(clean_df):
    features = build_features_for_instrument(clean_df)
    for expected in [
        "log_return_1d", "log_return_20d", "sma_200", "close_over_sma_10",
        "rsi_14", "atr_14", "volume_ratio_20", "macd_hist", "bb_pct_b",
    ]:
        assert expected in features.columns


def test_late_rows_are_fully_warmed_up(clean_df):
    """After the slowest warm-up window (SMA-200), every feature should be
    non-NaN."""
    features = build_features_for_instrument(clean_df)
    assert not features.iloc[250:].isna().any().any()


def test_uses_adjusted_prices_not_raw_close(clean_df):
    """Regression test: build.py once read df['close'] instead of
    df['adj_close'], which would inject a fake return spike at every split/
    dividend date (found via a real ~37% close-vs-adj_close divergence on
    INFY.NS). Simulate a stock split: adj_close is exactly half of close
    for the whole series, so a naive close-based return would be ~0, but
    the correct adj_close-based return should reflect the actual drift."""
    split_df = clean_df.copy()
    split_df["adj_close"] = split_df["close"] / 2
    split_df["adj_high"] = split_df["high"] / 2
    split_df["adj_low"] = split_df["low"] / 2

    features = build_features_for_instrument(split_df)
    expected = np.log(split_df["adj_close"] / split_df["adj_close"].shift(1))
    pd.testing.assert_series_equal(features["log_return_1d"], expected, check_names=False)


def test_full_pipeline_has_no_lookahead(clean_df):
    """The property that matters most: rebuilding the feature matrix with
    more history appended must not change any previously computed value."""
    truncate_at = 150
    full = build_features_for_instrument(clean_df)
    truncated = build_features_for_instrument(clean_df.iloc[:truncate_at])
    pd.testing.assert_frame_equal(full.iloc[:truncate_at], truncated)


def test_no_cross_asset_columns_when_references_omitted(clean_df):
    features = build_features_for_instrument(clean_df)
    cross_asset_cols = [c for c in features.columns if "nifty" in c or "gold" in c or "silver" in c]
    assert cross_asset_cols == []


def test_cross_asset_columns_present_when_references_given(clean_df, reference_close, gold_close, silver_close):
    features = build_features_for_instrument(
        clean_df, index_close=reference_close, gold_close=gold_close, silver_close=silver_close
    )
    for expected in [
        "beta_60d_vs_nifty", "corr_20d_vs_nifty", "excess_return_5d_vs_nifty",
        "gold_return_5d", "silver_return_20d", "gold_silver_ratio",
    ]:
        assert expected in features.columns


def test_full_pipeline_with_cross_asset_features_has_no_lookahead(
    clean_df, reference_close, gold_close, silver_close
):
    truncate_at = 150
    full = build_features_for_instrument(clean_df, reference_close, gold_close, silver_close)
    truncated = build_features_for_instrument(
        clean_df.iloc[:truncate_at],
        reference_close.iloc[:truncate_at],
        gold_close.iloc[:truncate_at],
        silver_close.iloc[:truncate_at],
    )
    pd.testing.assert_frame_equal(full.iloc[:truncate_at], truncated)


def test_universe_output_has_identical_columns_across_all_instruments(tmp_path, monkeypatch):
    """Regression test: build_features_for_universe used to skip computing
    cross-asset features for the reference instruments themselves (e.g. no
    beta-vs-Nifty column for ^NSEI), which gave different instruments
    different column sets. That was invisible until Phase 4 pooled all
    instruments into one panel — pandas concat silently filled the missing
    columns with NaN for whichever instrument lacked them, breaking model
    fitting. Every instrument's feature file must have the same columns."""
    monkeypatch.setattr(build, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(build, "FEATURES_DATA_DIR", tmp_path / "features")
    (tmp_path / "processed").mkdir()

    instruments = [
        Instrument(build.REFERENCE_INDEX_TICKER, "Nifty 50", AssetClass.INDEX),
        Instrument(build.GOLD_TICKER, "Gold ETF", AssetClass.COMMODITY),
        Instrument(build.SILVER_TICKER, "Silver ETF", AssetClass.COMMODITY),
        Instrument("STOCK.NS", "A Stock", AssetClass.STOCK, "IT"),
    ]
    rng = np.random.default_rng(3)
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="B", name="date")
    for instrument in instruments:
        close = 100 + np.cumsum(rng.normal(0.05, 1.0, n))
        df = pd.DataFrame(
            {
                "open": close - 0.5, "high": close + 1.0, "low": close - 1.0,
                "close": close, "adj_close": close, "adj_high": close + 1.0,
                "adj_low": close - 1.0, "volume": rng.integers(1000, 5000, n).astype(float),
            },
            index=dates,
        )
        df.to_parquet(tmp_path / "processed" / f"{instrument.ticker.replace('^', 'IDX_')}.parquet")

    build_features_for_universe(instruments=instruments)

    column_sets = {
        instrument.ticker: set(
            pd.read_parquet(tmp_path / "features" / f"{instrument.ticker.replace('^', 'IDX_')}.parquet").columns
        )
        for instrument in instruments
    }
    first_ticker, first_columns = next(iter(column_sets.items()))
    for ticker, columns in column_sets.items():
        assert columns == first_columns, f"{ticker} has different columns than {first_ticker}"
