import numpy as np
import pandas as pd
import pytest

from app.features.build import build_features_for_instrument


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
            "volume": rng.integers(1000, 5000, n).astype(float),
        },
        index=pd.date_range("2023-01-01", periods=n, freq="B", name="date"),
    )


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


def test_full_pipeline_has_no_lookahead(clean_df):
    """The property that matters most: rebuilding the feature matrix with
    more history appended must not change any previously computed value."""
    truncate_at = 150
    full = build_features_for_instrument(clean_df)
    truncated = build_features_for_instrument(clean_df.iloc[:truncate_at])
    pd.testing.assert_frame_equal(full.iloc[:truncate_at], truncated)
