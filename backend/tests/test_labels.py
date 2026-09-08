import numpy as np
import pandas as pd
import pytest

from app.features import labels


@pytest.fixture
def close() -> pd.Series:
    rng = np.random.default_rng(21)
    n = 100
    prices = 100 + np.cumsum(rng.normal(0.05, 1.0, n))
    return pd.Series(prices, index=pd.date_range("2024-01-01", periods=n, freq="B"))


@pytest.fixture
def daily_vol(close) -> pd.Series:
    return np.log(close / close.shift(1)).rolling(20).std()


# --- forward_log_return ------------------------------------------------------


def test_forward_log_return_matches_manual_calc(close):
    result = labels.forward_log_return(close, horizon=5)
    manual = np.log(close.iloc[10 + 5] / close.iloc[10])
    assert result.iloc[10] == pytest.approx(manual)


def test_forward_log_return_last_n_rows_are_nan(close):
    horizon = 5
    result = labels.forward_log_return(close, horizon)
    assert result.iloc[-horizon:].isna().all()
    assert result.iloc[: -horizon].notna().all()


# --- volatility_threshold ------------------------------------------------------


def test_volatility_threshold_scales_with_sqrt_horizon(daily_vol):
    t5 = labels.volatility_threshold(daily_vol, horizon=5, k=1.0)
    t20 = labels.volatility_threshold(daily_vol, horizon=20, k=1.0)
    ratio = (t20 / t5).dropna()
    assert ratio.iloc[0] == pytest.approx(np.sqrt(20 / 5))


def test_volatility_threshold_scales_with_k(daily_vol):
    t_1 = labels.volatility_threshold(daily_vol, horizon=5, k=1.0)
    t_2 = labels.volatility_threshold(daily_vol, horizon=5, k=2.0)
    ratio = (t_2 / t_1).dropna()
    np.testing.assert_allclose(ratio.to_numpy(), 2.0)


def test_volatility_threshold_has_no_lookahead(daily_vol):
    truncate_at = 60
    full = labels.volatility_threshold(daily_vol, horizon=5, k=1.0)
    truncated = labels.volatility_threshold(daily_vol.iloc[:truncate_at], horizon=5, k=1.0)
    pd.testing.assert_series_equal(full.iloc[:truncate_at], truncated, check_names=False)


# --- label_end_date ------------------------------------------------------


def test_label_end_date_matches_manual_offset(close):
    result = labels.label_end_date(close.index, horizon=5)
    assert result.iloc[10] == close.index[15]


def test_label_end_date_is_nat_for_last_n_rows(close):
    horizon = 5
    result = labels.label_end_date(close.index, horizon)
    assert result.iloc[-horizon:].isna().all()


def test_label_end_date_all_nat_when_horizon_exceeds_length():
    short_index = pd.date_range("2024-01-01", periods=3, freq="B")
    result = labels.label_end_date(short_index, horizon=10)
    assert result.isna().all()


# --- build_labels_for_instrument ------------------------------------------------------


def test_build_labels_basic_shape(close, daily_vol):
    df = labels.build_labels_for_instrument(close, daily_vol, k=1.0)
    for n in labels.HORIZONS:
        for prefix in ["forward_return", "label_threshold", "label", "label_end_date"]:
            assert f"{prefix}_{n}d" in df.columns


def test_label_is_one_iff_forward_return_meets_threshold(close, daily_vol):
    df = labels.build_labels_for_instrument(close, daily_vol, k=1.0)
    valid = df.dropna(subset=["forward_return_5d", "label_threshold_5d"])
    expected = (valid["forward_return_5d"] >= valid["label_threshold_5d"])
    actual = valid["label_5d"].astype(bool)
    pd.testing.assert_series_equal(actual, expected, check_names=False)


def test_label_is_na_when_forward_return_undefined(close, daily_vol):
    df = labels.build_labels_for_instrument(close, daily_vol, k=1.0)
    tail = df.iloc[-5:]
    assert tail["label_5d"].isna().all()
