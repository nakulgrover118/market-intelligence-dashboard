import numpy as np
import pandas as pd
import pytest

from app.features import technical as ta


@pytest.fixture
def price_series() -> pd.Series:
    """A synthetic random-walk-ish close series, long enough for the
    warm-up periods of every indicator (MACD's slow EMA + signal, etc.)."""
    rng = np.random.default_rng(42)
    steps = rng.normal(loc=0.05, scale=1.0, size=200)
    prices = 100 + np.cumsum(steps)
    return pd.Series(prices, index=pd.date_range("2023-01-01", periods=200, freq="B"), name="close")


@pytest.fixture
def hlc(price_series):
    """Synthetic High/Low/Close consistent with each other (High >= Close >= Low)."""
    high = price_series + 1.0
    low = price_series - 1.0
    return high, low, price_series


# --- Correctness spot-checks -------------------------------------------------


def test_sma_matches_manual_mean(price_series):
    result = ta.sma(price_series, window=5)
    manual = price_series.iloc[10:15].mean()
    assert result.iloc[14] == pytest.approx(manual)


def test_rsi_approaches_100_for_monotonic_increase():
    close = pd.Series(np.arange(1, 100, dtype=float))
    result = ta.rsi(close, window=14)
    assert result.iloc[-1] > 99


def test_rsi_approaches_0_for_monotonic_decrease():
    close = pd.Series(np.arange(100, 1, -1, dtype=float))
    result = ta.rsi(close, window=14)
    assert result.iloc[-1] < 1


def test_macd_histogram_equals_macd_minus_signal(price_series):
    result = ta.macd(price_series)
    pd.testing.assert_series_equal(
        result["macd_hist"],
        result["macd_line"] - result["macd_signal"],
        check_names=False,
    )


def test_atr_is_non_negative(hlc):
    high, low, close = hlc
    result = ta.average_true_range(high, low, close)
    assert (result.dropna() >= 0).all()


def test_bollinger_band_ordering(price_series):
    result = ta.bollinger_bands(price_series)
    valid = result.dropna()
    assert (valid["bb_upper"] >= valid["bb_mid"]).all()
    assert (valid["bb_mid"] >= valid["bb_lower"]).all()


def test_volume_ratio_is_one_for_constant_volume():
    volume = pd.Series([1000.0] * 30)
    result = ta.volume_ratio(volume, window=10)
    assert result.iloc[-1] == pytest.approx(1.0)


# --- No-lookahead invariance --------------------------------------------------
# The property that actually matters for this project: a feature computed
# "as of" day t must not change when more history is appended after day t.
# If it did, the feature would only be reproducible in hindsight, which is
# exactly what would make a backtest built on it invalid.

TRUNCATE_AT = 100


def _assert_prefix_unchanged(full_result, truncated_result):
    if isinstance(full_result, pd.DataFrame):
        pd.testing.assert_frame_equal(full_result.iloc[:TRUNCATE_AT], truncated_result)
    else:
        pd.testing.assert_series_equal(full_result.iloc[:TRUNCATE_AT], truncated_result, check_names=False)


def test_sma_has_no_lookahead(price_series):
    full = ta.sma(price_series, window=20)
    truncated = ta.sma(price_series.iloc[:TRUNCATE_AT], window=20)
    _assert_prefix_unchanged(full, truncated)


def test_ema_has_no_lookahead(price_series):
    full = ta.ema(price_series, span=20)
    truncated = ta.ema(price_series.iloc[:TRUNCATE_AT], span=20)
    _assert_prefix_unchanged(full, truncated)


def test_log_return_has_no_lookahead(price_series):
    full = ta.log_return(price_series, window=5)
    truncated = ta.log_return(price_series.iloc[:TRUNCATE_AT], window=5)
    _assert_prefix_unchanged(full, truncated)


def test_rate_of_change_has_no_lookahead(price_series):
    full = ta.rate_of_change(price_series, window=5)
    truncated = ta.rate_of_change(price_series.iloc[:TRUNCATE_AT], window=5)
    _assert_prefix_unchanged(full, truncated)


def test_realized_volatility_has_no_lookahead(price_series):
    full = ta.realized_volatility(price_series, window=20)
    truncated = ta.realized_volatility(price_series.iloc[:TRUNCATE_AT], window=20)
    _assert_prefix_unchanged(full, truncated)


def test_rsi_has_no_lookahead(price_series):
    full = ta.rsi(price_series)
    truncated = ta.rsi(price_series.iloc[:TRUNCATE_AT])
    _assert_prefix_unchanged(full, truncated)


def test_macd_has_no_lookahead(price_series):
    full = ta.macd(price_series)
    truncated = ta.macd(price_series.iloc[:TRUNCATE_AT])
    _assert_prefix_unchanged(full, truncated)


def test_atr_has_no_lookahead(hlc):
    high, low, close = hlc
    full = ta.average_true_range(high, low, close)
    truncated = ta.average_true_range(high.iloc[:TRUNCATE_AT], low.iloc[:TRUNCATE_AT], close.iloc[:TRUNCATE_AT])
    _assert_prefix_unchanged(full, truncated)


def test_bollinger_bands_has_no_lookahead(price_series):
    full = ta.bollinger_bands(price_series)
    truncated = ta.bollinger_bands(price_series.iloc[:TRUNCATE_AT])
    _assert_prefix_unchanged(full, truncated)


def test_volume_ratio_has_no_lookahead():
    volume = pd.Series(np.random.default_rng(1).integers(500, 5000, size=200).astype(float))
    full = ta.volume_ratio(volume, window=20)
    truncated = ta.volume_ratio(volume.iloc[:TRUNCATE_AT], window=20)
    _assert_prefix_unchanged(full, truncated)
