import numpy as np
import pandas as pd
import pytest

from app.features import cross_asset as xa


@pytest.fixture
def reference_close() -> pd.Series:
    rng = np.random.default_rng(3)
    n = 300
    returns = rng.normal(0.0003, 0.01, n)
    prices = 1000 * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=pd.date_range("2023-01-01", periods=n, freq="B"))


def _instrument_with_known_beta(reference_close: pd.Series, beta: float, noise_scale: float = 0.001) -> pd.Series:
    rng = np.random.default_rng(99)
    reference_ret = np.log(reference_close / reference_close.shift(1)).dropna()
    idiosyncratic = rng.normal(0, noise_scale, len(reference_ret))
    instrument_ret = beta * reference_ret.values + idiosyncratic
    prices = 100 * np.exp(np.concatenate([[0], np.cumsum(instrument_ret)]))
    return pd.Series(prices, index=reference_close.index)


def test_rolling_beta_recovers_known_beta(reference_close):
    instrument_close = _instrument_with_known_beta(reference_close, beta=1.5)
    result = xa.rolling_beta(instrument_close, reference_close, window=200)
    assert result.dropna().iloc[-1] == pytest.approx(1.5, abs=0.1)


def test_rolling_correlation_near_one_for_scaled_series(reference_close):
    # A pure positive linear scaling of the reference's returns must be
    # (near) perfectly correlated with it.
    instrument_close = _instrument_with_known_beta(reference_close, beta=2.0, noise_scale=0.0)
    result = xa.rolling_correlation(instrument_close, reference_close, window=60)
    assert result.dropna().iloc[-1] == pytest.approx(1.0, abs=1e-6)


def test_excess_log_return_is_zero_for_identical_series(reference_close):
    result = xa.excess_log_return(reference_close, reference_close, window=10)
    np.testing.assert_allclose(result.dropna().to_numpy(), 0.0, atol=1e-9)


def test_price_ratio_matches_manual_division():
    a = pd.Series([10.0, 20.0, 30.0], index=pd.date_range("2024-01-01", periods=3))
    b = pd.Series([2.0, 4.0, 5.0], index=pd.date_range("2024-01-01", periods=3))
    result = xa.price_ratio(a, b)
    pd.testing.assert_series_equal(result, a / b, check_names=False)


def test_alignment_drops_dates_missing_from_reference_and_does_not_ffill():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    instrument_close = pd.Series(np.arange(100.0, 110.0), index=dates)
    reference_close = pd.Series(np.arange(200.0, 210.0), index=dates).drop(dates[5])

    result = xa.rolling_correlation(instrument_close, reference_close, window=3)

    assert dates[5] not in result.index
    assert len(result) == len(dates) - 1


# --- No-lookahead invariance --------------------------------------------------

TRUNCATE_AT = 150


def test_rolling_beta_has_no_lookahead(reference_close):
    instrument_close = _instrument_with_known_beta(reference_close, beta=0.8)
    full = xa.rolling_beta(instrument_close, reference_close, window=60)
    truncated = xa.rolling_beta(
        instrument_close.iloc[:TRUNCATE_AT], reference_close.iloc[:TRUNCATE_AT], window=60
    )
    pd.testing.assert_series_equal(full.iloc[:TRUNCATE_AT], truncated, check_names=False)


def test_rolling_correlation_has_no_lookahead(reference_close):
    instrument_close = _instrument_with_known_beta(reference_close, beta=0.8)
    full = xa.rolling_correlation(instrument_close, reference_close, window=60)
    truncated = xa.rolling_correlation(
        instrument_close.iloc[:TRUNCATE_AT], reference_close.iloc[:TRUNCATE_AT], window=60
    )
    pd.testing.assert_series_equal(full.iloc[:TRUNCATE_AT], truncated, check_names=False)


def test_excess_log_return_has_no_lookahead(reference_close):
    instrument_close = _instrument_with_known_beta(reference_close, beta=0.8)
    full = xa.excess_log_return(instrument_close, reference_close, window=20)
    truncated = xa.excess_log_return(
        instrument_close.iloc[:TRUNCATE_AT], reference_close.iloc[:TRUNCATE_AT], window=20
    )
    pd.testing.assert_series_equal(full.iloc[:TRUNCATE_AT], truncated, check_names=False)
