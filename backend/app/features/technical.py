"""Technical/statistical indicators, implemented from their definitions.

Every function here follows one invariant: the value at row t depends only
on rows <= t (the current bar's close and everything before it). No
`center=True` rolling windows, no negative shifts. `tests/test_technical.py`
enforces this with a generic "no lookahead" check run against every
indicator, not just spot-checked math.
"""

import numpy as np
import pandas as pd


def log_return(close: pd.Series, window: int) -> pd.Series:
    return np.log(close / close.shift(window))


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def rate_of_change(close: pd.Series, window: int) -> pd.Series:
    return close.pct_change(window)


def realized_volatility(close: pd.Series, window: int) -> pd.Series:
    """Rolling std of daily log returns — a trailing measure of how noisy
    the series has recently been, not annualized (left raw; annualizing is
    a cosmetic scale change the model doesn't need)."""
    daily_log_return = np.log(close / close.shift(1))
    return daily_log_return.rolling(window).std()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI. Wilder's smoothing is itself just an EMA with
    alpha = 1/window, which is trailing-only by construction."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd_line": macd_line, "macd_signal": signal_line, "macd_hist": histogram}
    )


def average_true_range(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = sma(close, window)
    std = close.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / mid
    percent_b = (close - lower) / (upper - lower)
    return pd.DataFrame(
        {"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_width": width, "bb_pct_b": percent_b}
    )


def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    return volume / volume.rolling(window).mean()
