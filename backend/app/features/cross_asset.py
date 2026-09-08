"""Features that require more than one instrument's series.

Two instruments rarely trade on an identical set of dates (listing gaps,
halts). Every function here aligns inputs via an *inner join* on date
before computing anything — never forward-fill, since a fabricated "flat"
day for a missing date would quietly bias a correlation or beta estimate.
Results are left on that inner-joined index; callers reindex back onto
their own instrument's calendar, which correctly introduces NaN wherever
the reference series has no data for that date (see build.py).
"""

import numpy as np
import pandas as pd


def _log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def _align(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner")
    return joined["a"], joined["b"]


def rolling_beta(instrument_close: pd.Series, reference_close: pd.Series, window: int) -> pd.Series:
    """Trailing OLS beta of instrument log-returns against reference
    log-returns: cov(instrument, reference) / var(reference)."""
    instrument_ret, reference_ret = _align(_log_returns(instrument_close), _log_returns(reference_close))
    cov = instrument_ret.rolling(window).cov(reference_ret)
    var = reference_ret.rolling(window).var()
    return cov / var


def rolling_correlation(instrument_close: pd.Series, reference_close: pd.Series, window: int) -> pd.Series:
    instrument_ret, reference_ret = _align(_log_returns(instrument_close), _log_returns(reference_close))
    return instrument_ret.rolling(window).corr(reference_ret)


def excess_log_return(instrument_close: pd.Series, reference_close: pd.Series, window: int) -> pd.Series:
    """instrument's N-day log return minus the reference's N-day log return
    over the same window — a simple relative-strength signal."""
    instrument_fwd = np.log(instrument_close / instrument_close.shift(window))
    reference_fwd = np.log(reference_close / reference_close.shift(window))
    instrument_fwd, reference_fwd = _align(instrument_fwd, reference_fwd)
    return instrument_fwd - reference_fwd


def price_ratio(numerator_close: pd.Series, denominator_close: pd.Series) -> pd.Series:
    numerator, denominator = _align(numerator_close, denominator_close)
    return numerator / denominator
