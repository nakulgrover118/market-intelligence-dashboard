import pandas as pd
import pytest

from app.data import clean
from app.data.paths import ticker_filename
from app.data.universe import AssetClass, Instrument


def _raw_df(n_bad: int = 3, n_good: int = 5, bad_col: str = "Close") -> pd.DataFrame:
    """`bad_col` rows are set to 0.0 for the first `n_bad` rows — defaults to
    a core column (Close) since that's what truncation actually keys on."""
    dates = pd.date_range("2020-01-01", periods=n_bad + n_good, freq="B", tz="Asia/Kolkata")
    df = pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(n_bad + n_good)],
            "High": [10.0 + i for i in range(n_bad + n_good)],
            "Low": [9.0 + i for i in range(n_bad + n_good)],
            "Close": [9.5 + i for i in range(n_bad + n_good)],
            "Adj Close": [9.5 + i for i in range(n_bad + n_good)],
            "Volume": [1000] * (n_bad + n_good),
            "Dividends": [0.0] * (n_bad + n_good),
            "Stock Splits": [0.0] * (n_bad + n_good),
        },
        index=dates,
    )
    df.loc[df.index[:n_bad], bad_col] = 0.0
    return df


def _raw_df_stale_leading_period(n_stale: int = 3, n_good: int = 5) -> pd.DataFrame:
    """The NESTLEIND.NS case: flat price, zero volume, for a leading run."""
    dates = pd.date_range("2020-01-01", periods=n_stale + n_good, freq="B", tz="Asia/Kolkata")
    close = [500.0] * n_stale + [500.0 + i for i in range(n_good)]
    volume = [0] * n_stale + [1000 + i for i in range(n_good)]
    return pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Adj Close": close,
            "Volume": volume,
            "Dividends": [0.0] * (n_stale + n_good),
            "Stock Splits": [0.0] * (n_stale + n_good),
        },
        index=dates,
    )


@pytest.fixture
def instrument():
    return Instrument("TEST.NS", "Test Co", AssetClass.STOCK, "IT")


def _write_raw(tmp_path, monkeypatch, instrument, df):
    monkeypatch.setattr(clean, "RAW_DATA_DIR", tmp_path / "raw")
    monkeypatch.setattr(clean, "PROCESSED_DATA_DIR", tmp_path / "processed")
    (tmp_path / "raw").mkdir()
    df.to_parquet((tmp_path / "raw") / ticker_filename(instrument.ticker))


def test_truncates_leading_bad_core_price_rows(tmp_path, monkeypatch, instrument):
    df = _raw_df(n_bad=3, n_good=5, bad_col="Close")
    _write_raw(tmp_path, monkeypatch, instrument, df)

    result = clean.clean_instrument(instrument)

    assert result.rows_in == 8
    assert result.rows_out == 5
    assert result.rows_dropped_leading_unusable == 3

    cleaned = pd.read_parquet(tmp_path / "processed" / ticker_filename(instrument.ticker))
    assert (cleaned["close"] > 0).all()


def test_bad_open_alone_does_not_truncate(tmp_path, monkeypatch, instrument):
    """Open isn't used by any current feature (see clean.py's _CORE_PRICE_COLS
    comment — this is the GOLDBEES.NS case from Phase 1), so a broken Open
    must not cause rows to be dropped."""
    df = _raw_df(n_bad=3, n_good=5, bad_col="Open")
    _write_raw(tmp_path, monkeypatch, instrument, df)

    result = clean.clean_instrument(instrument)

    assert result.rows_out == 8
    assert result.rows_dropped_leading_unusable == 0


def test_truncates_leading_stale_zero_volume_rows(tmp_path, monkeypatch, instrument):
    df = _raw_df_stale_leading_period(n_stale=3, n_good=5)
    _write_raw(tmp_path, monkeypatch, instrument, df)

    result = clean.clean_instrument(instrument)

    assert result.rows_out == 5
    assert result.rows_dropped_leading_unusable == 3

    cleaned = pd.read_parquet(tmp_path / "processed" / ticker_filename(instrument.ticker))
    assert (cleaned["volume"] > 0).all()


def test_index_zero_volume_does_not_truncate(tmp_path, monkeypatch):
    """The ^NSEI case: an index can legitimately have Volume == 0 for its
    entire history (Yahoo doesn't populate real trade volume for an index),
    while its price is completely valid. Must not be truncated like a
    stock/ETF's zero-volume placeholder data would be."""
    index_instrument = Instrument("^NSEI", "Nifty 50", AssetClass.INDEX)
    df = _raw_df_stale_leading_period(n_stale=3, n_good=5)
    _write_raw(tmp_path, monkeypatch, index_instrument, df)

    result = clean.clean_instrument(index_instrument)

    assert result.rows_out == 8
    assert result.rows_dropped_leading_unusable == 0


def test_derives_adjusted_high_low_from_adj_close_factor(tmp_path, monkeypatch, instrument):
    """A 2:1-split-style adj_close (half of raw close) must scale High/Low
    by the same factor, not leave them at raw values — otherwise ATR/etc.
    would see a fake volatility spike at the split date."""
    dates = pd.date_range("2020-01-01", periods=3, freq="B", tz="Asia/Kolkata")
    df = pd.DataFrame(
        {
            "Open": [100.0, 100.0, 100.0],
            "High": [110.0, 110.0, 110.0],
            "Low": [90.0, 90.0, 90.0],
            "Close": [100.0, 100.0, 100.0],
            "Adj Close": [50.0, 50.0, 50.0],  # half of Close, e.g. post a 2:1 split adjustment
            "Volume": [1000, 1000, 1000],
            "Dividends": [0.0, 0.0, 0.0],
            "Stock Splits": [0.0, 0.0, 0.0],
        },
        index=dates,
    )
    _write_raw(tmp_path, monkeypatch, instrument, df)

    clean.clean_instrument(instrument)

    cleaned = pd.read_parquet(tmp_path / "processed" / ticker_filename(instrument.ticker))
    assert (cleaned["adj_high"] == 55.0).all()
    assert (cleaned["adj_low"] == 45.0).all()


def test_renames_columns_to_snake_case(tmp_path, monkeypatch, instrument):
    df = _raw_df(n_bad=0, n_good=5)
    _write_raw(tmp_path, monkeypatch, instrument, df)

    clean.clean_instrument(instrument)

    cleaned = pd.read_parquet(tmp_path / "processed" / ticker_filename(instrument.ticker))
    assert list(cleaned.columns) == [
        "open", "high", "low", "close", "adj_close", "volume", "dividends", "stock_splits",
        "adj_high", "adj_low",
    ]


def test_drops_tz_and_names_index_date(tmp_path, monkeypatch, instrument):
    df = _raw_df(n_bad=0, n_good=5)
    _write_raw(tmp_path, monkeypatch, instrument, df)

    clean.clean_instrument(instrument)

    cleaned = pd.read_parquet(tmp_path / "processed" / ticker_filename(instrument.ticker))
    assert cleaned.index.tz is None
    assert cleaned.index.name == "date"


def test_drops_duplicate_dates_keeping_last(tmp_path, monkeypatch, instrument):
    df = _raw_df(n_bad=0, n_good=5)
    dup_row = df.iloc[[-1]].copy()
    dup_row["Close"] = 999.0
    df = pd.concat([df, dup_row])
    _write_raw(tmp_path, monkeypatch, instrument, df)

    result = clean.clean_instrument(instrument)

    assert result.rows_dropped_duplicate == 1
    cleaned = pd.read_parquet(tmp_path / "processed" / ticker_filename(instrument.ticker))
    assert cleaned["close"].iloc[-1] == 999.0


def test_all_bad_core_prices_raises(tmp_path, monkeypatch, instrument):
    df = _raw_df(n_bad=5, n_good=0, bad_col="Close")
    _write_raw(tmp_path, monkeypatch, instrument, df)

    with pytest.raises(ValueError):
        clean.clean_instrument(instrument)
