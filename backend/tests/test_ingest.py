from unittest.mock import patch

import pandas as pd
import pytest

from app.data import ingest
from app.data.universe import AssetClass, Instrument


def _sample_df(n: int = 5) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(n)],
            "High": [101.0 + i for i in range(n)],
            "Low": [99.0 + i for i in range(n)],
            "Close": [100.5 + i for i in range(n)],
            "Adj Close": [100.5 + i for i in range(n)],
            "Volume": [1000 + i for i in range(n)],
        },
        index=dates,
    )


def test_validate_clean_data_has_no_warnings():
    df = _sample_df()
    assert ingest._validate("TEST", df) == []


def test_validate_flags_duplicate_dates():
    df = _sample_df()
    df = pd.concat([df, df.iloc[[0]]])
    warnings = ingest._validate("TEST", df)
    assert any("duplicate" in w for w in warnings)


def test_validate_flags_non_positive_prices():
    df = _sample_df()
    df.loc[df.index[0], "Close"] = 0
    warnings = ingest._validate("TEST", df)
    assert any("non-positive" in w for w in warnings)


def test_ingest_instrument_writes_parquet(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "RAW_DATA_DIR", tmp_path)
    instrument = Instrument("TEST.NS", "Test Co", AssetClass.STOCK, "IT")

    with patch.object(ingest, "fetch_ticker_history", return_value=_sample_df()):
        result = ingest.ingest_instrument(instrument)

    assert result.ok
    assert result.rows == 5
    assert (tmp_path / "TEST.NS.parquet").exists()

    written = pd.read_parquet(tmp_path / "TEST.NS.parquet")
    assert len(written) == 5


def test_ingest_instrument_records_fetch_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest, "RAW_DATA_DIR", tmp_path)
    instrument = Instrument("BAD.NS", "Bad Co", AssetClass.STOCK, "IT")

    with patch.object(ingest, "fetch_ticker_history", side_effect=RuntimeError("boom")):
        result = ingest.ingest_instrument(instrument)

    assert not result.ok
    assert "boom" in result.error


def test_fetch_ticker_history_retries_then_raises(monkeypatch):
    monkeypatch.setattr(ingest.time, "sleep", lambda _: None)
    with patch.object(ingest.yf, "Ticker", side_effect=RuntimeError("network down")):
        with pytest.raises(RuntimeError):
            ingest.fetch_ticker_history("WHATEVER.NS", max_retries=2)
