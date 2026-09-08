import numpy as np
import pandas as pd
import pytest

from app.data.universe import AssetClass, Instrument
from app.models import dataset


def _write_instrument(tmp_path, monkeypatch, ticker, feature_df, label_df):
    monkeypatch.setattr(dataset, "FEATURES_DATA_DIR", tmp_path / "features")
    monkeypatch.setattr(dataset, "LABELS_DATA_DIR", tmp_path / "labels")
    (tmp_path / "features").mkdir(exist_ok=True)
    (tmp_path / "labels").mkdir(exist_ok=True)
    feature_df.to_parquet((tmp_path / "features") / f"{ticker}.parquet")
    label_df.to_parquet((tmp_path / "labels") / f"{ticker}.parquet")


def _sample_data(n=10, with_silver=True, with_nan=False):
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    feature_df = pd.DataFrame(
        {
            "log_return_1d": np.linspace(0.01, 0.02, n),
            "rsi_14": np.linspace(40, 60, n),
        },
        index=dates,
    )
    if with_silver:
        feature_df["silver_return_5d"] = np.linspace(-0.01, 0.01, n)
    if with_nan:
        feature_df.loc[dates[0], "rsi_14"] = np.nan

    label_df = pd.DataFrame(
        {
            "label_5d": [1, 0] * (n // 2),
            "label_end_date_5d": dates + pd.Timedelta(days=5),
            "forward_return_5d": np.linspace(-0.02, 0.03, n),
        },
        index=dates,
    )
    return feature_df, label_df


@pytest.fixture
def instrument():
    return Instrument("TEST.NS", "Test Co", AssetClass.STOCK, "IT")


def test_basic_columns_and_metadata(tmp_path, monkeypatch, instrument):
    feature_df, label_df = _sample_data()
    _write_instrument(tmp_path, monkeypatch, instrument.ticker, feature_df, label_df)

    panel = dataset.load_modeling_dataset(horizon=5, instruments=[instrument])

    assert "label" in panel.columns
    assert "label_end_date" in panel.columns
    assert (panel["ticker"] == "TEST.NS").all()
    assert (panel["sector"] == "IT").all()
    assert panel["label"].dtype == np.int64 or panel["label"].dtype == int


def test_silver_features_excluded_by_default(tmp_path, monkeypatch, instrument):
    feature_df, label_df = _sample_data(with_silver=True)
    _write_instrument(tmp_path, monkeypatch, instrument.ticker, feature_df, label_df)

    panel = dataset.load_modeling_dataset(horizon=5, instruments=[instrument])

    assert not any("silver" in c for c in panel.columns)


def test_silver_features_included_when_requested(tmp_path, monkeypatch, instrument):
    feature_df, label_df = _sample_data(with_silver=True)
    _write_instrument(tmp_path, monkeypatch, instrument.ticker, feature_df, label_df)

    panel = dataset.load_modeling_dataset(horizon=5, instruments=[instrument], include_silver_features=True)

    assert "silver_return_5d" in panel.columns


def test_rows_with_nan_features_are_dropped(tmp_path, monkeypatch, instrument):
    feature_df, label_df = _sample_data(n=10, with_nan=True)
    _write_instrument(tmp_path, monkeypatch, instrument.ticker, feature_df, label_df)

    panel = dataset.load_modeling_dataset(horizon=5, instruments=[instrument])

    assert len(panel) == 9
    assert panel["rsi_14"].notna().all()


def test_raises_when_instruments_have_inconsistent_columns(tmp_path, monkeypatch):
    instrument_a = Instrument("A.NS", "A Co", AssetClass.STOCK, "IT")
    instrument_b = Instrument("B.NS", "B Co", AssetClass.STOCK, "Banking")

    monkeypatch.setattr(dataset, "FEATURES_DATA_DIR", tmp_path / "features")
    monkeypatch.setattr(dataset, "LABELS_DATA_DIR", tmp_path / "labels")
    (tmp_path / "features").mkdir(exist_ok=True)
    (tmp_path / "labels").mkdir(exist_ok=True)

    feature_df_a, label_df_a = _sample_data(with_silver=False)
    feature_df_a.to_parquet((tmp_path / "features") / "A.NS.parquet")
    label_df_a.to_parquet((tmp_path / "labels") / "A.NS.parquet")

    feature_df_b, label_df_b = _sample_data(with_silver=False)
    feature_df_b = feature_df_b.drop(columns=["rsi_14"])  # missing a column B's peers have
    feature_df_b.to_parquet((tmp_path / "features") / "B.NS.parquet")
    label_df_b.to_parquet((tmp_path / "labels") / "B.NS.parquet")

    with pytest.raises(ValueError, match="inconsistent feature columns"):
        dataset.load_modeling_dataset(horizon=5, instruments=[instrument_a, instrument_b])


def test_pools_multiple_instruments(tmp_path, monkeypatch):
    instrument_a = Instrument("A.NS", "A Co", AssetClass.STOCK, "IT")
    instrument_b = Instrument("B.NS", "B Co", AssetClass.STOCK, "Banking")

    monkeypatch.setattr(dataset, "FEATURES_DATA_DIR", tmp_path / "features")
    monkeypatch.setattr(dataset, "LABELS_DATA_DIR", tmp_path / "labels")
    (tmp_path / "features").mkdir(exist_ok=True)
    (tmp_path / "labels").mkdir(exist_ok=True)
    for ticker in ["A.NS", "B.NS"]:
        feature_df, label_df = _sample_data(with_silver=False)
        feature_df.to_parquet((tmp_path / "features") / f"{ticker}.parquet")
        label_df.to_parquet((tmp_path / "labels") / f"{ticker}.parquet")

    panel = dataset.load_modeling_dataset(horizon=5, instruments=[instrument_a, instrument_b])

    assert set(panel["ticker"].unique()) == {"A.NS", "B.NS"}
    assert len(panel) == 20
