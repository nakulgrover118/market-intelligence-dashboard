import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.data.universe import UNIVERSE
from app.main import app
from app.models.baseline import CATEGORICAL_COLUMNS
from app.models.gbm import build_lgbm_pipeline
from app.services import model_registry as registry_module
from app.services.model_registry import get_registry


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    """Persists a real, small trained model plus feature data for two of
    the real universe's tickers, wires the registry to read from a temp
    dir, and clears the lru_cache singleton so each test starts fresh."""
    monkeypatch.setattr(registry_module, "MODELS_DATA_DIR", tmp_path / "models")
    monkeypatch.setattr(registry_module, "FEATURES_DATA_DIR", tmp_path / "features")
    (tmp_path / "models").mkdir()
    (tmp_path / "features").mkdir()
    get_registry.cache_clear()

    rng = np.random.default_rng(0)
    n = 500
    numeric_columns = ["feature_a", "feature_b"]
    train_df = pd.DataFrame(
        {
            "feature_a": rng.normal(0, 1, n),
            "feature_b": rng.normal(0, 1, n),
            "sector": rng.choice(["IT", "Banking"], size=n),
        }
    )
    label = rng.binomial(1, 0.3, n)
    pipeline = build_lgbm_pipeline(numeric_columns, n_estimators=20)
    pipeline.fit(train_df[numeric_columns + CATEGORICAL_COLUMNS], label)

    artifact = {
        "pipeline": pipeline,
        "numeric_columns": numeric_columns,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "trained_through": pd.Timestamp("2024-01-01"),
        "horizon": 5,
    }
    joblib.dump(artifact, tmp_path / "models" / "lightgbm_5d.joblib")

    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    two_tickers = [UNIVERSE[0].ticker, UNIVERSE[1].ticker]
    for ticker in two_tickers:
        feature_df = pd.DataFrame(
            {"feature_a": rng.normal(0, 1, 10), "feature_b": rng.normal(0, 1, 10)}, index=dates
        )
        feature_df.to_parquet(tmp_path / "features" / f"{ticker}.parquet")

    yield TestClient(app)
    get_registry.cache_clear()


def test_list_instruments_returns_full_universe(api_client):
    response = api_client.get("/instruments")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == len(UNIVERSE)
    assert {"ticker", "name", "sector", "asset_class"} <= set(body[0].keys())


def test_latest_predictions_returns_only_tickers_with_feature_data(api_client):
    response = api_client.get("/predictions/latest?horizon=5")
    assert response.status_code == 200
    body = response.json()
    # Only the 2 tickers with feature files fixture-provided should appear,
    # not all 23+ instruments in the universe.
    assert len(body) == 2
    assert all(0.0 <= row["probability"] <= 1.0 for row in body)


def test_latest_predictions_sorted_by_probability_descending(api_client):
    response = api_client.get("/predictions/latest?horizon=5")
    probs = [row["probability"] for row in response.json()]
    assert probs == sorted(probs, reverse=True)


def test_latest_predictions_missing_model_returns_503(api_client):
    response = api_client.get("/predictions/latest?horizon=20")
    assert response.status_code == 503


def test_prediction_detail_includes_top_features(api_client):
    ticker = UNIVERSE[0].ticker
    response = api_client.get(f"/predictions/{ticker}?horizon=5")
    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == ticker
    # Fixture's tiny model has only 4 transformed features (2 numeric + 2
    # one-hot sector columns) — top_n=10 correctly returns all of them.
    assert 0 < len(body["top_features"]) <= 10
    assert {"feature", "feature_value", "shap_value"} <= set(body["top_features"][0].keys())


def test_prediction_detail_unknown_ticker_returns_404(api_client):
    response = api_client.get("/predictions/NOTAREALTICKER.NS?horizon=5")
    assert response.status_code == 404


def test_prediction_detail_ticker_without_feature_data_returns_404(api_client):
    # A real universe ticker that has no feature file in this fixture.
    ticker_without_data = UNIVERSE[2].ticker
    response = api_client.get(f"/predictions/{ticker_without_data}?horizon=5")
    assert response.status_code == 404


def test_health_still_works_alongside_new_routes(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
