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

NUMERIC_COLUMNS = ["feature_a", "feature_b"]


def _fit_fake_pipeline(seed: int):
    rng = np.random.default_rng(seed)
    n = 500
    train_df = pd.DataFrame(
        {
            "feature_a": rng.normal(0, 1, n),
            "feature_b": rng.normal(0, 1, n),
            "sector": rng.choice(["IT", "Banking"], size=n),
        }
    )
    label = rng.binomial(1, 0.3, n)
    pipeline = build_lgbm_pipeline(NUMERIC_COLUMNS, n_estimators=20)
    pipeline.fit(train_df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS], label)
    return pipeline


def _persist_artifact(models_dir, direction: str, horizon: int, seed: int) -> None:
    artifact = {
        "pipeline": _fit_fake_pipeline(seed),
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "trained_through": pd.Timestamp("2024-01-01"),
        "horizon": horizon,
        "direction": direction,
    }
    joblib.dump(artifact, models_dir / f"lightgbm_{direction}_{horizon}d.joblib")


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    """Persists real, small trained models (both directions, horizon=5
    only) plus feature data for two of the real universe's tickers, wires
    the registry to read from a temp dir, and clears the lru_cache
    singleton so each test starts fresh."""
    monkeypatch.setattr(registry_module, "MODELS_DATA_DIR", tmp_path / "models")
    monkeypatch.setattr(registry_module, "FEATURES_DATA_DIR", tmp_path / "features")
    (tmp_path / "models").mkdir()
    (tmp_path / "features").mkdir()
    get_registry.cache_clear()

    _persist_artifact(tmp_path / "models", "up", 5, seed=0)
    _persist_artifact(tmp_path / "models", "down", 5, seed=1)

    rng = np.random.default_rng(2)
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
    for row in body:
        assert 0.0 <= row["up_probability"] <= 1.0
        assert 0.0 <= row["down_probability"] <= 1.0


def test_latest_predictions_sorted_by_max_probability_descending(api_client):
    response = api_client.get("/predictions/latest?horizon=5")
    body = response.json()
    max_probs = [max(row["up_probability"], row["down_probability"]) for row in body]
    assert max_probs == sorted(max_probs, reverse=True)


def test_latest_predictions_missing_model_returns_503(api_client):
    response = api_client.get("/predictions/latest?horizon=20")
    assert response.status_code == 503


def test_prediction_detail_includes_up_and_down_top_features(api_client):
    ticker = UNIVERSE[0].ticker
    response = api_client.get(f"/predictions/{ticker}?horizon=5")
    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == ticker
    assert 0.0 <= body["up_probability"] <= 1.0
    assert 0.0 <= body["down_probability"] <= 1.0
    # Fixture's tiny model has only 4 transformed features (2 numeric + 2
    # one-hot sector columns) — top_n=10 correctly returns all of them.
    for key in ("up_top_features", "down_top_features"):
        assert 0 < len(body[key]) <= 10
        assert {"feature", "feature_value", "shap_value"} <= set(body[key][0].keys())
    # Up and down are genuinely different models (different fixture
    # seeds) — this would catch a bug where one direction's result gets
    # reused for the other.
    assert body["up_probability"] != body["down_probability"]


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
