import joblib
import numpy as np
import pandas as pd
import pytest

from app.models.baseline import CATEGORICAL_COLUMNS
from app.models.gbm import build_lgbm_pipeline
from app.services import model_registry as registry_module
from app.services.model_registry import ModelNotAvailableError, ModelRegistry

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
def trained_artifact_dir(tmp_path, monkeypatch):
    """Persists real, tiny LightGBM pipelines for both directions at
    horizon=5 (up and down), plus a feature file for one ticker —
    exercising the real prediction/explanation path, not a mock."""
    monkeypatch.setattr(registry_module, "MODELS_DATA_DIR", tmp_path / "models")
    monkeypatch.setattr(registry_module, "FEATURES_DATA_DIR", tmp_path / "features")
    (tmp_path / "models").mkdir()
    (tmp_path / "features").mkdir()

    _persist_artifact(tmp_path / "models", "up", 5, seed=0)
    _persist_artifact(tmp_path / "models", "down", 5, seed=1)

    rng = np.random.default_rng(2)
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    feature_df = pd.DataFrame(
        {"feature_a": rng.normal(0, 1, 10), "feature_b": rng.normal(0, 1, 10)}, index=dates
    )
    feature_df.to_parquet(tmp_path / "features" / "TEST.NS.parquet")

    return tmp_path


def test_predict_returns_probability_and_date(trained_artifact_dir):
    registry = ModelRegistry(horizons=(5,), directions=("up", "down"))
    result = registry.predict_and_explain("up", 5, "TEST.NS", include_explanation=False)

    assert result is not None
    assert result["direction"] == "up"
    assert 0.0 <= result["probability"] <= 1.0
    assert result["as_of_date"] == "2024-01-12T00:00:00"  # 10th business day from 2024-01-01
    assert "top_features" not in result


def test_predict_includes_explanation_when_requested(trained_artifact_dir):
    registry = ModelRegistry(horizons=(5,), directions=("up", "down"))
    result = registry.predict_and_explain("up", 5, "TEST.NS", include_explanation=True, top_n=2)

    assert len(result["top_features"]) == 2
    assert {"feature", "feature_value", "shap_value"} <= set(result["top_features"][0].keys())


def test_predict_returns_none_for_unknown_ticker(trained_artifact_dir):
    registry = ModelRegistry(horizons=(5,), directions=("up", "down"))
    result = registry.predict_and_explain("up", 5, "NOPE.NS", include_explanation=False)
    assert result is None


def test_predict_raises_for_unpersisted_horizon(trained_artifact_dir):
    registry = ModelRegistry(horizons=(5,), directions=("up", "down"))
    with pytest.raises(ModelNotAvailableError):
        registry.predict_and_explain("up", 20, "TEST.NS")


def test_predict_raises_for_unpersisted_direction(trained_artifact_dir):
    registry = ModelRegistry(horizons=(5,), directions=("up",))  # "down" not loaded
    with pytest.raises(ModelNotAvailableError):
        registry.predict_and_explain("down", 5, "TEST.NS")


def test_available_models_reflects_what_was_actually_persisted(trained_artifact_dir):
    registry = ModelRegistry(horizons=(5, 20), directions=("up", "down"))
    assert registry.available_models() == [("down", 5), ("up", 5)]


def test_predict_both_directions_returns_up_and_down(trained_artifact_dir):
    registry = ModelRegistry(horizons=(5,), directions=("up", "down"))
    both = registry.predict_both_directions(5, "TEST.NS", include_explanation=False)

    assert both is not None
    assert set(both.keys()) == {"up", "down"}
    assert both["up"]["direction"] == "up"
    assert both["down"]["direction"] == "down"
    # Different models (different seeds/labels), so predictions shouldn't
    # coincidentally be identical — a real check that both are actually
    # being computed from distinct artifacts, not one reused for both.
    assert both["up"]["probability"] != both["down"]["probability"]


def test_predict_both_directions_returns_none_for_unknown_ticker(trained_artifact_dir):
    registry = ModelRegistry(horizons=(5,), directions=("up", "down"))
    assert registry.predict_both_directions(5, "NOPE.NS") is None
