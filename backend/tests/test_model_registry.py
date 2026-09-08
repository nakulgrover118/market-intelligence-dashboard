import joblib
import numpy as np
import pandas as pd
import pytest

from app.models.baseline import CATEGORICAL_COLUMNS
from app.models.gbm import build_lgbm_pipeline
from app.services import model_registry as registry_module
from app.services.model_registry import ModelNotAvailableError, ModelRegistry


@pytest.fixture
def trained_artifact_dir(tmp_path, monkeypatch):
    """Fits a tiny real LightGBM pipeline and persists it in the same
    shape app.models.persist would, plus a feature file for one ticker —
    exercising the real prediction/explanation path, not a mock."""
    monkeypatch.setattr(registry_module, "MODELS_DATA_DIR", tmp_path / "models")
    monkeypatch.setattr(registry_module, "FEATURES_DATA_DIR", tmp_path / "features")
    (tmp_path / "models").mkdir()
    (tmp_path / "features").mkdir()

    rng = np.random.default_rng(0)
    n = 500
    train_df = pd.DataFrame(
        {
            "feature_a": rng.normal(0, 1, n),
            "feature_b": rng.normal(0, 1, n),
            "sector": rng.choice(["IT", "Banking"], size=n),
        }
    )
    label = rng.binomial(1, 0.3, n)
    numeric_columns = ["feature_a", "feature_b"]
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
    feature_df = pd.DataFrame(
        {"feature_a": rng.normal(0, 1, 10), "feature_b": rng.normal(0, 1, 10)}, index=dates
    )
    feature_df.to_parquet(tmp_path / "features" / "TEST.NS.parquet")

    return tmp_path


def test_predict_returns_probability_and_date(trained_artifact_dir):
    registry = ModelRegistry(horizons=(5,))
    result = registry.predict_and_explain(5, "TEST.NS", include_explanation=False)

    assert result is not None
    assert 0.0 <= result["probability"] <= 1.0
    assert result["as_of_date"] == "2024-01-12T00:00:00"  # 10th business day from 2024-01-01
    assert "top_features" not in result


def test_predict_includes_explanation_when_requested(trained_artifact_dir):
    registry = ModelRegistry(horizons=(5,))
    result = registry.predict_and_explain(5, "TEST.NS", include_explanation=True, top_n=2)

    assert len(result["top_features"]) == 2
    assert {"feature", "feature_value", "shap_value"} <= set(result["top_features"][0].keys())


def test_predict_returns_none_for_unknown_ticker(trained_artifact_dir):
    registry = ModelRegistry(horizons=(5,))
    result = registry.predict_and_explain(5, "NOPE.NS", include_explanation=False)
    assert result is None


def test_predict_raises_for_unpersisted_horizon(trained_artifact_dir):
    registry = ModelRegistry(horizons=(5,))
    with pytest.raises(ModelNotAvailableError):
        registry.predict_and_explain(20, "TEST.NS")


def test_available_horizons_reflects_what_was_actually_persisted(trained_artifact_dir):
    registry = ModelRegistry(horizons=(5, 20))
    assert registry.available_horizons() == [5]
