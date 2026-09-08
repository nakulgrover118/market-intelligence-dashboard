import numpy as np
import pandas as pd
import pytest

from app.models.explain import (
    compute_shap_values,
    explain_single_prediction,
    global_feature_importance,
    transformed_feature_matrix,
)
from app.models.gbm import build_lgbm_pipeline


def test_global_feature_importance_ranks_by_mean_abs_shap():
    feature_names = ["a", "b", "c"]
    shap_values = np.array(
        [
            [1.0, -0.1, 0.0],
            [-1.0, 0.2, 0.0],
            [2.0, 0.1, 0.0],
        ]
    )
    result = global_feature_importance(shap_values, feature_names)

    assert list(result["feature"]) == ["a", "b", "c"]
    assert result.iloc[0]["mean_abs_shap"] == pytest.approx((1.0 + 1.0 + 2.0) / 3)
    assert result.iloc[2]["mean_abs_shap"] == pytest.approx(0.0)


def test_explain_single_prediction_sorts_by_absolute_contribution():
    feature_names = ["a", "b", "c"]
    shap_values = np.array([[0.1, -0.9, 0.3]])
    X_transformed = np.array([[10.0, 20.0, 30.0]])

    result = explain_single_prediction(shap_values, feature_names, row_idx=0, X_transformed=X_transformed)

    assert list(result["feature"]) == ["b", "c", "a"]
    # feature_value must stay correctly paired with its own feature after sorting
    b_row = result[result["feature"] == "b"].iloc[0]
    assert b_row["feature_value"] == 20.0
    assert b_row["shap_value"] == -0.9


@pytest.fixture
def synthetic_panel() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    n = 2000
    dominant = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)
    prob = 1 / (1 + np.exp(-3 * dominant))
    label = rng.binomial(1, prob)
    return pd.DataFrame(
        {
            "dominant_feature": dominant,
            "noise_feature": noise,
            "sector": rng.choice(["IT", "Banking"], size=n),
            "date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "label": label,
            "label_end_date": pd.date_range("2020-01-01", periods=n, freq="D") + pd.Timedelta(days=5),
        }
    )


def test_shap_values_correctly_identify_the_dominant_feature(synthetic_panel):
    """End-to-end check against a known ground truth: fit a real LightGBM
    on data where one feature genuinely drives the label and one is pure
    noise, compute real SHAP values via TreeExplainer, and verify the
    dominant feature actually ranks first — validates the whole
    pipeline -> transform -> TreeExplainer -> importance chain, not just
    that we called the library."""
    numeric_columns = ["dominant_feature", "noise_feature"]
    pipeline = build_lgbm_pipeline(numeric_columns, n_estimators=50)
    pipeline.fit(synthetic_panel[numeric_columns + ["sector"]], synthetic_panel["label"])

    X_transformed, feature_names = transformed_feature_matrix(pipeline, synthetic_panel)
    assert X_transformed.shape[1] == len(feature_names)

    shap_values = compute_shap_values(pipeline, X_transformed)
    assert shap_values.shape == X_transformed.shape

    importance = global_feature_importance(shap_values, feature_names)
    assert importance.iloc[0]["feature"] == "numeric__dominant_feature"
