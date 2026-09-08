"""SHAP-based explainability for the tuned LightGBM model.

Walk-forward evaluation (Phases 4-6) trains a fresh model per outer fold —
correct for validating methodology, but there's no single model there to
point explainability at. This module fits one "final" production model on
*all* available historical data using the Phase 5 tuned hyperparameters —
the model that would actually serve a live prediction — and explains that.

shap.TreeExplainer is used rather than a model-agnostic explainer
(KernelExplainer etc.): for tree ensembles it computes exact Shapley
values efficiently from the tree structure (Lundberg's TreeSHAP), not a
slow sampling-based approximation, so there's no real tradeoff to weigh.
"""

import logging

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from app.data.paths import RESULTS_DATA_DIR
from app.models.baseline import CATEGORICAL_COLUMNS, feature_columns
from app.models.dataset import load_modeling_dataset
from app.models.gbm import build_lgbm_pipeline, tune_lgbm_hyperparameters

logger = logging.getLogger(__name__)


def fit_final_model(horizon: int, dev_initial_train_end: str, dev_cutoff: str) -> tuple[Pipeline, pd.DataFrame]:
    """Tunes hyperparameters the same way Phase 5 did (on an isolated
    pre-2014 dev split, never touching the data this final model is fit
    on), then fits one pipeline on the *entire* available panel. This is
    deliberately not walk-forward evaluated — it's the production model,
    not a methodology check."""
    panel = load_modeling_dataset(horizon=horizon)
    best_params = tune_lgbm_hyperparameters(panel, dev_initial_train_end, dev_cutoff)

    numeric_columns = feature_columns(panel)
    pipeline = build_lgbm_pipeline(numeric_columns, **best_params)
    pipeline.fit(panel[numeric_columns + CATEGORICAL_COLUMNS], panel["label"])
    return pipeline, panel


def transformed_feature_matrix(pipeline: Pipeline, panel: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Runs the panel through the pipeline's preprocessing step only,
    returning the matrix SHAP needs (post-scaling, post-one-hot) alongside
    correctly ordered feature names — get_feature_names_out() is what
    keeps SHAP values from being silently mislabeled against the wrong
    column after one-hot encoding expands `sector` into several columns."""
    numeric_columns = feature_columns(panel)
    preprocessor = pipeline.named_steps["preprocess"]
    X_transformed = preprocessor.transform(panel[numeric_columns + CATEGORICAL_COLUMNS])
    feature_names = list(preprocessor.get_feature_names_out())
    return X_transformed, feature_names


def compute_shap_values(pipeline: Pipeline, X_transformed: np.ndarray) -> np.ndarray:
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    shap_values = explainer.shap_values(X_transformed)
    # Older/newer shap+lightgbm combinations differ on returning a single
    # array (positive-class contributions) vs a [neg, pos] list for binary
    # classification — normalize to the positive-class array either way.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    return shap_values


def global_feature_importance(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Mean absolute SHAP value per feature — "how much does this feature
    move predictions, on average, in either direction" — sorted
    descending. This is the model's own account of what matters, in
    contrast to Phase 4d's univariate mutual information, which turned out
    to be misled by a duplicated macro feature; SHAP is computed from the
    actual fitted model's real predictions, not a standalone statistic."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def explain_single_prediction(
    shap_values: np.ndarray, feature_names: list[str], row_idx: int, X_transformed: np.ndarray
) -> pd.DataFrame:
    """Per-feature contribution for one row, sorted by |contribution| —
    the "why did THIS row get THIS probability" view."""
    row_shap = shap_values[row_idx]
    row_values = X_transformed[row_idx]
    return (
        pd.DataFrame({"feature": feature_names, "feature_value": row_values, "shap_value": row_shap})
        .reindex(np.abs(row_shap).argsort()[::-1])
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    RESULTS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for horizon in (5, 20):
        print(f"\n=== Explainability — Horizon {horizon}d ===")
        pipeline, panel = fit_final_model(horizon, dev_initial_train_end="2011-12-31", dev_cutoff="2014-01-01")
        X_transformed, feature_names = transformed_feature_matrix(pipeline, panel)
        shap_values = compute_shap_values(pipeline, X_transformed)

        importance = global_feature_importance(shap_values, feature_names)
        importance.to_csv(RESULTS_DATA_DIR / f"shap_global_importance_{horizon}d.csv", index=False)
        print("Top 15 features by mean |SHAP|:")
        print(importance.head(15).to_string(index=False))

        shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False, max_display=15)
        plt.gcf().set_size_inches(9, 7)
        plt.tight_layout()
        plt.savefig(RESULTS_DATA_DIR / f"shap_summary_{horizon}d.png", dpi=150)
        plt.close()

        # Example individual explanations: today's highest- and lowest-
        # probability predictions, showing which features drove each one.
        numeric_columns = feature_columns(panel)
        y_prob = pipeline.predict_proba(panel[numeric_columns + CATEGORICAL_COLUMNS])[:, 1]
        panel_with_prob = panel.assign(y_prob=y_prob)
        latest_date = panel_with_prob["date"].max()
        latest_rows = panel_with_prob[panel_with_prob["date"] == latest_date].sort_values(
            "y_prob", ascending=False
        )

        example_frames = []
        for label, row in [("highest_prob", latest_rows.iloc[0]), ("lowest_prob", latest_rows.iloc[-1])]:
            explanation = explain_single_prediction(shap_values, feature_names, row.name, X_transformed).head(10)
            explanation["example"] = label
            explanation["ticker"] = row["ticker"]
            explanation["date"] = row["date"]
            explanation["predicted_prob"] = row["y_prob"]
            example_frames.append(explanation)
            print(f"\n{label}: {row['ticker']} on {row['date'].date()}, predicted P={row['y_prob']:.3f}")
            print(explanation[["feature", "feature_value", "shap_value"]].to_string(index=False))

        pd.concat(example_frames, ignore_index=True).to_csv(
            RESULTS_DATA_DIR / f"shap_example_explanations_{horizon}d.csv", index=False
        )
