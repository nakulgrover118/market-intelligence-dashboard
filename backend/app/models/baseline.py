"""Logistic regression baseline, fit and evaluated per walk-forward fold.

This is deliberately the simplest model in the project. Its job isn't to
be good — it's to set the floor that Phase 5's gradient-boosted model has
to beat before that added complexity is justified. Every fold refits the
pipeline (scaler + encoder + model) from scratch on that fold's training
data only, so no statistic (mean, std, category set) is ever computed
using data from the future relative to that fold's test period.
"""

import logging
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.data.paths import RESULTS_DATA_DIR
from app.models.cv import Fold, purged_embargoed_walk_forward_splits
from app.models.dataset import load_modeling_dataset
from app.models.evaluate import FoldMetrics, build_fold_metrics, summarize_folds

logger = logging.getLogger(__name__)

NON_FEATURE_COLUMNS = {"ticker", "sector", "date", "label", "label_end_date"}
CATEGORICAL_COLUMNS = ["sector"]
PipelineBuilder = Callable[[list[str]], Pipeline]


def feature_columns(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns if c not in NON_FEATURE_COLUMNS]


def build_pipeline(numeric_columns: list[str]) -> Pipeline:
    """No class_weight='balanced': that would reweight the loss to help
    accuracy on the minority class, but at the cost of distorting the
    predicted probabilities away from their true calibration — and
    calibrated probabilities are the entire point of this project (see
    docs/roadmap.md). Fit on the natural class distribution; address
    calibration explicitly in Phase 6 if needed."""
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric_columns),
            ("sector", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )
    return Pipeline([("preprocess", preprocessor), ("model", LogisticRegression(max_iter=1000))])


def run_walk_forward_predictions(
    panel: pd.DataFrame, folds: list[Fold], pipeline_builder: PipelineBuilder = build_pipeline
) -> pd.DataFrame:
    """Row-level out-of-sample predictions for every fold, concatenated in
    fold order. `pipeline_builder` is pluggable so the exact same
    walk-forward protocol (same folds, same preprocessing shape) can be
    reused to compare model types fairly — see app/models/diagnostics.py
    (nonlinear vs linear) and app/models/calibration.py, which chains
    these row-level predictions across folds to calibrate each fold using
    the previous fold's genuine out-of-sample track record."""
    numeric_columns = feature_columns(panel)
    frames = []
    for fold in folds:
        train = panel.loc[fold.train_idx]
        test = panel.loc[fold.test_idx]

        pipeline = pipeline_builder(numeric_columns)
        pipeline.fit(train[numeric_columns + CATEGORICAL_COLUMNS], train["label"])
        y_prob = pipeline.predict_proba(test[numeric_columns + CATEGORICAL_COLUMNS])[:, 1]

        frames.append(
            pd.DataFrame(
                {"fold": fold.name, "date": test["date"].to_numpy(), "y_true": test["label"].to_numpy(), "y_prob": y_prob},
                index=test.index,
            )
        )
    return pd.concat(frames)


def run_walk_forward_evaluation(
    panel: pd.DataFrame, folds: list[Fold], pipeline_builder: PipelineBuilder = build_pipeline
) -> list[FoldMetrics]:
    predictions = run_walk_forward_predictions(panel, folds, pipeline_builder)
    results = []
    for fold in folds:
        fold_predictions = predictions[predictions["fold"] == fold.name]
        train_labels = panel.loc[fold.train_idx, "label"].to_numpy()
        fm = build_fold_metrics(
            fold.name, train_labels, fold_predictions["y_true"].to_numpy(), fold_predictions["y_prob"].to_numpy()
        )
        results.append(fm)
        logger.info(
            "fold %s: n_train=%d n_test=%d brier=%.4f (naive %.4f) log_loss=%.4f (naive %.4f) roc_auc=%.3f",
            fm.fold, fm.n_train, fm.n_test, fm.brier, fm.naive_brier, fm.log_loss, fm.naive_log_loss, fm.roc_auc,
        )
    return results


def run_baseline_for_horizon(horizon: int, initial_train_end: str, embargo_days: int) -> pd.DataFrame:
    panel = load_modeling_dataset(horizon=horizon)
    folds = purged_embargoed_walk_forward_splits(
        panel, initial_train_end=initial_train_end, embargo_days=embargo_days
    )
    fold_metrics = run_walk_forward_evaluation(panel, folds)
    return summarize_folds(fold_metrics)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    RESULTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for horizon in (5, 20):
        print(f"\n=== Horizon {horizon}d ===")
        summary = run_baseline_for_horizon(horizon, initial_train_end="2013-12-31", embargo_days=horizon)
        summary.to_csv(RESULTS_DATA_DIR / f"baseline_logreg_{horizon}d.csv", index=False)
        print(summary.to_string(index=False))
        print("\nMean across folds:")
        print(summary[["brier", "naive_brier", "log_loss", "naive_log_loss", "roc_auc"]].mean().to_string())
