"""Tuned LightGBM model: Phase 5's answer to whether a properly-tuned
nonlinear model beats the Phase 4 baseline on the real (k=1.5) target.

Hyperparameters are tuned once on an isolated validation split entirely
before the outer walk-forward folds' test period (see
tune_lgbm_hyperparameters), then applied as fixed values across all outer
folds — tuning separately per fold would be expensive and isn't standard
practice. The one honest limitation: tuning on a single validation split
(not nested CV) risks mildly overfitting the hyperparameter choice to that
period's quirks; a reasonable simplification for this project's scope, but
worth stating rather than hiding.
"""

import itertools
import logging

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.data.paths import RESULTS_DATA_DIR
from app.models.baseline import CATEGORICAL_COLUMNS, feature_columns, run_walk_forward_evaluation
from app.models.cv import purged_embargoed_walk_forward_splits
from app.models.dataset import load_modeling_dataset
from app.models.evaluate import evaluate_predictions, summarize_folds

logger = logging.getLogger(__name__)

# A small, explicit grid rather than an exhaustive search — this project's
# goal is a defensible, reasonably-tuned model, not squeezing out the last
# 0.001 of log-loss. Kept deliberately modest so tuning stays fast.
DEFAULT_PARAM_GRID = {
    "num_leaves": [15, 31],
    "learning_rate": [0.05, 0.1],
    "min_child_samples": [20, 50],
    "n_estimators": [100, 300],
    "reg_lambda": [0.0, 1.0],
}


def build_lgbm_pipeline(numeric_columns: list[str], **lgbm_params) -> Pipeline:
    """Same preprocessing shape as the logistic regression baseline and
    the Phase 4d diagnostic's GBM pipeline — keeping it identical means
    any performance difference is attributable to the model/tuning, not
    preprocessing choices."""
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric_columns),
            ("sector", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )
    model = LGBMClassifier(random_state=0, verbose=-1, **lgbm_params)
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def _param_combinations(param_grid: dict[str, list]) -> list[dict]:
    keys = list(param_grid.keys())
    return [dict(zip(keys, values)) for values in itertools.product(*param_grid.values())]


def tune_lgbm_hyperparameters(
    panel: pd.DataFrame,
    dev_initial_train_end: str,
    dev_cutoff: str,
    param_grid: dict[str, list] = DEFAULT_PARAM_GRID,
) -> dict:
    """Tunes on ONE validation fold carved from data strictly before
    `dev_cutoff` — callers must ensure `dev_cutoff` is before the earliest
    outer walk-forward test year, so tuning never sees outer-fold test
    data. Selects by log-loss (matching this project's actual evaluation
    goal — not accuracy, not AUC)."""
    dev_panel = panel[panel["date"] < pd.Timestamp(dev_cutoff)]
    dev_folds = purged_embargoed_walk_forward_splits(
        dev_panel, initial_train_end=dev_initial_train_end, embargo_days=5
    )
    if not dev_folds:
        raise ValueError("No dev fold produced — check dev_initial_train_end/dev_cutoff bounds")
    dev_fold = dev_folds[0]

    numeric_columns = feature_columns(panel)
    train = dev_panel.loc[dev_fold.train_idx]
    valid = dev_panel.loc[dev_fold.test_idx]

    best_params, best_log_loss = None, float("inf")
    for params in _param_combinations(param_grid):
        pipeline = build_lgbm_pipeline(numeric_columns, **params)
        pipeline.fit(train[numeric_columns + CATEGORICAL_COLUMNS], train["label"])
        y_prob = pipeline.predict_proba(valid[numeric_columns + CATEGORICAL_COLUMNS])[:, 1]
        log_loss = evaluate_predictions(valid["label"].to_numpy(), y_prob)["log_loss"]
        logger.info("params=%s -> dev log_loss=%.4f", params, log_loss)
        if log_loss < best_log_loss:
            best_params, best_log_loss = params, log_loss

    logger.info("Selected params=%s (dev log_loss=%.4f)", best_params, best_log_loss)
    return best_params


def run_tuned_gbm_for_horizon(
    horizon: int,
    dev_initial_train_end: str,
    dev_cutoff: str,
    outer_initial_train_end: str,
    embargo_days: int,
) -> tuple[dict, pd.DataFrame]:
    panel = load_modeling_dataset(horizon=horizon)
    best_params = tune_lgbm_hyperparameters(panel, dev_initial_train_end, dev_cutoff)

    def _tuned_pipeline_builder(numeric_columns: list[str]) -> Pipeline:
        return build_lgbm_pipeline(numeric_columns, **best_params)

    folds = purged_embargoed_walk_forward_splits(
        panel, initial_train_end=outer_initial_train_end, embargo_days=embargo_days
    )
    fold_metrics = run_walk_forward_evaluation(panel, folds, _tuned_pipeline_builder)
    return best_params, summarize_folds(fold_metrics)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    RESULTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for horizon in (5, 20):
        print(f"\n=== Tuned LightGBM — Horizon {horizon}d ===")
        best_params, summary = run_tuned_gbm_for_horizon(
            horizon,
            dev_initial_train_end="2011-12-31",
            dev_cutoff="2014-01-01",
            outer_initial_train_end="2013-12-31",
            embargo_days=horizon,
        )
        print("Tuned hyperparameters:", best_params)
        summary.to_csv(RESULTS_DATA_DIR / f"gbm_tuned_{horizon}d.csv", index=False)
        print(summary.to_string(index=False))
        print("\nMean across folds:")
        print(summary[["brier", "naive_brier", "log_loss", "naive_log_loss", "roc_auc"]].mean().to_string())
