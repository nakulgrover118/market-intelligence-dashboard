import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from app.models.gbm import _param_combinations, build_lgbm_pipeline, tune_lgbm_hyperparameters


def test_param_combinations_produces_full_cartesian_product():
    grid = {"a": [1, 2], "b": [10, 20, 30]}
    combos = _param_combinations(grid)
    assert len(combos) == 6
    assert {"a": 1, "b": 10} in combos
    assert {"a": 2, "b": 30} in combos


@pytest.fixture
def synthetic_panel() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 3000
    signal = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)
    prob = 1 / (1 + np.exp(-2 * signal))
    label = rng.binomial(1, prob)
    dates = pd.date_range("2009-01-01", periods=n, freq="D")
    sectors = rng.choice(["IT", "Banking", "Energy"], size=n)
    return pd.DataFrame(
        {
            "signal": signal,
            "noise": noise,
            "sector": sectors,
            "date": dates,
            "label": label,
            "label_end_date": dates + pd.Timedelta(days=5),
        }
    )


def test_pipeline_learns_real_signal(synthetic_panel):
    split = len(synthetic_panel) // 2
    train = synthetic_panel.iloc[:split]
    test = synthetic_panel.iloc[split:]

    pipeline = build_lgbm_pipeline(["signal", "noise"], n_estimators=50)
    pipeline.fit(train[["signal", "noise", "sector"]], train["label"])
    y_prob = pipeline.predict_proba(test[["signal", "noise", "sector"]])[:, 1]

    assert roc_auc_score(test["label"], y_prob) > 0.8


def test_tune_selects_the_actual_best_candidate_by_log_loss(synthetic_panel):
    """Deterministic check: independently recompute dev log-loss for every
    candidate in a small grid and assert the function returns the true
    argmin, not just *some* member of the grid."""
    from app.models.baseline import CATEGORICAL_COLUMNS, feature_columns
    from app.models.cv import purged_embargoed_walk_forward_splits
    from app.models.evaluate import evaluate_predictions
    from app.models.gbm import _param_combinations

    grid = {"n_estimators": [10, 100], "num_leaves": [7, 31]}
    dev_cutoff = "2016-01-01"
    dev_panel = synthetic_panel[synthetic_panel["date"] < pd.Timestamp(dev_cutoff)]
    dev_initial_train_end = "2010-12-31"

    best_params = tune_lgbm_hyperparameters(
        synthetic_panel, dev_initial_train_end, dev_cutoff, param_grid=grid
    )

    fold = purged_embargoed_walk_forward_splits(dev_panel, dev_initial_train_end, embargo_days=5)[0]
    numeric_columns = feature_columns(synthetic_panel)
    train = dev_panel.loc[fold.train_idx]
    valid = dev_panel.loc[fold.test_idx]

    losses = {}
    for params in _param_combinations(grid):
        pipeline = build_lgbm_pipeline(numeric_columns, **params)
        pipeline.fit(train[numeric_columns + CATEGORICAL_COLUMNS], train["label"])
        y_prob = pipeline.predict_proba(valid[numeric_columns + CATEGORICAL_COLUMNS])[:, 1]
        losses[tuple(sorted(params.items()))] = evaluate_predictions(
            valid["label"].to_numpy(), y_prob
        )["log_loss"]

    true_best = min(losses, key=losses.get)
    assert tuple(sorted(best_params.items())) == true_best
