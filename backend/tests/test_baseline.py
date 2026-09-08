import numpy as np
import pandas as pd
import pytest

from app.models.baseline import build_pipeline, feature_columns, run_walk_forward_evaluation
from app.models.cv import Fold


def test_feature_columns_excludes_metadata():
    panel = pd.DataFrame(
        {
            "rsi_14": [1.0], "log_return_1d": [0.1],
            "ticker": ["A"], "sector": ["IT"], "date": [pd.Timestamp("2024-01-01")],
            "label": [1], "label_end_date": [pd.Timestamp("2024-01-10")],
        }
    )
    cols = feature_columns(panel)
    assert set(cols) == {"rsi_14", "log_return_1d"}


@pytest.fixture
def synthetic_panel() -> pd.DataFrame:
    """A pooled-panel-shaped dataset where `signal` is strongly predictive
    of `label` and `noise` carries no information, plus a sector column —
    close enough to the real panel's shape to exercise the full pipeline
    (scaling + one-hot + logistic regression) meaningfully."""
    rng = np.random.default_rng(5)
    n = 2000
    signal = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)
    # label depends strongly on `signal`, not at all on `noise`
    prob = 1 / (1 + np.exp(-3 * signal))
    label = rng.binomial(1, prob)
    dates = pd.date_range("2015-01-01", periods=n, freq="D")
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

    numeric_cols = ["signal", "noise"]
    pipeline = build_pipeline(numeric_cols)
    pipeline.fit(train[numeric_cols + ["sector"]], train["label"])
    y_prob = pipeline.predict_proba(test[numeric_cols + ["sector"]])[:, 1]

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(test["label"], y_prob)
    assert auc > 0.85  # signal is strongly predictive; pipeline should find it


def test_run_walk_forward_evaluation_end_to_end(synthetic_panel):
    n = len(synthetic_panel)
    split = n // 2
    fold = Fold(
        name="synthetic",
        train_idx=synthetic_panel.index[:split],
        test_idx=synthetic_panel.index[split:],
    )
    results = run_walk_forward_evaluation(synthetic_panel, [fold])

    assert len(results) == 1
    fm = results[0]
    assert fm.fold == "synthetic"
    assert fm.roc_auc > 0.85
    # The model should clearly beat the naive base-rate baseline, since
    # `signal` genuinely predicts the label.
    assert fm.brier < fm.naive_brier
    assert fm.log_loss < fm.naive_log_loss
