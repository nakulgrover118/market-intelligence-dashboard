import numpy as np
import pandas as pd
import pytest

from app.models.cv import Fold
from app.models.diagnostics import (
    compare_logreg_vs_gbm,
    feature_label_correlations,
    feature_mutual_information,
)


@pytest.fixture
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(9)
    n = 1000
    linear_signal = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)
    # nonlinear_signal predicts the label only in its extreme tails —
    # a linear correlation should mostly miss this, mutual information
    # should pick it up.
    nonlinear_signal = rng.normal(0, 1, n)
    label = (linear_signal + 0.5 * (np.abs(nonlinear_signal) > 1.5).astype(float) > 0.3).astype(int)
    return pd.DataFrame(
        {
            "linear_signal": linear_signal,
            "noise": noise,
            "nonlinear_signal": nonlinear_signal,
            "sector": rng.choice(["IT", "Banking"], size=n),
            "date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "label": label,
            "label_end_date": pd.date_range("2020-01-01", periods=n, freq="D") + pd.Timedelta(days=5),
        }
    )


def test_correlations_rank_linear_signal_above_noise(panel):
    corr = feature_label_correlations(panel)
    assert abs(corr["linear_signal"]) > abs(corr["noise"])


def test_correlations_are_sorted_by_absolute_value(panel):
    corr = feature_label_correlations(panel)
    abs_values = corr.abs().to_numpy()
    assert (abs_values[:-1] >= abs_values[1:]).all()


def test_mutual_information_is_nonnegative(panel):
    mi = feature_mutual_information(panel)
    assert (mi >= 0).all()


def test_mutual_information_detects_noise_as_low(panel):
    mi = feature_mutual_information(panel)
    # noise carries no information about the label by construction; it
    # shouldn't rank above both real signal columns.
    assert mi["noise"] < mi["linear_signal"] or mi["noise"] < mi["nonlinear_signal"]


def test_compare_logreg_vs_gbm_returns_both_models(panel):
    split = len(panel) // 2
    fold = Fold(name="test", train_idx=panel.index[:split], test_idx=panel.index[split:])
    results = compare_logreg_vs_gbm(panel, [fold])
    assert set(results["model"]) == {"logreg", "gbm"}
    assert len(results) == 2
