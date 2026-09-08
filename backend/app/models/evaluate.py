"""Probability-quality metrics — deliberately not accuracy.

This project predicts probabilities, not classification decisions, so the
metrics that matter are ones that score the probability itself: Brier
score (mean squared error between predicted probability and the 0/1
outcome) and log-loss both reward being confidently right and punish being
confidently wrong; ROC-AUC checks whether the model has any discriminative
signal at all, independent of whether its probabilities are calibrated.
None of these is "did we call it 1465/2000 times" — accuracy is not
reported anywhere in this module on purpose.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


@dataclass
class FoldMetrics:
    fold: str
    n_train: int
    n_test: int
    train_base_rate: float
    test_base_rate: float
    brier: float
    log_loss: float
    roc_auc: float
    naive_brier: float  # predicting train_base_rate for every test row
    naive_log_loss: float


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """ROC-AUC is undefined when the test set has only one class present
    (no positive/negative pair to rank) — returns NaN rather than raising,
    since that's a property of a specific (usually small) fold, not an
    error in the pipeline."""
    metrics = {
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
    }
    if len(np.unique(y_true)) < 2:
        metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def naive_baseline_metrics(y_true: np.ndarray, train_base_rate: float) -> dict[str, float]:
    """The floor any model must beat: predicting the training set's
    historical positive rate for every single test row, with no other
    information. A model that can't beat this has learned nothing."""
    y_prob = np.full(len(y_true), train_base_rate)
    return {
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
    }


def build_fold_metrics(
    fold_name: str, y_train: np.ndarray, y_test: np.ndarray, y_prob: np.ndarray
) -> FoldMetrics:
    train_base_rate = float(np.mean(y_train))
    model_metrics = evaluate_predictions(y_test, y_prob)
    naive_metrics = naive_baseline_metrics(y_test, train_base_rate)
    return FoldMetrics(
        fold=fold_name,
        n_train=len(y_train),
        n_test=len(y_test),
        train_base_rate=train_base_rate,
        test_base_rate=float(np.mean(y_test)),
        brier=model_metrics["brier"],
        log_loss=model_metrics["log_loss"],
        roc_auc=model_metrics["roc_auc"],
        naive_brier=naive_metrics["brier"],
        naive_log_loss=naive_metrics["log_loss"],
    )


def summarize_folds(fold_metrics: list[FoldMetrics]) -> pd.DataFrame:
    return pd.DataFrame([vars(fm) for fm in fold_metrics])
