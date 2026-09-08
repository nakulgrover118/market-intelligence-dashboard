import numpy as np
import pytest

from app.models.evaluate import (
    build_fold_metrics,
    evaluate_predictions,
    naive_baseline_metrics,
    summarize_folds,
)


def test_perfect_predictions_have_near_zero_brier_and_log_loss():
    y_true = np.array([0, 1, 0, 1, 1])
    y_prob = y_true.astype(float)
    metrics = evaluate_predictions(y_true, y_prob)
    assert metrics["brier"] < 1e-6
    assert metrics["log_loss"] < 0.01


def test_constant_half_prediction_matches_theoretical_values():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_prob = np.full(len(y_true), 0.5)
    metrics = evaluate_predictions(y_true, y_prob)
    assert metrics["brier"] == pytest.approx(0.25)
    assert metrics["log_loss"] == pytest.approx(np.log(2))


def test_roc_auc_is_nan_when_only_one_class_present():
    y_true = np.array([1, 1, 1, 1])
    y_prob = np.array([0.6, 0.7, 0.4, 0.9])
    metrics = evaluate_predictions(y_true, y_prob)
    assert np.isnan(metrics["roc_auc"])


def test_roc_auc_perfect_separation():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    metrics = evaluate_predictions(y_true, y_prob)
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_naive_baseline_brier_matches_manual_formula():
    y_true = np.array([0, 1, 1, 0, 1])
    train_base_rate = 0.6
    metrics = naive_baseline_metrics(y_true, train_base_rate)
    expected_brier = np.mean((train_base_rate - y_true) ** 2)
    assert metrics["brier"] == pytest.approx(expected_brier)


def test_build_fold_metrics_fields():
    y_train = np.array([0, 1, 1, 0, 1, 0])
    y_test = np.array([0, 1, 1, 0])
    y_prob = np.array([0.2, 0.8, 0.6, 0.3])

    fm = build_fold_metrics("2020", y_train, y_test, y_prob)

    assert fm.fold == "2020"
    assert fm.n_train == 6
    assert fm.n_test == 4
    assert fm.train_base_rate == pytest.approx(0.5)
    assert fm.test_base_rate == pytest.approx(0.5)
    assert fm.roc_auc == pytest.approx(1.0)
    # The model's predictions are more confident/correct than the naive
    # base-rate guess here, so it should score a lower (better) Brier.
    assert fm.brier < fm.naive_brier


def test_summarize_folds_produces_one_row_per_fold():
    y_train = np.array([0, 1, 1, 0])
    y_test = np.array([0, 1])
    y_prob = np.array([0.3, 0.7])
    fold_metrics = [
        build_fold_metrics("2020", y_train, y_test, y_prob),
        build_fold_metrics("2021", y_train, y_test, y_prob),
    ]
    df = summarize_folds(fold_metrics)
    assert len(df) == 2
    assert list(df["fold"]) == ["2020", "2021"]
    assert "brier" in df.columns
