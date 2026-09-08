import numpy as np
import pandas as pd
import pytest

from app.models.calibration import (
    apply_isotonic_calibrator,
    apply_platt_calibrator,
    chain_calibrate,
    evaluate_calibration_methods,
    expected_calibration_error,
    fit_isotonic_calibrator,
    fit_platt_calibrator,
    plot_reliability_diagram,
    reliability_table,
)
from app.models.evaluate import evaluate_predictions


def _miscalibrated_data(n=4000, seed=1):
    """True probabilities from a logistic model, but the "raw score" the
    model reports is badly overconfident (squared), a classic miscalibration
    shape. Split into a fit set and a fresh held-out set so calibration
    quality is judged out-of-sample, not by overfitting the same data."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    true_prob = 1 / (1 + np.exp(-x))
    y = rng.binomial(1, true_prob)
    overconfident_score = true_prob**3  # pushes toward 0/1, badly miscalibrated
    half = n // 2
    return (
        y[:half], overconfident_score[:half],  # fit set
        y[half:], overconfident_score[half:],  # held-out set
    )


def test_platt_calibration_improves_brier_on_held_out_data():
    y_fit, score_fit, y_test, score_test = _miscalibrated_data()
    calibrator = fit_platt_calibrator(y_fit, score_fit)
    calibrated = apply_platt_calibrator(calibrator, score_test)

    raw_brier = evaluate_predictions(y_test, score_test)["brier"]
    calibrated_brier = evaluate_predictions(y_test, calibrated)["brier"]
    assert calibrated_brier < raw_brier


def test_isotonic_calibration_improves_brier_on_held_out_data():
    y_fit, score_fit, y_test, score_test = _miscalibrated_data()
    calibrator = fit_isotonic_calibrator(y_fit, score_fit)
    calibrated = apply_isotonic_calibrator(calibrator, score_test)

    raw_brier = evaluate_predictions(y_test, score_test)["brier"]
    calibrated_brier = evaluate_predictions(y_test, calibrated)["brier"]
    assert calibrated_brier < raw_brier


def test_calibration_does_not_change_ranking_ability():
    """Both calibrators are monotonic transforms, so ROC-AUC (a purely
    rank-based metric) must be unchanged by calibration — a real change
    here would indicate a bug (e.g. a non-monotonic mapping)."""
    y_fit, score_fit, y_test, score_test = _miscalibrated_data()
    calibrator = fit_platt_calibrator(y_fit, score_fit)
    calibrated = apply_platt_calibrator(calibrator, score_test)

    raw_auc = evaluate_predictions(y_test, score_test)["roc_auc"]
    calibrated_auc = evaluate_predictions(y_test, calibrated)["roc_auc"]
    assert calibrated_auc == pytest.approx(raw_auc, abs=1e-9)


@pytest.fixture
def biased_fold_predictions() -> pd.DataFrame:
    """Two folds, both with the SAME systematic overconfidence bias
    (raw score = true_prob^3). Fold 1 is used to learn the correction;
    fold 2's raw scores carry the identical bias, so a correct chain
    calibration should pull fold 2's calibrated probabilities much closer
    to its true probabilities than its raw scores were."""
    rng = np.random.default_rng(7)

    def _fold(n, fold_name, start_date):
        x = rng.normal(0, 1, n)
        true_prob = 1 / (1 + np.exp(-x))
        y = rng.binomial(1, true_prob)
        raw = true_prob**3
        dates = pd.date_range(start_date, periods=n, freq="D")
        return pd.DataFrame({"fold": fold_name, "date": dates, "y_true": y, "y_prob": raw, "true_prob": true_prob})

    fold1 = _fold(2000, "2020", "2020-01-01")
    fold2 = _fold(2000, "2021", "2021-01-01")
    return pd.concat([fold1, fold2], ignore_index=True)


def test_chain_calibrate_first_fold_is_unchanged(biased_fold_predictions):
    result = chain_calibrate(biased_fold_predictions, method="sigmoid")
    fold1 = result[result["fold"] == "2020"]
    np.testing.assert_array_equal(fold1["y_prob_calibrated"].to_numpy(), fold1["y_prob"].to_numpy())


def test_chain_calibrate_corrects_bias_using_prior_fold(biased_fold_predictions):
    result = chain_calibrate(biased_fold_predictions, method="sigmoid")
    fold2 = result[result["fold"] == "2021"]

    raw_brier = evaluate_predictions(fold2["y_true"].to_numpy(), fold2["y_prob"].to_numpy())["brier"]
    calibrated_brier = evaluate_predictions(fold2["y_true"].to_numpy(), fold2["y_prob_calibrated"].to_numpy())["brier"]
    assert calibrated_brier < raw_brier


def test_chain_calibrate_expanding_window_uses_all_prior_folds():
    """With window='expanding', fold 3's calibrator must be fit on folds
    1+2 pooled, not just fold 2 — verified by checking that a bias only
    present in fold 1 (absent from fold 2) still gets corrected in fold 3
    when using 'expanding', but would NOT be as fully corrected with
    'last' (which only sees fold 2, unbiased, and so applies no
    correction)."""
    rng = np.random.default_rng(11)

    def _fold(n, fold_name, start_date, biased: bool):
        x = rng.normal(0, 1, n)
        true_prob = 1 / (1 + np.exp(-x))
        y = rng.binomial(1, true_prob)
        raw = true_prob**3 if biased else true_prob  # biased folds are overconfident
        dates = pd.date_range(start_date, periods=n, freq="D")
        return pd.DataFrame({"fold": fold_name, "date": dates, "y_true": y, "y_prob": raw})

    # fold1: biased: fold2: unbiased; fold3: biased again (like fold1)
    fold1 = _fold(3000, "f1", "2020-01-01", biased=True)
    fold2 = _fold(3000, "f2", "2021-01-01", biased=False)
    fold3 = _fold(3000, "f3", "2022-01-01", biased=True)
    predictions = pd.concat([fold1, fold2, fold3], ignore_index=True)

    last_window = chain_calibrate(predictions, "sigmoid", window="last")
    expanding_window = chain_calibrate(predictions, "sigmoid", window="expanding")

    f3_true = predictions[predictions["fold"] == "f3"]["y_true"].to_numpy()
    f3_last = last_window[last_window["fold"] == "f3"]["y_prob_calibrated"].to_numpy()
    f3_expanding = expanding_window[expanding_window["fold"] == "f3"]["y_prob_calibrated"].to_numpy()

    last_brier = evaluate_predictions(f3_true, f3_last)["brier"]
    expanding_brier = evaluate_predictions(f3_true, f3_expanding)["brier"]
    # 'last' only saw fold2 (unbiased) so it under-corrects fold3's bias;
    # 'expanding' also saw fold1 (biased, like fold3) and should correct
    # fold3's actual bias better.
    assert expanding_brier < last_brier


def test_reliability_table_matches_manual_calc():
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    table = reliability_table(y_true, y_prob, n_bins=2, strategy="quantile")

    assert table["count"].sum() == 8
    # low-prob bin (first 4 values): mean_predicted = mean(0.1,0.2,0.3,0.4)=0.25, mean_observed = mean(0,0,1,1)=0.5
    low_bin = table.iloc[0]
    assert low_bin["mean_predicted"] == pytest.approx(0.25)
    assert low_bin["mean_observed"] == pytest.approx(0.5)


def test_ece_is_near_zero_for_perfectly_calibrated_predictions():
    rng = np.random.default_rng(3)
    n = 5000
    true_prob = rng.uniform(0.05, 0.95, n)
    y_true = rng.binomial(1, true_prob)
    ece = expected_calibration_error(y_true, true_prob, n_bins=10)
    assert ece < 0.03


def test_evaluate_calibration_methods_excludes_first_fold(biased_fold_predictions):
    result = evaluate_calibration_methods(biased_fold_predictions, methods=("sigmoid", "isotonic"))
    assert "2020" not in set(result["fold"])
    assert set(result["fold"]) == {"2021"}
    assert set(result["variant"]) == {"raw", "sigmoid", "isotonic"}


def test_evaluate_calibration_methods_shows_calibrated_beats_raw_brier(biased_fold_predictions):
    result = evaluate_calibration_methods(biased_fold_predictions, methods=("sigmoid",))
    raw_brier = result[result["variant"] == "raw"]["brier"].iloc[0]
    sigmoid_brier = result[result["variant"] == "sigmoid"]["brier"].iloc[0]
    assert sigmoid_brier < raw_brier


def test_plot_reliability_diagram_writes_a_file(biased_fold_predictions, tmp_path):
    out_path = tmp_path / "reliability.png"
    plot_reliability_diagram(biased_fold_predictions, ("sigmoid",), out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_ece_is_large_for_badly_miscalibrated_predictions():
    rng = np.random.default_rng(3)
    n = 5000
    true_prob = rng.uniform(0.05, 0.95, n)
    y_true = rng.binomial(1, true_prob)
    # Systematically overconfident but still varies with true_prob (a
    # constant prediction would trivially look "calibrated in aggregate"
    # even with zero discrimination, since ECE bins by predicted value —
    # with only one distinct value there's nothing for it to catch).
    badly_miscalibrated = np.clip(true_prob * 2.5, 0, 1)
    ece = expected_calibration_error(y_true, badly_miscalibrated, n_bins=10)
    assert ece > 0.1
