"""Post-hoc probability calibration on top of a tuned model's raw scores.

Phase 5 showed the tuned LightGBM model has real ranking ability (ROC-AUC
well above 0.5) but its raw probabilities are only partially calibrated
(Brier/log-loss roughly tied with the naive baseline at the 20-day
horizon). This module fits a calibrator (Platt/sigmoid or isotonic) on top
of the raw scores and checks whether either closes that gap.

Calibration source: each outer walk-forward fold's calibrator is fit on
the PREVIOUS fold's genuine out-of-sample predictions (real held-out
predictions on a test period the model never trained on) — not on any of
the current fold's own training or test data. This mirrors how a real
system would calibrate using its most recent live track record, and it
requires no new train/test carving: it reuses predictions
run_walk_forward_predictions already produces. The first fold has no
prior out-of-sample history to calibrate from, so it's left uncalibrated
(y_prob_calibrated == y_prob) and excluded from calibrated-vs-raw
comparisons.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from app.data.paths import RESULTS_DATA_DIR
from app.models.baseline import run_walk_forward_predictions
from app.models.cv import purged_embargoed_walk_forward_splits
from app.models.dataset import load_modeling_dataset
from app.models.evaluate import evaluate_predictions
from app.models.gbm import build_lgbm_pipeline, tune_lgbm_hyperparameters

logger = logging.getLogger(__name__)


def fit_platt_calibrator(y_true: np.ndarray, y_prob: np.ndarray) -> LogisticRegression:
    """Platt/sigmoid scaling: a 1-D logistic regression on the raw score."""
    calibrator = LogisticRegression()
    calibrator.fit(y_prob.reshape(-1, 1), y_true)
    return calibrator


def apply_platt_calibrator(calibrator: LogisticRegression, y_prob: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]


def fit_isotonic_calibrator(y_true: np.ndarray, y_prob: np.ndarray) -> IsotonicRegression:
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(y_prob, y_true)
    return calibrator


def apply_isotonic_calibrator(calibrator: IsotonicRegression, y_prob: np.ndarray) -> np.ndarray:
    return calibrator.predict(y_prob)


_CALIBRATORS = {
    "sigmoid": (fit_platt_calibrator, apply_platt_calibrator),
    "isotonic": (fit_isotonic_calibrator, apply_isotonic_calibrator),
}


def chain_calibrate(predictions: pd.DataFrame, method: str, window: str = "last") -> pd.DataFrame:
    """`predictions` must have columns fold, date, y_true, y_prob, with
    rows already in chronological fold order (as produced by
    run_walk_forward_predictions). Returns a copy with an added
    `y_prob_calibrated` column.

    `window` controls how much prior history each fold's calibrator is fit
    on:
    - "last": only the immediately preceding fold (~6k rows here) — the
      most temporally "fresh" track record, but a small, potentially noisy
      sample for isotonic regression in particular.
    - "expanding": all prior folds pooled — thematically consistent with
      the expanding-window training used everywhere else in this project,
      and gives isotonic/sigmoid much more data to fit a stable curve on,
      at the cost of including older, potentially less representative
      out-of-sample history."""
    fit_fn, apply_fn = _CALIBRATORS[method]

    fold_order = list(dict.fromkeys(predictions["fold"]))
    out = predictions.copy()
    out["y_prob_calibrated"] = out["y_prob"]

    prior_data = None
    for fold_name in fold_order:
        current_mask = out["fold"] == fold_name
        if prior_data is not None:
            calibrator = fit_fn(prior_data["y_true"].to_numpy(), prior_data["y_prob"].to_numpy())
            out.loc[current_mask, "y_prob_calibrated"] = apply_fn(
                calibrator, out.loc[current_mask, "y_prob"].to_numpy()
            )
        current_data = out.loc[current_mask, ["y_true", "y_prob"]]
        if window == "expanding" and prior_data is not None:
            prior_data = pd.concat([prior_data, current_data])
        else:
            prior_data = current_data
    return out


def reliability_table(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10, strategy: str = "quantile"
) -> pd.DataFrame:
    """Quantile binning by default: with an imbalanced label (~9% positive)
    raw probabilities cluster in a narrow low range, so equal-width bins
    would leave most bins nearly empty. Quantile bins keep roughly equal
    sample counts per bin, which is what makes both the reliability curve
    and the ECE below trustworthy rather than dominated by a couple of
    sparse bins."""
    df = pd.DataFrame({"y_true": np.asarray(y_true), "y_prob": np.asarray(y_prob)})
    if strategy == "quantile":
        df["bin"] = pd.qcut(df["y_prob"], q=n_bins, duplicates="drop")
    else:
        df["bin"] = pd.cut(df["y_prob"], bins=n_bins)
    table = (
        df.groupby("bin", observed=True)
        .agg(mean_predicted=("y_prob", "mean"), mean_observed=("y_true", "mean"), count=("y_true", "size"))
        .reset_index(drop=True)
    )
    return table


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10, strategy: str = "quantile"
) -> float:
    """Weighted average of |predicted - observed| across bins, weighted by
    bin population share. 0 = perfectly calibrated."""
    table = reliability_table(y_true, y_prob, n_bins, strategy)
    n = table["count"].sum()
    return float((table["count"] / n * (table["mean_predicted"] - table["mean_observed"]).abs()).sum())


def evaluate_calibration_methods(
    predictions: pd.DataFrame, methods: tuple[str, ...] = ("sigmoid", "isotonic"), window: str = "last"
) -> pd.DataFrame:
    """Per-fold comparison of raw vs each calibration method's Brier/
    log-loss/ROC-AUC/ECE, excluding the first fold (it has no prior fold
    to calibrate from, so it's identical under every variant). `window`
    is passed through to chain_calibrate — see its docstring."""
    fold_order = list(dict.fromkeys(predictions["fold"]))
    comparable_folds = fold_order[1:]

    variants = {"raw": predictions}
    for method in methods:
        variants[method] = chain_calibrate(predictions, method, window=window)

    rows = []
    for variant_name, df in variants.items():
        prob_col = "y_prob" if variant_name == "raw" else "y_prob_calibrated"
        for fold_name in comparable_folds:
            fold_df = df[df["fold"] == fold_name]
            y_true = fold_df["y_true"].to_numpy()
            y_prob = fold_df[prob_col].to_numpy()
            metrics = evaluate_predictions(y_true, y_prob)
            rows.append(
                {
                    "fold": fold_name,
                    "variant": variant_name,
                    "brier": metrics["brier"],
                    "log_loss": metrics["log_loss"],
                    "roc_auc": metrics["roc_auc"],
                    "ece": expected_calibration_error(y_true, y_prob),
                }
            )
    return pd.DataFrame(rows)


def plot_reliability_diagram(
    predictions: pd.DataFrame, methods: tuple[str, ...], out_path, window: str = "expanding"
) -> None:
    import matplotlib.pyplot as plt

    fold_order = list(dict.fromkeys(predictions["fold"]))
    comparable_folds = fold_order[1:]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")

    raw_comparable = predictions[predictions["fold"].isin(comparable_folds)]
    raw_table = reliability_table(raw_comparable["y_true"], raw_comparable["y_prob"])
    ax.plot(raw_table["mean_predicted"], raw_table["mean_observed"], marker="o", label="Raw (uncalibrated)")

    for method in methods:
        calibrated = chain_calibrate(predictions, method, window=window)
        calibrated_comparable = calibrated[calibrated["fold"].isin(comparable_folds)]
        table = reliability_table(calibrated_comparable["y_true"], calibrated_comparable["y_prob_calibrated"])
        ax.plot(table["mean_predicted"], table["mean_observed"], marker="o", label=method)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability diagram (pooled across comparable folds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    RESULTS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for horizon in (5, 20):
        print(f"\n=== Calibration — Horizon {horizon}d ===")
        panel = load_modeling_dataset(horizon=horizon)
        best_params = tune_lgbm_hyperparameters(panel, "2011-12-31", "2014-01-01")

        def _tuned_pipeline_builder(numeric_columns, _params=best_params):
            return build_lgbm_pipeline(numeric_columns, **_params)

        folds = purged_embargoed_walk_forward_splits(panel, initial_train_end="2013-12-31", embargo_days=horizon)
        predictions = run_walk_forward_predictions(panel, folds, _tuned_pipeline_builder)

        comparison = evaluate_calibration_methods(predictions, window="expanding")
        comparison.to_csv(RESULTS_DATA_DIR / f"calibration_comparison_{horizon}d.csv", index=False)
        print(comparison.groupby("variant")[["brier", "log_loss", "roc_auc", "ece"]].mean().round(4).to_string())

        plot_reliability_diagram(
            predictions, ("sigmoid", "isotonic"), RESULTS_DATA_DIR / f"reliability_diagram_{horizon}d.png"
        )
