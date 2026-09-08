"""Phase 8: two follow-up analyses on top of the already-computed
walk-forward out-of-sample predictions — not a re-run of Brier/log-loss/
AUC, which Phases 4-6 already cover.

1. Regime-stratified metrics: does the model's discriminative/calibration
   performance hold up evenly across volatility regimes, or is it
   concentrated in specific conditions? A direct follow-up to Phase 7's
   finding that realized_vol_20d dominates SHAP importance.

2. A cost-aware decision-value backtest: translating probabilities into a
   simple, non-overlapping-position threshold rule and checking whether
   it survives realistic transaction costs. This is explicitly a research
   due-diligence check — "would acting on this probability have added
   value net of costs" — not a trading strategy or product recommendation
   (see docs/roadmap.md: the project's stated goal is probabilities, not
   BUY/SELL signals).

Both reuse the genuine out-of-sample predictions already produced by
run_walk_forward_predictions — nothing here is fit or re-evaluated on
data the model trained on.
"""

import logging

import numpy as np
import pandas as pd

from app.data.paths import LABELS_DATA_DIR, RESULTS_DATA_DIR, ticker_filename
from app.models.baseline import run_walk_forward_predictions
from app.models.cv import purged_embargoed_walk_forward_splits
from app.models.dataset import load_modeling_dataset
from app.models.evaluate import evaluate_predictions, naive_baseline_metrics
from app.models.gbm import build_lgbm_pipeline, tune_lgbm_hyperparameters

logger = logging.getLogger(__name__)

# Illustrative round-trip cost assumption for a liquid NSE large-cap
# (brokerage + STT + slippage, approximate, not an audited costing model).
DEFAULT_TRANSACTION_COST = 0.002


def regime_stratified_metrics(predictions: pd.DataFrame, panel: pd.DataFrame, n_buckets: int = 3) -> pd.DataFrame:
    """`predictions` (fold, date, y_true, y_prob) and `panel` must share
    the same index (as produced by run_walk_forward_predictions against
    that panel). Buckets by realized_vol_20d *at prediction time* — the
    feature Phase 7 found dominates the model's reasoning — into
    `n_buckets` quantile groups, and reports the same Brier/log-loss/AUC
    metrics used everywhere else in this project, separately per bucket,
    against a bucket-specific naive baseline (that bucket's own empirical
    positive rate, not the fold-level training base rate — the fair
    comparator for "does the model add anything within this regime")."""
    vol = panel.loc[predictions.index, "realized_vol_20d"]
    labels = [f"q{i + 1}_of_{n_buckets}" for i in range(n_buckets)]
    bucketed = predictions.assign(vol_bucket=pd.qcut(vol, q=n_buckets, labels=labels))

    rows = []
    for bucket, group in bucketed.groupby("vol_bucket", observed=True):
        y_true = group["y_true"].to_numpy()
        y_prob = group["y_prob"].to_numpy()
        positive_rate = float(y_true.mean())
        metrics = evaluate_predictions(y_true, y_prob)
        naive = naive_baseline_metrics(y_true, positive_rate)
        rows.append(
            {
                "vol_bucket": bucket,
                "n": len(group),
                "positive_rate": positive_rate,
                "brier": metrics["brier"],
                "naive_brier": naive["brier"],
                "log_loss": metrics["log_loss"],
                "naive_log_loss": naive["log_loss"],
                "roc_auc": metrics["roc_auc"],
            }
        )
    return pd.DataFrame(rows)


def build_backtest_dataset(predictions: pd.DataFrame, panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Attaches `ticker` and `label_end_date` (from panel — same index) and
    the realized continuous `forward_return` (loaded fresh from the labels
    parquet here, purely for post-hoc P&L; it is never a model feature)."""
    meta = panel.loc[predictions.index, ["ticker", "label_end_date"]]
    merged = predictions.join(meta)

    return_col = f"forward_return_{horizon}d"
    forward_return_frames = []
    for ticker in merged["ticker"].unique():
        label_df = pd.read_parquet(LABELS_DATA_DIR / ticker_filename(ticker))
        frame = label_df[[return_col]].rename(columns={return_col: "forward_return"})
        frame["ticker"] = ticker
        frame["date"] = frame.index
        forward_return_frames.append(frame)
    forward_returns = pd.concat(forward_return_frames, ignore_index=True)

    return merged.merge(forward_returns, on=["ticker", "date"], how="left")


def simulate_threshold_strategy(backtest_df: pd.DataFrame, percentile_threshold: float) -> pd.DataFrame:
    """Take a position whenever y_prob is in the top `1 - percentile_threshold`
    of predictions, enforcing non-overlapping positions per instrument:
    once a position is taken, no new one opens for that ticker until its
    `label_end_date` (the exact trading-day-correct end of that position's
    holding window — already computed in Phase 3, not approximated here
    via calendar-day arithmetic)."""
    cutoff = backtest_df["y_prob"].quantile(percentile_threshold)
    candidates = backtest_df[backtest_df["y_prob"] >= cutoff].sort_values(["ticker", "date"])

    trades = []
    for _, group in candidates.groupby("ticker"):
        next_eligible_date = None
        for _, row in group.iterrows():
            if next_eligible_date is not None and row["date"] < next_eligible_date:
                continue
            trades.append(row)
            next_eligible_date = row["label_end_date"]
    return pd.DataFrame(trades)


def evaluate_strategy(
    trades: pd.DataFrame, baseline_mean_return: float, cost: float = DEFAULT_TRANSACTION_COST
) -> dict:
    net_returns = trades["forward_return"] - cost
    return {
        "n_trades": len(trades),
        "win_rate": float((net_returns > 0).mean()) if len(trades) else float("nan"),
        "mean_gross_return": float(trades["forward_return"].mean()) if len(trades) else float("nan"),
        "mean_net_return": float(net_returns.mean()) if len(trades) else float("nan"),
        "baseline_mean_return": baseline_mean_return,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    RESULTS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for horizon in (5, 20):
        print(f"\n=== Phase 8 — Horizon {horizon}d ===")
        panel = load_modeling_dataset(horizon=horizon)
        best_params = tune_lgbm_hyperparameters(panel, "2011-12-31", "2014-01-01")

        def _pipeline_builder(numeric_columns, _params=best_params):
            return build_lgbm_pipeline(numeric_columns, **_params)

        folds = purged_embargoed_walk_forward_splits(panel, initial_train_end="2013-12-31", embargo_days=horizon)
        predictions = run_walk_forward_predictions(panel, folds, _pipeline_builder)

        print("\n--- Part 1: regime-stratified metrics (by realized_vol_20d tercile) ---")
        regime_table = regime_stratified_metrics(predictions, panel, n_buckets=3)
        regime_table.to_csv(RESULTS_DATA_DIR / f"regime_stratified_{horizon}d.csv", index=False)
        print(regime_table.to_string(index=False))

        print("\n--- Part 2: cost-aware decision-value backtest (illustrative research, not a trading recommendation) ---")
        backtest_df = build_backtest_dataset(predictions, panel, horizon)
        baseline_mean_return = float(backtest_df["forward_return"].mean())

        strategy_rows = []
        for percentile in (0.5, 0.75, 0.9, 0.95):
            trades = simulate_threshold_strategy(backtest_df, percentile_threshold=percentile)
            result = evaluate_strategy(trades, baseline_mean_return)
            result["percentile_threshold"] = percentile
            strategy_rows.append(result)
        strategy_table = pd.DataFrame(strategy_rows)
        strategy_table.to_csv(RESULTS_DATA_DIR / f"decision_backtest_{horizon}d.csv", index=False)
        print(f"Baseline (unconditional) mean forward return per row: {baseline_mean_return:.4%}")
        print(strategy_table.to_string(index=False))
