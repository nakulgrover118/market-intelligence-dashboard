import numpy as np
import pandas as pd
import pytest

from app.models import backtest


@pytest.fixture
def regime_data():
    """Two vol regimes, both 2000 rows. In the low-vol regime, y_prob is
    genuinely informative (correlated with y_true); in the high-vol
    regime, y_prob is pure noise, unrelated to y_true. A correct regime
    analysis must show a much higher ROC-AUC for the low-vol bucket."""
    rng = np.random.default_rng(2)
    n_per_regime = 2000

    low_vol = np.full(n_per_regime, 0.1) + rng.normal(0, 0.01, n_per_regime)
    high_vol = np.full(n_per_regime, 0.9) + rng.normal(0, 0.01, n_per_regime)
    vol = np.concatenate([low_vol, high_vol])

    signal = rng.normal(0, 1, n_per_regime)
    prob_true = 1 / (1 + np.exp(-2 * signal))
    y_true_low = rng.binomial(1, prob_true)
    y_prob_low = prob_true  # informative

    y_true_high = rng.binomial(1, 0.3, n_per_regime)  # unrelated to y_prob_high
    y_prob_high = rng.uniform(0, 1, n_per_regime)  # pure noise

    y_true = np.concatenate([y_true_low, y_true_high])
    y_prob = np.concatenate([y_prob_low, y_prob_high])
    dates = pd.date_range("2020-01-01", periods=2 * n_per_regime, freq="D")

    predictions = pd.DataFrame({"fold": "2020", "date": dates, "y_true": y_true, "y_prob": y_prob})
    panel = pd.DataFrame({"realized_vol_20d": vol}, index=predictions.index)
    return predictions, panel


def test_regime_stratified_metrics_detects_regime_dependent_skill(regime_data):
    predictions, panel = regime_data
    result = backtest.regime_stratified_metrics(predictions, panel, n_buckets=2)

    assert len(result) == 2
    low_vol_row = result.iloc[0]  # qcut assigns lowest values to q1
    high_vol_row = result.iloc[1]
    assert low_vol_row["roc_auc"] > 0.8
    assert high_vol_row["roc_auc"] < 0.6


def _write_instrument_labels(tmp_path, monkeypatch, ticker, label_df):
    monkeypatch.setattr(backtest, "LABELS_DATA_DIR", tmp_path / "labels")
    (tmp_path / "labels").mkdir(exist_ok=True)
    label_df.to_parquet((tmp_path / "labels") / f"{ticker}.parquet")


def test_build_backtest_dataset_attaches_ticker_and_forward_return(tmp_path, monkeypatch):
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    label_df = pd.DataFrame(
        {"forward_return_5d": [0.01, 0.02, -0.01, 0.03, -0.02]}, index=dates
    )
    _write_instrument_labels(tmp_path, monkeypatch, "TEST.NS", label_df)

    predictions = pd.DataFrame(
        {"fold": "2024", "date": dates, "y_true": [1, 0, 1, 0, 1], "y_prob": [0.2, 0.1, 0.3, 0.15, 0.25]}
    )
    panel = pd.DataFrame(
        {"ticker": ["TEST.NS"] * 5, "label_end_date": dates + pd.Timedelta(days=5)}, index=predictions.index
    )

    result = backtest.build_backtest_dataset(predictions, panel, horizon=5)

    assert "forward_return" in result.columns
    assert list(result["forward_return"]) == [0.01, 0.02, -0.01, 0.03, -0.02]
    assert (result["ticker"] == "TEST.NS").all()


def test_simulate_threshold_strategy_enforces_non_overlapping_positions():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    # All 6 rows qualify (high y_prob); label_end_date for the first trade
    # extends past the next 2 candidate dates, so they must be skipped.
    backtest_df = pd.DataFrame(
        {
            "ticker": ["A.NS"] * 6,
            "date": dates,
            "y_prob": [0.9] * 6,
            "label_end_date": dates + pd.Timedelta(days=5),
            "forward_return": [0.05, 0.04, 0.03, 0.02, 0.01, 0.06],
        }
    )
    trades = backtest.simulate_threshold_strategy(backtest_df, percentile_threshold=0.0)

    # First trade at day0 (label_end_date = day5); next eligible entry is
    # the first candidate with date >= day5, i.e. day5 itself.
    assert list(trades["date"]) == [dates[0], dates[5]]


def test_simulate_threshold_strategy_respects_percentile_cutoff():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    backtest_df = pd.DataFrame(
        {
            "ticker": ["A.NS", "B.NS", "C.NS", "D.NS"],
            "date": dates,
            "y_prob": [0.1, 0.5, 0.8, 0.95],
            "label_end_date": dates + pd.Timedelta(days=1),
            "forward_return": [0.0, 0.0, 0.0, 0.0],
        }
    )
    trades = backtest.simulate_threshold_strategy(backtest_df, percentile_threshold=0.75)
    # Only the top quartile (y_prob >= 0.95th percentile of [.1,.5,.8,.95]) should qualify
    assert set(trades["ticker"]) <= {"D.NS", "C.NS"}
    assert "A.NS" not in set(trades["ticker"])


def test_evaluate_strategy_computes_net_return_and_win_rate():
    trades = pd.DataFrame({"forward_return": [0.05, -0.01, 0.10, -0.03]})
    result = backtest.evaluate_strategy(trades, baseline_mean_return=0.005, cost=0.02)

    net_returns = np.array([0.05, -0.01, 0.10, -0.03]) - 0.02
    assert result["n_trades"] == 4
    assert result["mean_net_return"] == pytest.approx(net_returns.mean())
    assert result["win_rate"] == pytest.approx((net_returns > 0).mean())
    assert result["baseline_mean_return"] == 0.005


def test_evaluate_strategy_handles_no_trades():
    trades = pd.DataFrame({"forward_return": []})
    result = backtest.evaluate_strategy(trades, baseline_mean_return=0.005)
    assert result["n_trades"] == 0
    assert np.isnan(result["win_rate"])
