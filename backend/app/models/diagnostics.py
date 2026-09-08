"""Diagnostic detour between the weak Phase 4c baseline and Phase 5.

Answers one question before investing in a fully-tuned gradient-boosting
build: is there real signal in these features at all, and if so, is it
linear (logistic regression should already have found it) or nonlinear
(a tree ensemble might find it where a linear model can't)?

Two checks:
1. Feature-label correlation and mutual information — cheap, tells us
   whether any individual feature has a meaningful linear or nonlinear
   univariate relationship with the label.
2. The more decisive test: run the *exact same* purged walk-forward
   protocol (same folds, same preprocessing) with a nonlinear model
   (HistGradientBoostingClassifier — fast enough for a gut-check, no
   tuning needed to answer this question) and compare fold-by-fold
   against the logistic regression baseline already computed.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.data.paths import PROCESSED_DATA_DIR, ticker_filename
from app.data.universe import UNIVERSE, Instrument
from app.features import technical as ta
from app.features.labels import DAILY_VOL_WINDOW, build_labels_for_instrument
from app.models.baseline import CATEGORICAL_COLUMNS, feature_columns, run_walk_forward_evaluation
from app.models.dataset import assemble_panel


def feature_label_correlations(panel: pd.DataFrame) -> pd.Series:
    """Pearson correlation of each feature with the binary label (point-
    biserial correlation — Pearson's formula applied to a 0/1 target is
    exactly that). Only captures linear relationships."""
    cols = feature_columns(panel)
    corr = panel[cols].corrwith(panel["label"].astype(float))
    return corr.reindex(corr.abs().sort_values(ascending=False).index)


def feature_mutual_information(panel: pd.DataFrame, random_state: int = 0) -> pd.Series:
    """Mutual information between each feature and the label — captures
    nonlinear relationships a correlation coefficient would miss entirely
    (e.g. a feature that predicts the label only in its extreme deciles)."""
    cols = feature_columns(panel)
    mi = mutual_info_classif(panel[cols], panel["label"], random_state=random_state)
    return pd.Series(mi, index=cols).sort_values(ascending=False)


def build_gbm_pipeline(numeric_columns: list[str]) -> Pipeline:
    """Same preprocessing shape as the logistic regression baseline
    (build_pipeline in baseline.py) — the only thing that varies between
    the two comparisons is the final estimator, so any performance
    difference is attributable to model flexibility, not preprocessing.
    Scaling doesn't help or hurt a tree-based model, but keeping it
    identical avoids that question entirely."""
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric_columns),
            ("sector", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )
    return Pipeline([("preprocess", preprocessor), ("model", HistGradientBoostingClassifier(random_state=0))])


def load_modeling_dataset_with_k(
    horizon: int,
    k: float,
    instruments: list[Instrument] = UNIVERSE,
    include_silver_features: bool = False,
) -> pd.DataFrame:
    """Same shape/contract as app.models.dataset.load_modeling_dataset, but
    computes labels on the fly for an arbitrary k instead of reading the
    persisted (k=0.5) label files — lets us explore alternate thresholds
    without touching the committed Phase 3 label data or its k=0.5 results.
    Reuses assemble_panel so the NaN policy and column-consistency guard
    aren't duplicated here."""

    def _load_labels_for_k(instrument: Instrument) -> pd.DataFrame:
        close = pd.read_parquet(PROCESSED_DATA_DIR / ticker_filename(instrument.ticker))["adj_close"]
        daily_vol = ta.realized_volatility(close, DAILY_VOL_WINDOW)
        return build_labels_for_instrument(close, daily_vol, k, horizons=(horizon,))

    return assemble_panel(instruments, horizon, _load_labels_for_k, include_silver_features)


def compare_logreg_vs_gbm(panel: pd.DataFrame, folds) -> pd.DataFrame:
    from app.models.baseline import build_pipeline
    from app.models.evaluate import summarize_folds

    logreg_results = summarize_folds(run_walk_forward_evaluation(panel, folds, build_pipeline))
    gbm_results = summarize_folds(run_walk_forward_evaluation(panel, folds, build_gbm_pipeline))

    logreg_results["model"] = "logreg"
    gbm_results["model"] = "gbm"
    return pd.concat([logreg_results, gbm_results], ignore_index=True)
