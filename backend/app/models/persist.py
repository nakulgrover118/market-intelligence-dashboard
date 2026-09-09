"""Trains and persists the final models (one per direction x horizon) so
the API can serve predictions without retraining on every request.

Two directions, not one: "up" (P(forward return >= +threshold)) and
"down" (P(forward return <= -threshold)) are independent models sharing
the same features and the same volatility-scaled threshold magnitude —
see docs/roadmap.md for why a single upside-only number can't honestly
support a bullish/bearish read, and why the fix was training this real
second model rather than relabeling the first one.

Re-run this script whenever the underlying data has been refreshed
(new ingestion -> cleaning -> features -> labels) and you want the served
models to reflect it. Nothing about the training methodology differs from
Phase 5/7's fit_final_model — this just adds persistence on top, now for
both directions.
"""

import logging

import joblib

from app.data.paths import MODELS_DATA_DIR
from app.models.baseline import CATEGORICAL_COLUMNS, feature_columns
from app.models.explain import fit_final_model

logger = logging.getLogger(__name__)

DEV_INITIAL_TRAIN_END = "2011-12-31"
DEV_CUTOFF = "2014-01-01"
HORIZONS = (5, 20)
DIRECTIONS = ("up", "down")


def artifact_filename(direction: str, horizon: int) -> str:
    return f"lightgbm_{direction}_{horizon}d.joblib"


def train_and_persist(direction: str, horizon: int) -> None:
    pipeline, panel = fit_final_model(horizon, DEV_INITIAL_TRAIN_END, DEV_CUTOFF, direction=direction)
    artifact = {
        "pipeline": pipeline,
        "numeric_columns": feature_columns(panel),
        "categorical_columns": CATEGORICAL_COLUMNS,
        "trained_through": panel["date"].max(),
        "horizon": horizon,
        "direction": direction,
    }
    MODELS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DATA_DIR / artifact_filename(direction, horizon)
    joblib.dump(artifact, out_path)
    logger.info(
        "Persisted %s %dd model (trained through %s) -> %s",
        direction, horizon, artifact["trained_through"].date(), out_path,
    )


def train_and_persist_all(
    horizons: tuple[int, ...] = HORIZONS, directions: tuple[str, ...] = DIRECTIONS
) -> None:
    for direction in directions:
        for horizon in horizons:
            train_and_persist(direction, horizon)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    train_and_persist_all()
