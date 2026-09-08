"""Trains and persists the Phase 7 final models (one per horizon) so the
API can serve predictions without retraining on every request.

Re-run this script whenever the underlying data has been refreshed
(new ingestion -> cleaning -> features -> labels) and you want the served
model to reflect it. Nothing about the training methodology differs from
Phase 7's fit_final_model — this just adds persistence on top.
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


def train_and_persist(horizon: int) -> None:
    pipeline, panel = fit_final_model(horizon, DEV_INITIAL_TRAIN_END, DEV_CUTOFF)
    artifact = {
        "pipeline": pipeline,
        "numeric_columns": feature_columns(panel),
        "categorical_columns": CATEGORICAL_COLUMNS,
        "trained_through": panel["date"].max(),
        "horizon": horizon,
    }
    MODELS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DATA_DIR / f"lightgbm_{horizon}d.joblib"
    joblib.dump(artifact, out_path)
    logger.info("Persisted %dd model (trained through %s) -> %s", horizon, artifact["trained_through"].date(), out_path)


def train_and_persist_all(horizons: tuple[int, ...] = HORIZONS) -> None:
    for horizon in horizons:
        train_and_persist(horizon)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    train_and_persist_all()
