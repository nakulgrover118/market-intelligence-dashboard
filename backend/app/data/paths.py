"""Central definitions for where each data layer lives on disk."""

import os
from pathlib import Path

# Defaults to a path computed relative to this file (assumes the repo's
# backend/app/data/paths.py layout) — correct for local dev, but fragile
# the moment app/ is deployed without that exact directory nesting (e.g.
# a Docker image that COPYs only backend/app/). MARKET_DATA_ROOT lets a
# deployment point at a mounted volume explicitly instead of relying on
# that assumption — see backend/Dockerfile and docs/deployment.md.
_default_data_root = Path(__file__).resolve().parents[3] / "data"
_DATA_ROOT = Path(os.environ.get("MARKET_DATA_ROOT", _default_data_root))

RAW_DATA_DIR = _DATA_ROOT / "raw"
PROCESSED_DATA_DIR = _DATA_ROOT / "processed"
FEATURES_DATA_DIR = _DATA_ROOT / "features"
LABELS_DATA_DIR = _DATA_ROOT / "labels"
RESULTS_DATA_DIR = _DATA_ROOT / "results"
MODELS_DATA_DIR = _DATA_ROOT / "models"


def ticker_filename(ticker: str) -> str:
    """Index tickers start with '^', which is not a valid filename character."""
    return f"{ticker.replace('^', 'IDX_')}.parquet"
