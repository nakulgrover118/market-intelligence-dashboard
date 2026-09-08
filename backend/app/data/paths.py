"""Central definitions for where each data layer lives on disk."""

from pathlib import Path

_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"

RAW_DATA_DIR = _DATA_ROOT / "raw"
PROCESSED_DATA_DIR = _DATA_ROOT / "processed"
FEATURES_DATA_DIR = _DATA_ROOT / "features"
LABELS_DATA_DIR = _DATA_ROOT / "labels"


def ticker_filename(ticker: str) -> str:
    """Index tickers start with '^', which is not a valid filename character."""
    return f"{ticker.replace('^', 'IDX_')}.parquet"
