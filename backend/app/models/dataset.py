"""Builds the pooled cross-sectional modeling dataset: one row per
(instrument, date), joining that instrument's features and labels.

Pooling all instruments into one panel is a deliberate choice (see
docs/roadmap.md): a single-instrument model trains on ~5,000 rows and is
noisy; pooling gives ~100k+ rows and lets the model learn relationships
that generalize across stocks, at the cost of needing `sector` as a
feature so it can still specialize per-sector rather than per-ticker.
"""

from typing import Callable

import numpy as np
import pandas as pd

from app.data.paths import FEATURES_DATA_DIR, LABELS_DATA_DIR, ticker_filename
from app.data.universe import UNIVERSE, Instrument

# silver_return_*/gold_silver_ratio are only defined from SILVERBEES.NS's
# 2022 listing onward (see docs/roadmap.md) — including them by default
# would force dropping ~80% of the full ~20-year history at the dropna()
# below. Excluded by default; pass include_silver_features=True for a
# separate, recent-period-only experiment instead.
_SILVER_FEATURE_MARKER = "silver"


def feature_columns_for_frame(feature_df: pd.DataFrame, include_silver_features: bool) -> list[str]:
    if include_silver_features:
        return list(feature_df.columns)
    return [c for c in feature_df.columns if _SILVER_FEATURE_MARKER not in c]


def _label_column_name(horizon: int, direction: str) -> str:
    if direction == "up":
        return f"label_{horizon}d"
    if direction == "down":
        return f"label_down_{horizon}d"
    raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")


def assemble_panel(
    instruments: list[Instrument],
    horizon: int,
    load_label_df: Callable[[Instrument], pd.DataFrame],
    include_silver_features: bool = False,
    direction: str = "up",
) -> pd.DataFrame:
    """Shared core: joins each instrument's features against whatever
    `load_label_df` returns for it, and pools the result into one panel.
    `load_label_df` is pluggable so this same, guarded assembly logic can
    back both the persisted labels in load_modeling_dataset below and the
    on-the-fly, alternate-k labels used in Phase 4c's diagnostic detour
    (app/models/diagnostics.py) — without duplicating the NaN policy or
    the column-consistency guard in two places. `direction` picks which
    label ("up" -> label_{n}d, "down" -> label_down_{n}d) becomes the
    panel's `label` column — both share the same `label_end_date` (the
    date column is direction-independent, only the threshold comparison
    differs)."""
    label_col = _label_column_name(horizon, direction)
    end_date_col = f"label_end_date_{horizon}d"

    frames = []
    for instrument in instruments:
        feature_df = pd.read_parquet(FEATURES_DATA_DIR / ticker_filename(instrument.ticker))
        label_df = load_label_df(instrument)

        feature_cols = feature_columns_for_frame(feature_df, include_silver_features)
        merged = feature_df[feature_cols].join(label_df[[label_col, end_date_col]], how="inner")
        merged = merged.replace([np.inf, -np.inf], np.nan).dropna()
        if merged.empty:
            continue

        merged = merged.rename(columns={label_col: "label", end_date_col: "label_end_date"})
        merged["ticker"] = instrument.ticker
        merged["sector"] = instrument.sector or instrument.asset_class.value
        merged["date"] = merged.index
        frames.append(merged.reset_index(drop=True))

    # A per-instrument feature file with a different column set than the
    # others would have pd.concat silently NaN-fill the gap for every row
    # of the instrument lacking it — invisible until model fitting fails,
    # far from the actual cause. Fail loudly here instead (see
    # test_universe_output_has_identical_columns_across_all_instruments in
    # test_build.py for the real bug this guards against).
    column_sets = {tuple(sorted(f.columns)) for f in frames}
    if len(column_sets) > 1:
        raise ValueError(
            f"Instruments have inconsistent feature columns ({len(column_sets)} distinct "
            "column sets) — pooling them would silently NaN-fill the gaps. Check that "
            "build_features_for_universe computes the same columns for every instrument."
        )

    panel = pd.concat(frames, ignore_index=True)
    panel["label"] = panel["label"].astype(int)
    return panel.sort_values("date").reset_index(drop=True)


def load_modeling_dataset(
    horizon: int,
    instruments: list[Instrument] = UNIVERSE,
    include_silver_features: bool = False,
    direction: str = "up",
) -> pd.DataFrame:
    """One row per (ticker, date) with feature columns, `label`,
    `label_end_date`, `ticker`, `sector`, and `date`, using the persisted
    (k=1.5) labels from Phase 3/4d. `direction="up"` targets a big upward
    move, `"down"` a big downward move — both are legitimate, independent
    models sharing the same features and the same volatility-scaled
    threshold magnitude (see docs/roadmap.md). Rows with any missing
    feature, an infinite feature value, or an undefined label are dropped
    here — this is the modeling-stage NaN policy referenced throughout
    Phase 2/3 (upstream layers only flag issues; this is where we finally
    act on them by exclusion)."""

    def _load_persisted_labels(instrument: Instrument) -> pd.DataFrame:
        return pd.read_parquet(LABELS_DATA_DIR / ticker_filename(instrument.ticker))

    return assemble_panel(instruments, horizon, _load_persisted_labels, include_silver_features, direction)
