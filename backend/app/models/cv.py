"""Purged, embargoed, expanding-window walk-forward cross-validation.

Two leakage risks specific to our N-day-forward labels, both handled here:

- Purge: a training row dated just before a test period can still have a
  label window (`label_end_date`) that extends into the test period — a
  20-day-forward label computed 5 days before the cutoff "knows about"
  returns that occur during what we're calling the test period. Dropped
  via the exact `label_end_date` criterion (computed in Phase 3), not a
  guess.
- Embargo: an additional conservative buffer of `embargo_days` immediately
  before the test period, dropped from training regardless of label
  overlap, as a safety margin against serially correlated *features*
  (e.g. volatility clustering) that purging alone doesn't strictly cover.

Folds are calendar-year test blocks with an expanding training window: the
first fold trains on everything up to `initial_train_end`; each subsequent
fold's training window grows to include the previous fold's test year.
This is standard walk-forward validation, applied to a pooled panel (every
instrument shares the same fold boundaries, since they share the same
underlying market timeline).
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class Fold:
    name: str
    train_idx: pd.Index
    test_idx: pd.Index


def purged_embargoed_walk_forward_splits(
    panel: pd.DataFrame,
    initial_train_end: str | pd.Timestamp,
    date_col: str = "date",
    label_end_date_col: str = "label_end_date",
    embargo_days: int = 5,
) -> list[Fold]:
    """`panel` must have one row per (instrument, date) observation, with
    `date_col` and `label_end_date_col` columns (see dataset.py).
    `initial_train_end` is required, not defaulted: it's a consequential
    methodology choice (how much history to bootstrap the first fold with)
    that should be made deliberately by the caller after looking at the
    panel's actual date range, not silently guessed here."""
    dates = pd.to_datetime(panel[date_col])
    label_ends = pd.to_datetime(panel[label_end_date_col])
    initial_train_end = pd.Timestamp(initial_train_end)

    test_years = sorted(
        year for year in dates.dt.year.unique() if pd.Timestamp(f"{year}-01-01") > initial_train_end
    )

    folds = []
    for year in test_years:
        test_start = pd.Timestamp(f"{year}-01-01")
        test_end = pd.Timestamp(f"{year}-12-31")

        test_mask = (dates >= test_start) & (dates <= test_end)
        if not test_mask.any():
            continue

        embargo_cutoff = test_start - pd.Timedelta(days=embargo_days)
        train_mask = (
            (dates < embargo_cutoff)  # embargo: buffer before the test period
            & (label_ends < test_start)  # purge: label window must not overlap test
        )
        if not train_mask.any():
            continue

        folds.append(
            Fold(name=str(year), train_idx=panel.index[train_mask], test_idx=panel.index[test_mask])
        )

    return folds
