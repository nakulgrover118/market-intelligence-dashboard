import pandas as pd
import pytest

from app.models.cv import purged_embargoed_walk_forward_splits

INITIAL_TRAIN_END = pd.Timestamp("2010-12-31")
EMBARGO_DAYS = 5
# embargo_cutoff for the 2011 test fold = 2011-01-01 - 5 days = 2010-12-27


@pytest.fixture
def panel() -> pd.DataFrame:
    rows = [
        # Comfortably in train for the 2011 fold: well before the embargo
        # cutoff, label window doesn't reach 2011.
        {"date": "2010-06-01", "label_end_date": "2010-06-10", "row": "clean_train"},
        # Also comfortably in train: still before the embargo cutoff
        # (2010-12-20 < 2010-12-27) and label doesn't overlap the test year.
        {"date": "2010-12-20", "label_end_date": "2010-12-30", "row": "train_near_boundary"},
        # PURGE case: date is well before the embargo cutoff, but the label
        # window (e.g. a 20-day-forward label) extends into the 2011 test
        # year. Must be excluded from train by the purge rule alone.
        {"date": "2010-12-01", "label_end_date": "2011-01-05", "row": "purge_only"},
        # EMBARGO case: date is on/after the embargo cutoff (2010-12-28 is
        # not < 2010-12-27), but the label window does NOT overlap the test
        # year. Must be excluded from train by the embargo rule alone.
        {"date": "2010-12-28", "label_end_date": "2010-12-29", "row": "embargo_only"},
        # Test row for the 2011 fold.
        {"date": "2011-03-15", "label_end_date": "2011-03-22", "row": "test_2011"},
        # This row is a TEST row for the 2011 fold, but should become
        # eligible TRAIN data for the 2012 fold (expanding window).
        {"date": "2011-06-01", "label_end_date": "2011-06-10", "row": "test_2011_train_2012"},
        # Test row for the 2012 fold.
        {"date": "2012-03-15", "label_end_date": "2012-03-22", "row": "test_2012"},
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["label_end_date"] = pd.to_datetime(df["label_end_date"])
    return df


def _row_labels(panel, idx) -> set[str]:
    return set(panel.loc[idx, "row"])


def test_two_folds_produced_for_2011_and_2012(panel):
    folds = purged_embargoed_walk_forward_splits(panel, INITIAL_TRAIN_END, embargo_days=EMBARGO_DAYS)
    assert [f.name for f in folds] == ["2011", "2012"]


def test_test_set_matches_calendar_year_exactly(panel):
    folds = purged_embargoed_walk_forward_splits(panel, INITIAL_TRAIN_END, embargo_days=EMBARGO_DAYS)
    fold_2011 = folds[0]
    assert _row_labels(panel, fold_2011.test_idx) == {"test_2011", "test_2011_train_2012"}


def test_purge_excludes_row_with_overlapping_label_window(panel):
    folds = purged_embargoed_walk_forward_splits(panel, INITIAL_TRAIN_END, embargo_days=EMBARGO_DAYS)
    fold_2011_train = _row_labels(panel, folds[0].train_idx)
    assert "purge_only" not in fold_2011_train


def test_embargo_excludes_row_within_buffer_even_without_label_overlap(panel):
    folds = purged_embargoed_walk_forward_splits(panel, INITIAL_TRAIN_END, embargo_days=EMBARGO_DAYS)
    fold_2011_train = _row_labels(panel, folds[0].train_idx)
    assert "embargo_only" not in fold_2011_train


def test_clean_rows_are_included_in_train(panel):
    folds = purged_embargoed_walk_forward_splits(panel, INITIAL_TRAIN_END, embargo_days=EMBARGO_DAYS)
    fold_2011_train = _row_labels(panel, folds[0].train_idx)
    assert "clean_train" in fold_2011_train
    assert "train_near_boundary" in fold_2011_train


def test_expanding_window_includes_prior_test_year_in_later_train(panel):
    folds = purged_embargoed_walk_forward_splits(panel, INITIAL_TRAIN_END, embargo_days=EMBARGO_DAYS)
    fold_2012_train = _row_labels(panel, folds[1].train_idx)
    assert "test_2011_train_2012" in fold_2012_train
    # And the row that was purge-excluded from the 2011 fold is no longer
    # purge-relevant for 2012 (its label window ended well before 2012).
    assert "purge_only" in fold_2012_train


def test_no_row_appears_in_both_train_and_test_of_the_same_fold(panel):
    folds = purged_embargoed_walk_forward_splits(panel, INITIAL_TRAIN_END, embargo_days=EMBARGO_DAYS)
    for fold in folds:
        assert set(fold.train_idx) & set(fold.test_idx) == set()


def test_test_periods_across_folds_do_not_overlap(panel):
    folds = purged_embargoed_walk_forward_splits(panel, INITIAL_TRAIN_END, embargo_days=EMBARGO_DAYS)
    all_test_idx = [idx for fold in folds for idx in fold.test_idx]
    assert len(all_test_idx) == len(set(all_test_idx))


def test_no_test_years_before_or_equal_to_initial_train_end_year(panel):
    folds = purged_embargoed_walk_forward_splits(panel, pd.Timestamp("2011-12-31"), embargo_days=EMBARGO_DAYS)
    assert [f.name for f in folds] == ["2012"]
