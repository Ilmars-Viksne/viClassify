"""Focused checks for forward validation and uncertainty helpers."""

import numpy as np
import pandas as pd

from viclassify_cli import (
    ClassificationAnalysis,
    Config,
    moving_block_bootstrap_indices,
    wilson_interval,
)


def make_analysis(tmp_path, **overrides):
    defaults = {
        "data": tmp_path / "unused.csv",
        "output": tmp_path / "out",
        "target": "Y",
        "features": ("X",),
        "class_order": None,
        "test_fraction": 0.2,
        "cv_folds": 3,
        "cv_test_size": 4,
        "cv_min_class_count": 1,
        "random_state": 17,
        "n_jobs": 1,
        "permutation_repeats": 1,
        "dpi": 50,
    }
    defaults.update(overrides)
    return ClassificationAnalysis(Config(**defaults))


def test_expanding_cv_is_ordered_and_respects_gap(tmp_path):
    analysis = make_analysis(tmp_path, cv_gap=2)
    labels = pd.Series(["A", "B"] * 12)
    splits = analysis.validated_cv_splits(labels)

    validation_rows = []
    for train, valid in splits:
        assert train.max() < valid.min()
        assert valid.min() - train.max() - 1 >= 2
        validation_rows.extend(valid.tolist())
    assert len(validation_rows) == len(set(validation_rows))


def test_wilson_absent_class_is_not_estimable():
    lower, upper = wilson_interval(0, 0)
    assert np.isnan(lower)
    assert np.isnan(upper)


def test_moving_block_bootstrap_is_deterministic():
    left = moving_block_bootstrap_indices(20, 5, np.random.default_rng(41))
    right = moving_block_bootstrap_indices(20, 5, np.random.default_rng(41))
    np.testing.assert_array_equal(left, right)
    assert len(left) == 20
