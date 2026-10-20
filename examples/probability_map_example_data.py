"""Shared synthetic dataset for the classifier probability-map examples."""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import make_classification

RANDOM_STATE = 42


def create_example_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Create a reproducible binary dataset with four numeric predictors."""
    values, labels = make_classification(
        n_samples=600,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        class_sep=1.25,
        flip_y=0.03,
        weights=[0.50, 0.50],
        random_state=RANDOM_STATE,
    )
    X = pd.DataFrame(values, columns=["X1", "X2", "X3", "X4"])
    y = pd.Series(labels, name="class").map({0: "O", 1: "F"})
    return X, y
