"""Pytest configuration and fixtures for viClassify tests."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from viclassify_cli import ClassificationAnalysis, Config

# =============================================================================
# Deterministic Dataset Fixtures
# =============================================================================


@pytest.fixture
def binary_chronological_dataset() -> str:
    """Fixture A: Binary chronological dataset with clear row identity."""
    # 20 rows, test_fraction=0.25 -> split=15, holdout=5
    # Both classes in training and holdout
    rows = []
    for i in range(20):
        x1 = float(i)
        x2 = float(i * 10)
        # Alternate classes to ensure both in training and holdout
        y = "O" if i % 2 == 0 else "F"
        rows.append(f"{x1},{x2},{y}")
    return "X1,X2,Y\n" + "\n".join(rows)


@pytest.fixture
def multiclass_chronological_dataset() -> str:
    """Fixture B: Multiclass chronological dataset with 3 classes."""
    # 24 rows, test_fraction=0.25 -> split=18, holdout=6
    # 3 classes: O, B, G - each appears in training and holdout
    rows = []
    classes = ["O", "B", "G"]
    for i in range(24):
        x1 = float(i)
        x2 = float(i * 10)
        x3 = float(i * 100)
        y = classes[i % 3]
        rows.append(f"{x1},{x2},{x3},{y}")
    return "X1,X2,X3,Y\n" + "\n".join(rows)


@pytest.fixture
def absent_holdout_class_dataset() -> str:
    """Fixture C: Multiclass dataset with absent class in holdout.

    Classes: O, B, G, M
    G appears only in training (first 15 rows), not in holdout (last 5 rows).
    """
    # 20 rows, test_fraction=0.25 -> split=15, holdout=5
    rows = []
    # Training rows (0-14): all 4 classes
    for i in range(15):
        x1 = float(i)
        x2 = float(i * 10)
        x3 = float(i * 100)
        x4 = float(i * 1000)
        x5 = float(i * 10000)
        x6 = float(i * 100000)
        x7 = float(i * 1000000)
        if i < 4:
            y = "O"
        elif i < 8:
            y = "B"
        elif i < 12:
            y = "G"
        else:
            y = "M"
        rows.append(f"{x1},{x2},{x3},{x4},{x5},{x6},{x7},{y}")

    # Holdout rows (15-19): only O, B, M (no G)
    for i in range(15, 20):
        x1 = float(i)
        x2 = float(i * 10)
        x3 = float(i * 100)
        x4 = float(i * 1000)
        x5 = float(i * 10000)
        x6 = float(i * 100000)
        x7 = float(i * 1000000)
        if i < 17:
            y = "O"
        elif i < 18:
            y = "B"
        else:
            y = "M"
        rows.append(f"{x1},{x2},{x3},{x4},{x5},{x6},{x7},{y}")

    return "X1,X2,X3,X4,X5,X6,X7,Y\n" + "\n".join(rows)


@pytest.fixture
def missing_training_class_dataset() -> str:
    """Fixture D: Dataset where a class only appears in holdout.

    Class 'M' only appears in the final holdout rows.
    """
    # 20 rows, test_fraction=0.25 -> split=15, holdout=5
    # Training (0-14): only O, B, G
    # Holdout (15-19): includes M
    rows = []
    for i in range(15):
        x1 = float(i)
        x2 = float(i * 10)
        x3 = float(i * 100)
        x4 = float(i * 1000)
        x5 = float(i * 10000)
        x6 = float(i * 100000)
        x7 = float(i * 1000000)
        if i < 5:
            y = "O"
        elif i < 10:
            y = "B"
        else:
            y = "G"
        rows.append(f"{x1},{x2},{x3},{x4},{x5},{x6},{x7},{y}")

    for i in range(15, 20):
        x1 = float(i)
        x2 = float(i * 10)
        x3 = float(i * 100)
        x4 = float(i * 1000)
        x5 = float(i * 10000)
        x6 = float(i * 100000)
        x7 = float(i * 1000000)
        y = "M"  # Only in holdout
        rows.append(f"{x1},{x2},{x3},{x4},{x5},{x6},{x7},{y}")

    return "X1,X2,X3,X4,X5,X6,X7,Y\n" + "\n".join(rows)


@pytest.fixture
def imbalanced_dataset() -> str:
    """Highly imbalanced dataset for metric behavior tests."""
    # 100 rows, 90% class O, 10% class F
    rows = []
    for i in range(100):
        x1 = float(i)
        x2 = float(i * 10)
        y = "O" if i < 90 else "F"
        rows.append(f"{x1},{x2},{y}")
    return "X1,X2,Y\n" + "\n".join(rows)


@pytest.fixture
def temp_data_file(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    """Create a temporary CSV file from a dataset fixture."""
    dataset_name = request.param
    dataset = request.getfixturevalue(dataset_name)
    data_file = tmp_path / f"{dataset_name}.csv"
    data_file.write_text(dataset, encoding="utf-8")
    return data_file


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory."""
    output_dir = tmp_path / "run"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def minimal_config(temp_data_file: Path, temp_output_dir: Path) -> Config:
    """Create a minimal Config for fast testing."""
    # We need to infer features from the data file
    df = pd.read_csv(temp_data_file)
    features = tuple([c for c in df.columns if c != "Y"])
    return Config(
        data=temp_data_file,
        output=temp_output_dir,
        target="Y",
        features=features,
        class_order=None,
        test_fraction=0.25,
        cv_folds=2,
        random_state=42,
        n_jobs=1,
        permutation_repeats=1,
        dpi=50,
        near_constant_threshold=0.99,
        selection_metric="macro_f1",
    )


@pytest.fixture
def minimal_grids() -> dict[str, dict[str, list[Any]]]:
    """Minimal one-candidate grids for fast testing."""
    return {
        "Logistic Regression": {"classifier__C": [1.0]},
        "Random Forest": {
            "classifier__n_estimators": [10],
            "classifier__min_samples_leaf": [1],
            "classifier__max_features": ["sqrt"],
        },
        "Extra Trees": {
            "classifier__n_estimators": [10],
            "classifier__min_samples_leaf": [1],
            "classifier__max_features": ["sqrt"],
        },
        "Histogram Gradient Boosting": {
            "classifier__learning_rate": [0.1],
            "classifier__max_iter": [50],
            "classifier__max_leaf_nodes": [15],
            "classifier__l2_regularization": [0.0],
        },
    }


# =============================================================================
# Helper Functions
# =============================================================================


def create_analysis(
    data_file: Path,
    output_dir: Path,
    features: tuple[str, ...],
    class_order: tuple[str, ...] | None = None,
    test_fraction: float = 0.25,
    cv_folds: int = 2,
    grids: dict[str, dict[str, list[Any]]] | None = None,
) -> ClassificationAnalysis:
    """Create a ClassificationAnalysis with minimal grids for fast testing."""
    cfg = Config(
        data=data_file,
        output=output_dir,
        target="Y",
        features=features,
        class_order=class_order,
        test_fraction=test_fraction,
        cv_folds=cv_folds,
        random_state=42,
        n_jobs=1,
        permutation_repeats=1,
        dpi=50,
        near_constant_threshold=0.99,
        selection_metric="macro_f1",
    )
    analysis = ClassificationAnalysis(cfg)
    if grids is not None:
        analysis.grids = grids
    return analysis


def run_analysis_stages(
    analysis: ClassificationAnalysis,
    stages: list[str] | None = None,
) -> None:
    """Run specific stages of the analysis pipeline."""
    if stages is None:
        stages = [
            "load",
            "audit_and_describe",
            "correlations",
            "exploratory_plots",
            "create_chronological_split",
            "build_models",
            "tune_on_training_only",
            "evaluate_untouched_holdout",
            "export_evaluation_logistic_outputs",
            "refit_and_export_production_models",
            "metadata",
            "write_generated_file_manifest",
        ]

    for stage in stages:
        getattr(analysis, stage)()


# =============================================================================
# Metric Calculation Helpers
# =============================================================================


def compute_expected_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    classes: list[str] | None = None,
) -> dict[str, float]:
    """Compute expected metric values using sklearn with explicit labels."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        log_loss,
        matthews_corrcoef,
        precision_score,
        recall_score,
    )

    if classes is None:
        classes = sorted(np.unique(y_true).tolist())

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": recall_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        ),
        "macro_precision": precision_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, y_pred, labels=classes, average="weighted", zero_division=0
        ),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    if y_proba is not None:
        try:
            metrics["log_loss"] = log_loss(y_true, y_proba, labels=classes)
        except ValueError:
            metrics["log_loss"] = np.nan

    return metrics


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (slower)"
    )
    config.addinivalue_line("markers", "slow: marks tests as slow")


# Ensure matplotlib figures are closed after each test
@pytest.fixture(autouse=True)
def close_figures():
    """Automatically close matplotlib figures after each test."""
    import matplotlib.pyplot as plt

    yield
    plt.close("all")
