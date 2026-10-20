"""Plot observed ``data_OF.csv`` values through the fitted logistic model.

The production model artifact preserves its imputation, scaling, feature
order, and class order. Run from any working directory with the project
dependencies installed:

    python examples/logistic_regression_maps.py

The PNG files are saved under ``results_OF/decision_probability_maps/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from viclassify.decision_probability_maps import (
    DecisionProbabilityMapper,
    ProbabilityMapConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data_OF.csv"
RESULTS_DIRECTORY = PROJECT_ROOT / "results_OF"
MODEL_FILE = RESULTS_DIRECTORY / "production_logistic_regression.joblib"
OUTPUT_DIRECTORY = (
    RESULTS_DIRECTORY / "decision_probability_maps" / "logistic_regression"
)
TARGET_COLUMN = "Y"
PREDICTOR_PAIRS = [
    ("X1", "X2"),
    ("X1", "X3"),
    ("X2", "X4"),
]


def load_data_and_model() -> tuple[pd.DataFrame, pd.Series, Any]:
    """Load observed predictors, labels, and the exported production model."""
    for path in (DATA_FILE, MODEL_FILE):
        if not path.is_file():
            raise FileNotFoundError(f"Required input file not found: {path}")

    payload = joblib.load(MODEL_FILE)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Unexpected model artifact format: {MODEL_FILE}")

    features = payload.get("features")
    if not features:
        raise ValueError("The exported model artifact does not list its predictors.")

    data = pd.read_csv(DATA_FILE)
    missing_columns = [
        column for column in [*features, TARGET_COLUMN] if column not in data
    ]
    if missing_columns:
        raise ValueError(
            f"{DATA_FILE.name} is missing required columns: {missing_columns}"
        )

    X_observed = data.loc[:, features].copy()
    y_observed = data[TARGET_COLUMN].copy()
    estimator = payload["model"]
    if list(estimator.feature_names_in_) != list(features):
        raise ValueError(
            "Model predictor order does not match its exported feature list."
        )
    return X_observed, y_observed, estimator


def main() -> None:
    """Generate maps on the original measured predictor coordinates."""
    X_observed, y_observed, model = load_data_and_model()

    mapper = DecisionProbabilityMapper(
        estimator=model,
        X_train=X_observed,
        y_train=y_observed,
        config=ProbabilityMapConfig(
            grid_resolution=200,
            padding_fraction=0.05,
            contour_levels=(0.10, 0.25, 0.50, 0.75, 0.90),
            multiclass_columns=3,
            maximum_classes_per_figure=2,
            probability_colormap="viridis",
            show_decision_overview=True,
            show_probability_contours=True,
            show_other_class_points=True,
            show_fixed_median_annotation=True,
            show_colorbar=True,
            show_legends=True,
            dpi=160,
            file_format="png",
        ),
    )
    created_files = mapper.plot_selected_pairs_multiclass(
        predictor_pairs=PREDICTOR_PAIRS,
        X_points=X_observed,
        y_points=y_observed,
        output_directory=OUTPUT_DIRECTORY,
        filename_prefix="data_OF_logistic_regression",
    )

    print("Observed-data logistic-regression maps created successfully.")
    print(f"Observations: {len(X_observed)}")
    print(f"Predictors: {list(X_observed.columns)}")
    print(f"Model classes: {list(model.classes_)}")
    print(f"Output directory: {OUTPUT_DIRECTORY}")
    for file_path in created_files:
        print(file_path)


if __name__ == "__main__":
    main()
