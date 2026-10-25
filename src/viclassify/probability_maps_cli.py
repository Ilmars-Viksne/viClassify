"""Unified command-line entry point for saved-model probability maps."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import joblib
import pandas as pd

from viclassify.decision_probability_maps import (
    DecisionProbabilityMapper,
    ProbabilityMapConfig,
)

_MODEL_FILES = {
    "logistic_regression": "production_logistic_regression.joblib",
    "histogram_gradient_boosting": "production_histogram_gradient_boosting.joblib",
    "extra_trees": "production_extra_trees.joblib",
    "random_forest": "production_random_forest.joblib",
}
_PREDICTOR_PAIRS = [
    ("X1", "X2"), ("X1", "X3"), ("X1", "X4"), ("X1", "X5"), ("X1", "X7"),
    ("X2", "X3"), ("X2", "X4"), ("X2", "X5"), ("X2", "X7"),
    ("X3", "X4"), ("X3", "X5"), ("X3", "X7"),
    ("X4", "X5"), ("X4", "X7"), ("X5", "X7"),
]
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    """Build the probability-map CLI parser."""
    parser = argparse.ArgumentParser(
        description="Plot two-predictor decision and probability maps from a saved production model."
    )
    parser.add_argument("--model", choices=tuple(_MODEL_FILES), required=True)
    parser.add_argument("--data", type=Path, default=_PROJECT_ROOT / "data_OF.csv")
    parser.add_argument("--results-dir", type=Path, default=_PROJECT_ROOT / "results_OF")
    parser.add_argument("--target", default="Y")
    parser.add_argument(
        "--maximum-classes-per-figure", type=int,
        help="Maximum class panels per figure; defaults to every model class.",
    )
    parser.add_argument("--grid-resolution", type=int, default=200)
    parser.add_argument("--dpi", type=int, default=160)
    return parser


def run(args: argparse.Namespace) -> list[Path]:
    """Generate maps according to parsed options and return created files."""
    data_file = args.data.expanduser().resolve()
    results_directory = args.results_dir.expanduser().resolve()
    model_file = results_directory / _MODEL_FILES[args.model]
    for path in (data_file, model_file):
        if not path.is_file():
            raise FileNotFoundError(f"Required input file not found: {path}")

    payload = joblib.load(model_file)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Unexpected model artifact format: {model_file}")
    features = payload.get("features")
    if not features:
        raise ValueError("The exported model artifact does not list its predictors.")

    data = pd.read_csv(data_file)
    missing = [column for column in [*features, args.target] if column not in data]
    if missing:
        raise ValueError(f"{data_file.name} is missing required columns: {missing}")
    X = data.loc[:, features].copy()
    y = data[args.target].copy()
    model = payload["model"]
    if list(model.feature_names_in_) != list(features):
        raise ValueError("Model predictor order does not match its exported feature list.")

    absent_pair_features = sorted(
        {name for pair in _PREDICTOR_PAIRS for name in pair} - set(features)
    )
    pairs = [pair for pair in _PREDICTOR_PAIRS if all(name in features for name in pair)]
    if not pairs:
        raise ValueError(
            "The model has no predictors covered by the default X1-X7 pairs. "
            "Use a model with those predictors or extend the CLI pair configuration."
        )
    if absent_pair_features:
        print(f"Skipping default pairs that use unavailable predictors: {absent_pair_features}")

    class_limit = (
        len(model.classes_)
        if args.maximum_classes_per_figure is None
        else args.maximum_classes_per_figure
    )
    if class_limit < 1:
        raise ValueError("--maximum-classes-per-figure must be at least 1.")
    output = results_directory / "decision_probability_maps" / args.model
    mapper = DecisionProbabilityMapper(
        estimator=model,
        X_train=X,
        y_train=y,
        config=ProbabilityMapConfig(
            grid_resolution=args.grid_resolution,
            padding_fraction=0.05,
            contour_levels=(0.10, 0.25, 0.50, 0.75, 0.90),
            multiclass_columns=3,
            maximum_classes_per_figure=class_limit,
            probability_colormap="viridis",
            show_decision_overview=True,
            show_probability_contours=True,
            show_other_class_points=True,
            show_fixed_median_annotation=True,
            show_colorbar=True,
            show_legends=True,
            dpi=args.dpi,
            file_format="png",
        ),
    )
    files = mapper.plot_selected_pairs_multiclass(
        predictor_pairs=pairs,
        X_points=X,
        y_points=y,
        output_directory=output,
        filename_prefix=f"{data_file.stem}_{args.model}",
    )
    print(f"Observations: {len(X)}")
    print(f"Predictors: {list(X.columns)}")
    print(f"Model classes: {list(model.classes_)}")
    print(f"Output directory: {output}")
    for path in files:
        print(path)
    return files


def main(argv: Sequence[str] | None = None) -> None:
    """Run the unified probability-map CLI."""
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
