"""Command-line analysis of numeric predictor correlations with a target."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from scipy.stats import pointbiserialr

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def calculate_target_correlations(
    data: pd.DataFrame,
    target_column: str = "Y",
    positive_class: str | None = None,
) -> pd.DataFrame:
    """Calculate point-biserial correlations, using one-vs-rest for multiclass.

    For a binary target, one selected positive class is compared with the other
    class. If no class is selected, the last class encountered in the data is
    used. For a multiclass target, every class is compared with all remaining
    classes and ``positive_class`` must be omitted.
    """
    if target_column not in data.columns:
        raise ValueError(f"Input data must contain target column {target_column!r}.")

    target = data[target_column]
    labels = list(pd.unique(target.dropna()))
    if len(labels) < 2:
        raise ValueError("Target must contain at least two nonmissing classes.")

    features = data.drop(columns=target_column).select_dtypes(include="number")
    if features.empty:
        raise ValueError("Input data contains no numeric predictor columns.")

    if len(labels) == 2:
        if positive_class is None:
            selected_class = labels[-1]
        else:
            matches = [label for label in labels if str(label) == positive_class]
            if len(matches) != 1:
                raise ValueError(
                    f"Positive class {positive_class!r} was not found. "
                    f"Available classes: {[str(label) for label in labels]}"
                )
            selected_class = matches[0]
        comparison_classes = [selected_class]
        comparison_labels = {
            selected_class: f"{selected_class} vs rest",
        }
    else:
        if positive_class is not None:
            raise ValueError(
                "--positive-class applies only to a binary target; "
                "multiclass correlations are calculated one-vs-rest for every class."
            )
        comparison_classes = labels
        comparison_labels = {label: f"{label} vs rest" for label in comparison_classes}

    rows: list[dict[str, object]] = []
    for feature in features.columns:
        values = features[feature]
        valid = values.notna() & target.notna()
        if int(valid.sum()) < 2:
            raise ValueError(f"Not enough complete observations for {feature!r}.")
        observed_target = target.loc[valid]
        observed_values = values.loc[valid]
        for label in comparison_classes:
            is_class = observed_target.eq(label)
            class_support = int(is_class.sum())
            rest_support = int((~is_class).sum())
            if class_support == 0 or rest_support == 0:
                raise ValueError(
                    f"Feature {feature!r} has no complete observations in one "
                    f"side of the {label!r}-versus-rest comparison."
                )
            result = pointbiserialr(is_class.astype(int), observed_values)
            rows.append(
                {
                    "feature": feature,
                    "target": target_column,
                    "class": str(label),
                    "comparison": comparison_labels[label],
                    "n_observations": int(valid.sum()),
                    "class_observations": class_support,
                    "rest_observations": rest_support,
                    "point_biserial_r": float(result.statistic),
                    "p_value": float(result.pvalue),
                }
            )

    return pd.DataFrame(rows).sort_values(
        "point_biserial_r",
        key=lambda values: values.abs(),
        ascending=False,
        ignore_index=True,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the target-correlation CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Calculate point-biserial correlations between numeric predictors "
            "and a binary or multiclass target."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=_PROJECT_ROOT / "data_OF.csv",
        help="Input CSV file (default: project data_OF.csv).",
    )
    parser.add_argument("--target", default="Y", help="Target column (default: Y).")
    parser.add_argument(
        "--positive-class",
        help=(
            "Positive class for a binary target. Defaults to the last class "
            "encountered in the input."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output CSV path. Defaults to results_<dataset>/target_correlation.csv "
            "beside the input data."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Load the CSV, calculate correlations, and save the report."""
    args = build_parser().parse_args(argv)
    data_file = args.data.expanduser().resolve()
    if not data_file.is_file():
        raise FileNotFoundError(f"Input data file not found: {data_file}")

    data = pd.read_csv(data_file)
    results = calculate_target_correlations(
        data,
        target_column=args.target,
        positive_class=args.positive_class,
    )
    dataset_name = data_file.stem.removeprefix("data_")
    output_file = (
        args.output.expanduser().resolve()
        if args.output is not None
        else data_file.parent / f"results_{dataset_name}" / "target_correlation.csv"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_file, index=False)

    classes = list(pd.unique(data[args.target].dropna()))
    if len(classes) == 2:
        positive = (
            args.positive_class if args.positive_class is not None else str(classes[-1])
        )
        print(f"Point-biserial correlations with {args.target} ({positive} vs rest):")
    else:
        print(f"One-vs-rest point-biserial correlations with {args.target}:")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(f"\nSaved results to: {output_file}")


if __name__ == "__main__":
    main()
