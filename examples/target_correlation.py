"""Calculate each predictor's point-biserial correlation with target Y.

Run from any working directory with the project dependencies installed:

    python examples/target_correlation.py

The script reads ``data_OF.csv``, encodes O as 0 and F as 1, prints the
correlations, and saves them to ``results_OF/target_correlation.csv``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import pointbiserialr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data_OF.csv"
TARGET_COLUMN = "Y"
TARGET_ENCODING = {"O": 0, "F": 1}
OUTPUT_FILE = PROJECT_ROOT / "results_OF" / "target_correlation.csv"


def calculate_target_correlations(data: pd.DataFrame) -> pd.DataFrame:
    """Return point-biserial correlations and p-values for numeric features."""
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Input data must contain target column {TARGET_COLUMN!r}.")

    target = data[TARGET_COLUMN].map(TARGET_ENCODING)
    if target.isna().any():
        unknown = sorted(data.loc[target.isna(), TARGET_COLUMN].dropna().unique())
        raise ValueError(f"Target contains missing or unknown labels: {unknown}")

    features = data.drop(columns=TARGET_COLUMN).select_dtypes(include="number")
    if features.empty:
        raise ValueError("Input data contains no numeric predictor columns.")

    rows = []
    for feature in features.columns:
        valid = features[feature].notna() & target.notna()
        if valid.sum() < 2:
            raise ValueError(f"Not enough complete observations for {feature!r}.")
        result = pointbiserialr(target[valid], features.loc[valid, feature])
        rows.append(
            {
                "feature": feature,
                "target": TARGET_COLUMN,
                "target_encoding": "O=0, F=1",
                "n_observations": int(valid.sum()),
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


def main() -> None:
    """Load the project data, calculate correlations, and write the report."""
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Input data file not found: {DATA_FILE}")

    data = pd.read_csv(DATA_FILE)
    results = calculate_target_correlations(data)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print(f"Point-biserial correlations with {TARGET_COLUMN} (O=0, F=1):")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(f"\nSaved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
