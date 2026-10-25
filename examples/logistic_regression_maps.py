"""Compatibility wrapper for the unified Logistic Regression maps CLI.

Run with the existing OBGM defaults:

    python examples/logistic_regression_maps.py

Pass options such as ``--data`` and ``--results-dir`` to select another dataset.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from viclassify.probability_maps_cli import main as probability_maps_main

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data_OBGM.csv"
DEFAULT_RESULTS = PROJECT_ROOT / "results_OBGM"


def main(argv: Sequence[str] | None = None) -> None:
    """Run the shared probability-map workflow with Logistic Regression."""
    forwarded_args = sys.argv[1:] if argv is None else list(argv)
    probability_maps_main(
        [
            "--model",
            "logistic_regression",
            "--data",
            str(DEFAULT_DATA),
            "--results-dir",
            str(DEFAULT_RESULTS),
            *forwarded_args,
        ]
    )


if __name__ == "__main__":
    main()
