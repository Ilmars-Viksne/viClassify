"""Compatibility wrapper for the unified Histogram Gradient Boosting maps CLI.

Run with the project defaults:

    python examples/histogram_gradient_boosting_probability_maps.py

Pass options such as ``--data`` and ``--results-dir`` to select another dataset.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from viclassify.probability_maps_cli import main as probability_maps_main


def main(argv: Sequence[str] | None = None) -> None:
    """Run the shared maps workflow with Histogram Gradient Boosting."""
    forwarded_args = sys.argv[1:] if argv is None else argv
    probability_maps_main(["--model", "histogram_gradient_boosting", *forwarded_args])


if __name__ == "__main__":
    main()
