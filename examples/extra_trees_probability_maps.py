"""Compatibility wrapper for the unified Extra Trees probability-map CLI.

Run with the project defaults:

    python examples/extra_trees_probability_maps.py

Pass options such as ``--data`` and ``--results-dir`` to use another dataset.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from viclassify.probability_maps_cli import main as probability_maps_main


def main(argv: Sequence[str] | None = None) -> None:
    """Run the shared probability-map workflow with the Extra Trees model."""
    forwarded_args = sys.argv[1:] if argv is None else argv
    probability_maps_main(["--model", "extra_trees", *forwarded_args])


if __name__ == "__main__":
    main()
