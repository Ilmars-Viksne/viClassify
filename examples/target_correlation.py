"""Compatibility wrapper for the target-correlation CLI.

Examples:

    python examples/target_correlation.py
    python examples/target_correlation.py --data data_OBGM.csv
    python examples/target_correlation.py --data data_OBGM.csv --target Y
"""

from __future__ import annotations

from viclassify.target_correlation import main

if __name__ == "__main__":
    main()
