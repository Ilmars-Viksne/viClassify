"""Backward-compatible script and import path for viClassify."""

import sys
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parent / "src"
if _SOURCE_ROOT.is_dir():
    sys.path.insert(0, str(_SOURCE_ROOT))

from viclassify import __version__
from viclassify.cli import (
    ClassificationAnalysis,
    Config,
    OutputTransaction,
    align_probabilities_for_log_loss,
    main,
    moving_block_bootstrap_indices,
    os,
    parse_args,
    shutil,
    validate_probability_matrix,
    wilson_interval,
)

__all__ = [
    "ClassificationAnalysis",
    "Config",
    "OutputTransaction",
    "__version__",
    "align_probabilities_for_log_loss",
    "main",
    "moving_block_bootstrap_indices",
    "os",
    "parse_args",
    "shutil",
    "validate_probability_matrix",
    "wilson_interval",
]

if __name__ == "__main__":
    main()
