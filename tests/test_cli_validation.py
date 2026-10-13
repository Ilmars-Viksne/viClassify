"""Tests for CLI validation related to isolation."""

import subprocess
import sys
from pathlib import Path

import pytest

from viclassify_cli import ClassificationAnalysis, Config, parse_args


class TestCLIValidation:
    """Tests for CLI validation related to isolation."""

    def test_test_fraction_validation(self, tmp_path):
        """Test CLI rejects invalid test-fraction values."""
        # Create minimal valid data
        data = "X1,X2,Y\n0,0,O\n1,10,F\n2,20,O\n3,30,F\n"
        data_file = tmp_path / "data.csv"
        data_file.write_text(data, encoding="utf-8")
        output_dir = tmp_path / "cli_test"

        # Test test-fraction <= 0
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "viclassify_cli",
                "--data",
                str(data_file),
                "--output",
                str(output_dir),
                "--test-fraction",
                "0",
                "--cv-folds",
                "2",
                "--n-jobs",
                "1",
                "--permutation-repeats",
                "1",
                "--dpi",
                "50",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            check=False,
        )
        assert result.returncode != 0
        assert "test-fraction must be between 0 and 1" in result.stderr

        # Test test-fraction >= 1
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "viclassify_cli",
                "--data",
                str(data_file),
                "--output",
                str(output_dir),
                "--test-fraction",
                "1",
                "--cv-folds",
                "2",
                "--n-jobs",
                "1",
                "--permutation-repeats",
                "1",
                "--dpi",
                "50",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            check=False,
        )
        assert result.returncode != 0
        assert "test-fraction must be between 0 and 1" in result.stderr

    def test_cv_folds_validation(self, tmp_path):
        """Test CLI rejects invalid cv-folds values."""
        data = "X1,X2,Y\n0,0,O\n1,10,F\n2,20,O\n3,30,F\n"
        data_file = tmp_path / "data.csv"
        data_file.write_text(data, encoding="utf-8")
        output_dir = tmp_path / "cli_test"

        # Test cv-folds < 2
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "viclassify_cli",
                "--data",
                str(data_file),
                "--output",
                str(output_dir),
                "--cv-folds",
                "1",
                "--n-jobs",
                "1",
                "--permutation-repeats",
                "1",
                "--dpi",
                "50",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            check=False,
        )
        assert result.returncode != 0
        assert "cv-folds must be at least 2" in result.stderr

    def test_duplicate_features_validation(self, tmp_path):
        """Test CLI rejects duplicate feature names."""
        data = "X1,X2,Y\n0,0,O\n1,10,F\n"
        data_file = tmp_path / "data.csv"
        data_file.write_text(data, encoding="utf-8")
        output_dir = tmp_path / "cli_test"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "viclassify_cli",
                "--data",
                str(data_file),
                "--output",
                str(output_dir),
                "--features",
                "X1",
                "X1",
                "--n-jobs",
                "1",
                "--permutation-repeats",
                "1",
                "--dpi",
                "50",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            check=False,
        )
        assert result.returncode != 0
        assert "duplicate column names" in result.stderr

    def test_target_in_features_validation(self, tmp_path):
        """Test CLI rejects target included in features."""
        data = "X1,X2,Y\n0,0,O\n1,10,F\n"
        data_file = tmp_path / "data.csv"
        data_file.write_text(data, encoding="utf-8")
        output_dir = tmp_path / "cli_test"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "viclassify_cli",
                "--data",
                str(data_file),
                "--output",
                str(output_dir),
                "--target",
                "Y",
                "--features",
                "X1",
                "Y",
                "--n-jobs",
                "1",
                "--permutation-repeats",
                "1",
                "--dpi",
                "50",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            check=False,
        )
        assert result.returncode != 0
        assert "target must not also appear in --features" in result.stderr

    def test_empty_target_validation(self, tmp_path):
        """Test CLI rejects empty target."""
        data = "X1,X2,Y\n0,0,O\n1,10,F\n"
        data_file = tmp_path / "data.csv"
        data_file.write_text(data, encoding="utf-8")
        output_dir = tmp_path / "cli_test"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "viclassify_cli",
                "--data",
                str(data_file),
                "--output",
                str(output_dir),
                "--target",
                "",
                "--n-jobs",
                "1",
                "--permutation-repeats",
                "1",
                "--dpi",
                "50",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            check=False,
        )
        assert result.returncode != 0
        assert "target must not be empty" in result.stderr

    def test_invalid_n_jobs_validation(self, tmp_path):
        """Test CLI rejects invalid n-jobs values."""
        data = "X1,X2,Y\n0,0,O\n1,10,F\n"
        data_file = tmp_path / "data.csv"
        data_file.write_text(data, encoding="utf-8")
        output_dir = tmp_path / "cli_test"

        # Test n-jobs = 0
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "viclassify_cli",
                "--data",
                str(data_file),
                "--output",
                str(output_dir),
                "--n-jobs",
                "0",
                "--permutation-repeats",
                "1",
                "--dpi",
                "50",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            check=False,
        )
        assert result.returncode != 0
        assert "n-jobs must be -1 or a positive integer" in result.stderr

        # Test n-jobs < -1
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "viclassify_cli",
                "--data",
                str(data_file),
                "--output",
                str(output_dir),
                "--n-jobs",
                "-2",
                "--permutation-repeats",
                "1",
                "--dpi",
                "50",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            check=False,
        )
        assert result.returncode != 0
        assert "n-jobs must be -1 or a positive integer" in result.stderr

    def test_invalid_selection_metric_validation(self, tmp_path):
        """Test CLI rejects invalid selection-metric values."""
        data = "X1,X2,Y\n0,0,O\n1,10,F\n"
        data_file = tmp_path / "data.csv"
        data_file.write_text(data, encoding="utf-8")
        output_dir = tmp_path / "cli_test"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "viclassify_cli",
                "--data",
                str(data_file),
                "--output",
                str(output_dir),
                "--selection-metric",
                "invalid_metric",
                "--n-jobs",
                "1",
                "--permutation-repeats",
                "1",
                "--dpi",
                "50",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            check=False,
        )
        assert result.returncode != 0
        assert "invalid choice" in result.stderr.lower()

    def test_cli_version_flag(self):
        """Test --version CLI flag returns application version."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "viclassify_cli",
                "--version",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            check=False,
        )
        assert result.returncode == 0
        assert "0.4.0" in result.stdout

    def test_cli_parsing(self, tmp_path, monkeypatch):
        """Test CLI parse_args correctly parses options."""
        data_file = tmp_path / "data.csv"
        data_file.write_text("X1,X2,Y\n1,10,O\n2,20,F\n", encoding="utf-8")
        output_dir = tmp_path / "cli_out"

        monkeypatch.setattr(
            "sys.argv",
            [
                "viclassify_cli.py",
                "--data",
                str(data_file),
                "--output",
                str(output_dir),
                "--features",
                "X1",
                "X2",
                "--test-fraction",
                "0.25",
                "--cv-folds",
                "2",
                "--n-jobs",
                "1",
                "--permutation-repeats",
                "1",
                "--dpi",
                "50",
                "--selection-metric",
                "macro_f1",
            ],
        )

        args = parse_args()
        assert args.data == data_file
        assert args.output == output_dir
        assert args.features == ["X1", "X2"]
        assert args.test_fraction == 0.25
        assert args.cv_folds == 2

    @pytest.mark.integration
    def test_valid_cli_runs(self, tmp_path, minimal_grids):
        """Test valid CLI run executes end-to-end using reduced grid."""
        rows = [f"{i},{i * 10},{'O' if i % 2 == 0 else 'F'}" for i in range(12)]
        data = "X1,X2,Y\n" + "\n".join(rows) + "\n"
        data_file = tmp_path / "data.csv"
        data_file.write_text(data, encoding="utf-8")
        output_dir = tmp_path / "cli_test"

        cfg = Config(
            data=data_file,
            output=output_dir,
            target="Y",
            features=("X1", "X2"),
            class_order=None,
            test_fraction=0.25,
            cv_folds=2,
            random_state=42,
            n_jobs=1,
            permutation_repeats=1,
            dpi=50,
            selection_metric="macro_f1",
        )

        analysis = ClassificationAnalysis(cfg, grid_overrides=minimal_grids)
        analysis.run()

        assert output_dir.exists()
        assert (output_dir / "generated_files.txt").exists()
        assert (output_dir / "chronological_holdout_metrics.csv").exists()
