"""Tests for CLI validation related to isolation."""

import subprocess
import sys
from pathlib import Path

import pytest


class TestCLIValidation:
    """Tests for CLI validation related to isolation."""

    def test_test_fraction_validation(self, test_data_dir, test_output_dir):
        """Test CLI rejects invalid test-fraction values."""
        # Create minimal valid data
        data = "X1,X2,Y\n0,0,O\n1,10,F\n2,20,O\n3,30,F\n"
        data_file = test_data_dir / "data.csv"
        data_file.write_text(data)
        output_dir = test_output_dir / "cli_test"

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

    def test_cv_folds_validation(self, test_data_dir, test_output_dir):
        """Test CLI rejects invalid cv-folds values."""
        data = "X1,X2,Y\n0,0,O\n1,10,F\n2,20,O\n3,30,F\n"
        data_file = test_data_dir / "data.csv"
        data_file.write_text(data)
        output_dir = test_output_dir / "cli_test"

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

    def test_duplicate_features_validation(self, test_data_dir, test_output_dir):
        """Test CLI rejects duplicate feature names."""
        data = "X1,X2,Y\n0,0,O\n1,10,F\n"
        data_file = test_data_dir / "data.csv"
        data_file.write_text(data)
        output_dir = test_output_dir / "cli_test"

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

    def test_target_in_features_validation(self, test_data_dir, test_output_dir):
        """Test CLI rejects target included in features."""
        data = "X1,X2,Y\n0,0,O\n1,10,F\n"
        data_file = test_data_dir / "data.csv"
        data_file.write_text(data)
        output_dir = test_output_dir / "cli_test"

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

    def test_empty_target_validation(self, test_data_dir, test_output_dir):
        """Test CLI rejects empty target."""
        data = "X1,X2,Y\n0,0,O\n1,10,F\n"
        data_file = test_data_dir / "data.csv"
        data_file.write_text(data)
        output_dir = test_output_dir / "cli_test"

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

    def test_invalid_n_jobs_validation(self, test_data_dir, test_output_dir):
        """Test CLI rejects invalid n-jobs values."""
        data = "X1,X2,Y\n0,0,O\n1,10,F\n"
        data_file = test_data_dir / "data.csv"
        data_file.write_text(data)
        output_dir = test_output_dir / "cli_test"

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

    def test_invalid_selection_metric_validation(self, test_data_dir, test_output_dir):
        """Test CLI rejects invalid selection-metric values."""
        data = "X1,X2,Y\n0,0,O\n1,10,F\n"
        data_file = test_data_dir / "data.csv"
        data_file.write_text(data)
        output_dir = test_output_dir / "cli_test"

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

    @pytest.mark.integration
    def test_valid_cli_runs(self, test_data_dir, test_output_dir):
        """Test valid CLI arguments run without error."""
        # 12 rows, 2 classes (O and F, 6 each), test_fraction=0.25 -> split=9 (training=9, holdout=3)
        # Training (rows 0-8) has 5 O's and 4 F's -> rarest class has 4 >= 2
        rows = [f"{i},{i * 10},{'O' if i % 2 == 0 else 'F'}" for i in range(12)]
        data = "X1,X2,Y\n" + "\n".join(rows) + "\n"
        data_file = test_data_dir / "data.csv"
        data_file.write_text(data)
        output_dir = test_output_dir / "cli_test"

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
            check=False,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=180,
        )

        # Should run successfully (may have warnings but not errors)
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
