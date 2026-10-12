"""Tests for output transaction protection and output policy behavior."""

import json
from pathlib import Path

import pytest

from viclassify_cli import ClassificationAnalysis, Config, OutputTransaction


class TestOutputTransaction:
    """Tests for OutputTransaction and CLI output policy enforcement."""

    def test_default_refusal_when_nonempty(self, tmp_path, minimal_grids):
        """Test default output-policy 'fail' refuses to run if final directory is nonempty."""
        data_file = tmp_path / "data.csv"
        data_file.write_text("X1,X2,Y\n1,10,O\n2,20,F\n3,30,O\n4,40,F\n", encoding="utf-8")

        output_dir = tmp_path / "existing_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "stale_file.txt").write_text("old content", encoding="utf-8")

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
            output_policy="fail",
        )

        with pytest.raises(RuntimeError, match="exists and is nonempty"):
            ClassificationAnalysis(cfg)

    def test_acceptance_of_absent_directory(self, tmp_path):
        """Test absent output directory is accepted."""
        data_file = tmp_path / "data.csv"
        data_file.write_text("X1,X2,Y\n1,10,O\n2,20,F\n", encoding="utf-8")
        output_dir = tmp_path / "new_output"

        txn = OutputTransaction(data_file, output_dir, policy="fail")
        staging = txn.prepare()

        assert staging.exists()
        assert not output_dir.exists()
        txn.rollback()

    def test_acceptance_of_existing_empty_directory(self, tmp_path):
        """Test existing empty directory is accepted under 'fail' policy."""
        data_file = tmp_path / "data.csv"
        data_file.write_text("X1,X2,Y\n1,10,O\n2,20,F\n", encoding="utf-8")

        output_dir = tmp_path / "empty_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        txn = OutputTransaction(data_file, output_dir, policy="fail")
        staging = txn.prepare()

        assert staging.exists()
        txn.rollback()

    def test_successful_transaction_commit(self, tmp_path, minimal_grids):
        """Test successful analysis run commits staging to final directory."""
        data_file = tmp_path / "data.csv"
        data_file.write_text(
            "X1,X2,Y\n1,10,O\n2,20,F\n3,30,O\n4,40,F\n"
            "5,50,O\n6,60,F\n7,70,O\n8,80,F\n",
            encoding="utf-8",
        )
        output_dir = tmp_path / "final_results"

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
            output_policy="fail",
        )

        analysis = ClassificationAnalysis(cfg)
        analysis.grids = minimal_grids
        analysis.run()

        assert output_dir.exists()
        assert (output_dir / "generated_files.txt").exists()
        assert (output_dir / "run_metadata.json").exists()

        # Check metadata contains final requested path, not staging path
        with (output_dir / "run_metadata.json").open(encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["configuration"]["output"] == str(output_dir.resolve())

        # Check manifest contains relative paths with no staging references
        manifest_text = (output_dir / "generated_files.txt").read_text(encoding="utf-8")
        assert "staging-" not in manifest_text

    def test_failure_during_analysis_leaves_no_partial_final_run(self, tmp_path):
        """Test failure during analysis removes staging and leaves no partial output dir."""
        data_file = tmp_path / "invalid_data.csv"
        # Invalid data: target missing labels
        data_file.write_text("X1,X2,Y\n1,10,\n2,20,\n", encoding="utf-8")
        output_dir = tmp_path / "partial_output"

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
            output_policy="fail",
        )

        with pytest.raises(ValueError, match="Target contains missing or empty labels"):
            analysis = ClassificationAnalysis(cfg)
            analysis.run()

        assert not output_dir.exists()
        # Verify no orphan staging directories remain in parent
        staging_dirs = list(tmp_path.glob(".*.staging-*"))
        assert len(staging_dirs) == 0

    def test_existing_output_remains_unchanged_after_failed_replacement_run(
        self, tmp_path, minimal_grids
    ):
        """Test a failed run with policy='replace' leaves prior final directory untouched."""
        data_file = tmp_path / "data.csv"
        data_file.write_text(
            "X1,X2,Y\n1,10,O\n2,20,F\n3,30,O\n4,40,F\n"
            "5,50,O\n6,60,F\n7,70,O\n8,80,F\n",
            encoding="utf-8",
        )
        output_dir = tmp_path / "results"

        cfg_good = Config(
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
            output_policy="fail",
        )

        analysis_good = ClassificationAnalysis(cfg_good)
        analysis_good.grids = minimal_grids
        analysis_good.run()

        # Add a marker file to existing output
        (output_dir / "prior_marker.txt").write_text("prior_run_data", encoding="utf-8")

        # Now attempt a run that fails due to bad target feature setup
        bad_data_file = tmp_path / "bad_data.csv"
        bad_data_file.write_text("X1,X2,Y\n1,10,\n2,20,\n", encoding="utf-8")

        cfg_bad = Config(
            data=bad_data_file,
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
            output_policy="replace",
        )

        with pytest.raises(ValueError):
            analysis_bad = ClassificationAnalysis(cfg_bad)
            analysis_bad.run()

        # Prior directory should still exist and contain prior_marker.txt
        assert output_dir.exists()
        assert (output_dir / "prior_marker.txt").exists()
        assert (output_dir / "prior_marker.txt").read_text(
            encoding="utf-8"
        ) == "prior_run_data"

    def test_successful_replacement_removes_stale_files(self, tmp_path, minimal_grids):
        """Test policy='replace' removes stale files from previous run."""
        data_file = tmp_path / "data.csv"
        data_file.write_text(
            "X1,X2,Y\n1,10,O\n2,20,F\n3,30,O\n4,40,F\n"
            "5,50,O\n6,60,F\n7,70,O\n8,80,F\n",
            encoding="utf-8",
        )
        output_dir = tmp_path / "results"

        cfg1 = Config(
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
            output_policy="fail",
        )

        analysis1 = ClassificationAnalysis(cfg1)
        analysis1.grids = minimal_grids
        analysis1.run()

        stale_file = output_dir / "stale_from_run_1.txt"
        stale_file.write_text("old file", encoding="utf-8")

        cfg2 = Config(
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
            output_policy="replace",
        )

        analysis2 = ClassificationAnalysis(cfg2)
        analysis2.grids = minimal_grids
        analysis2.run()

        assert output_dir.exists()
        assert not (output_dir / "stale_from_run_1.txt").exists()

    def test_dangerous_output_paths_rejected(self, tmp_path):
        """Test rejection of dangerous output paths (cwd, input path, input parent)."""
        data_dir = tmp_path / "data_folder"
        data_dir.mkdir()
        data_file = data_dir / "input_data.csv"
        data_file.write_text("X1,X2,Y\n1,10,O\n2,20,F\n", encoding="utf-8")

        # 1. Output equals input path
        txn1 = OutputTransaction(data_file, data_file, policy="fail")
        with pytest.raises(ValueError, match="equal to input CSV path"):
            txn1.prepare()

        # 2. Output equals input parent directory
        txn2 = OutputTransaction(data_file, data_dir, policy="fail")
        with pytest.raises(ValueError, match="input CSV parent directory"):
            txn2.prepare()

        # 3. Output equals current working directory
        cwd = Path.cwd().resolve()
        txn3 = OutputTransaction(data_file, cwd, policy="fail")
        with pytest.raises(ValueError, match="current working directory"):
            txn3.prepare()
