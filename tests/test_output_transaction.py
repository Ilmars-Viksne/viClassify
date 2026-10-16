"""Tests for output transaction protection and output policy behavior."""

import json
import shutil
import traceback
from pathlib import Path

import pytest
from conftest import run_analysis_stages

from viclassify_cli import ClassificationAnalysis, Config, OutputTransaction


def _analysis(tmp_path):
    data_file = tmp_path / "data.csv"
    data_file.write_text("X,Y\n1,A\n2,B\n", encoding="utf-8")
    cfg = Config(
        data=data_file,
        output=tmp_path / "output",
        target="Y",
        features=("X",),
        class_order=None,
        test_fraction=0.25,
        cv_folds=2,
        cv_strategy="stratified-shuffled",
        random_state=42,
        n_jobs=1,
        permutation_repeats=1,
        dpi=50,
    )
    return ClassificationAnalysis(cfg)


def _valid_generated_files():
    return {"run_metadata.json", "artifact.txt"}


def _write_valid_staging(staging):
    (staging / "run_metadata.json").write_text("{}", encoding="utf-8")
    (staging / "run_status.json").write_text("{}", encoding="utf-8")
    (staging / "artifact.txt").write_text("artifact", encoding="utf-8")
    names = sorted(
        _valid_generated_files() | {"generated_files.txt", "run_status.json"}
    )
    (staging / "generated_files.txt").write_text(
        "\n".join(names) + "\n", encoding="utf-8"
    )


def _exception_notes(exc):
    return getattr(exc, "__notes__", getattr(exc, "_viclassify_notes", []))


def test_path_requires_prepared_output(tmp_path):
    analysis = _analysis(tmp_path)
    with pytest.raises(RuntimeError, match="has not been prepared"):
        analysis.path("result.csv")
    assert not analysis.generated_files


@pytest.mark.parametrize("name", ["", "   ", "."])
def test_path_rejects_empty_and_root_names(tmp_path, name):
    analysis = _analysis(tmp_path)
    analysis.prepare_output()
    try:
        before = set(analysis.generated_files)
        with pytest.raises(ValueError):
            analysis.path(name)
        assert analysis.generated_files == before
        assert list(analysis.working_output_dir.iterdir()) == []
    finally:
        analysis.transaction.rollback()


@pytest.mark.parametrize(
    "name", ["../escaped.txt", "nested/../../escaped.txt", "reports/../.."]
)
def test_path_rejects_parent_traversal(tmp_path, name):
    analysis = _analysis(tmp_path)
    analysis.prepare_output()
    try:
        before = set(analysis.generated_files)
        with pytest.raises(ValueError, match="unsafe|outside staging"):
            analysis.path(name)
        assert analysis.generated_files == before
        assert not (tmp_path / "escaped.txt").exists()
    finally:
        analysis.transaction.rollback()


def test_path_rejects_absolute_path(tmp_path):
    analysis = _analysis(tmp_path)
    analysis.prepare_output()
    destination = (tmp_path / "absolute.txt").resolve()
    try:
        with pytest.raises(ValueError, match="must be relative"):
            analysis.path(str(destination))
        assert not analysis.generated_files
        assert not destination.exists()
    finally:
        analysis.transaction.rollback()


@pytest.mark.parametrize(
    "name",
    [
        r"nested\artifact.txt",
        r"..\outside.txt",
        r"C:\results\artifact.txt",
        r"\\server\share\artifact.txt",
    ],
)
def test_path_rejects_windows_style_artifact_paths(tmp_path, name):
    analysis = _analysis(tmp_path)
    analysis.prepare_output()
    try:
        with pytest.raises(ValueError):
            analysis.path(name)
        assert not analysis.generated_files
        assert list(analysis.working_output_dir.iterdir()) == []
    finally:
        analysis.transaction.rollback()


def test_path_registers_safe_nested_path_without_creating_it(tmp_path):
    analysis = _analysis(tmp_path)
    staging = analysis.prepare_output()
    try:
        result = analysis.path("reports/nested/result.csv")
        assert result == staging / "reports/nested/result.csv"
        assert analysis.generated_files == {"reports/nested/result.csv"}
        assert not result.exists()
        assert not result.parent.exists()
    finally:
        analysis.transaction.rollback()


@pytest.mark.parametrize(
    "manifest_text, message",
    [
        (".\n", "root-like"),
        ("../outside.txt\n", "parent traversal"),
        ("nested/../../outside.txt\n", "parent traversal"),
        ("/absolute/file.txt\n", "absolute"),
        ("nested\\artifact.txt\n", "non-normalized"),
        ("..\\outside.txt\n", "non-normalized"),
        ("C:\\results\\artifact.txt\n", "non-normalized|absolute"),
        ("\\\\server\\share\\artifact.txt\n", "non-normalized|absolute"),
        ("\n", "blank lines"),
        ("   \n", "blank lines"),
        ("generated_files.txt\n\nrun_status.json\nrun_metadata.json\n", "blank lines"),
        (
            "generated_files.txt\n \t\nrun_status.json\nrun_metadata.json\n",
            "blank lines",
        ),
    ],
)
def test_manifest_rejects_malformed_artifact_names(tmp_path, manifest_text, message):
    data_file = tmp_path / "data.csv"
    data_file.write_text("X,Y\n1,A\n2,B\n", encoding="utf-8")
    txn = OutputTransaction(data_file, tmp_path / "output")
    staging = txn.prepare()
    (staging / "run_metadata.json").write_text("{}", encoding="utf-8")
    (staging / "run_status.json").write_text("{}", encoding="utf-8")
    (staging / "generated_files.txt").write_text(manifest_text, encoding="utf-8")
    try:
        with pytest.raises(RuntimeError, match=message):
            txn.validate_staged_output(set())
    finally:
        txn.rollback()


@pytest.mark.parametrize(
    "registry_entry, message",
    [
        (r"nested\artifact.txt", "non-normalized"),
        (r"..\outside.txt", "non-normalized"),
        (r"C:\results\artifact.txt", "non-normalized|absolute"),
        (r"\\server\share\artifact.txt", "non-normalized|absolute"),
        ("/absolute/artifact.txt", "absolute"),
        ("../outside.txt", "parent traversal"),
        ("nested/../../outside.txt", "parent traversal"),
        (".", "root-like"),
        ("", "empty artifact path"),
        ("   ", "empty artifact path"),
    ],
)
def test_registry_rejects_unsafe_artifact_names(tmp_path, registry_entry, message):
    data_file = tmp_path / "data.csv"
    data_file.write_text("X,Y\n1,A\n2,B\n", encoding="utf-8")
    txn = OutputTransaction(data_file, tmp_path / "output")
    staging = txn.prepare()
    (staging / "run_metadata.json").write_text("{}", encoding="utf-8")
    (staging / "run_status.json").write_text("{}", encoding="utf-8")
    (staging / "generated_files.txt").write_text(
        "generated_files.txt\nrun_status.json\nrun_metadata.json\n", encoding="utf-8"
    )
    try:
        with pytest.raises(RuntimeError, match=f"Generated-file registry.*({message})"):
            txn.validate_staged_output({registry_entry})
    finally:
        txn.rollback()


def test_registry_safety_precedes_manifest_set_mismatch(tmp_path):
    data_file = tmp_path / "data.csv"
    data_file.write_text("X,Y\n1,A\n2,B\n", encoding="utf-8")
    txn = OutputTransaction(data_file, tmp_path / "output")
    staging = txn.prepare()
    (staging / "run_metadata.json").write_text("{}", encoding="utf-8")
    (staging / "run_status.json").write_text("{}", encoding="utf-8")
    (staging / "generated_files.txt").write_text(
        "generated_files.txt\nrun_status.json\nrun_metadata.json\n", encoding="utf-8"
    )
    try:
        with pytest.raises(
            RuntimeError, match="Generated-file registry.*parent traversal"
        ):
            txn.validate_staged_output(
                {"../outside.txt", "run_status.json", "generated_files.txt"}
            )
    finally:
        txn.rollback()


def test_run_rolls_back_when_running_status_fails(tmp_path, monkeypatch):
    analysis = _analysis(tmp_path)
    sentinel = RuntimeError("running status failed")

    def fail_running(status, error_info=None):
        if status == "running":
            raise sentinel

    monkeypatch.setattr(analysis, "write_run_status", fail_running)
    with pytest.raises(RuntimeError) as caught:
        analysis.run()
    assert caught.value is sentinel
    assert not analysis.working_output_dir.exists()
    assert not analysis.final_output_path.exists()
    assert not analysis.transaction.backup_path.exists()
    assert not analysis.transaction.prepared
    assert not analysis.transaction.committed


def test_failed_status_error_does_not_hide_workflow_error(tmp_path, monkeypatch):
    analysis = _analysis(tmp_path)
    original = RuntimeError("workflow failed")

    def fail_load():
        raise original

    def fail_status(status, error_info=None):
        if status == "failed":
            raise OSError("status reporting failed")

    monkeypatch.setattr(analysis, "load", fail_load)
    monkeypatch.setattr(analysis, "write_run_status", fail_status)
    with pytest.raises(RuntimeError) as caught:
        analysis.run()
    assert caught.value is original
    assert any(
        "Failed-status recording also failed" in note
        for note in _exception_notes(caught.value)
    )
    assert not analysis.working_output_dir.exists()
    assert not analysis.final_output_path.exists()
    assert not analysis.transaction.backup_path.exists()
    assert not analysis.transaction.prepared
    assert not analysis.transaction.committed


def test_workflow_error_remains_primary_after_successful_cleanup(tmp_path, monkeypatch):
    analysis = _analysis(tmp_path)
    original = RuntimeError("primary workflow failure")

    def fail_load():
        raise original

    monkeypatch.setattr(analysis, "load", fail_load)
    with pytest.raises(RuntimeError, match="primary workflow failure") as caught:
        analysis.run()

    assert caught.value is original
    assert not analysis.working_output_dir.exists()
    assert not analysis.final_output_path.exists()
    assert not analysis.transaction.prepared
    assert not analysis.transaction.committed


def test_workflow_error_remains_primary_after_rollback_failure(tmp_path, monkeypatch):
    analysis = _analysis(tmp_path)
    original = RuntimeError("primary workflow failure")
    rollback_error = OSError("rollback blocked")

    def fail_load():
        raise original

    def fail_rollback():
        raise rollback_error

    prepare_output = analysis.prepare_output
    monkeypatch.setattr(analysis, "load", fail_load)
    with pytest.raises(RuntimeError, match="primary workflow failure") as caught:
        # run() constructs its transaction during output preparation.
        monkeypatch.setattr(
            analysis,
            "prepare_output",
            lambda: _prepare_with_rollback_failure(
                analysis, prepare_output, monkeypatch, fail_rollback
            ),
        )
        analysis.run()

    assert caught.value is original
    assert any(
        "Transaction rollback also failed: OSError: rollback blocked" in note
        for note in _exception_notes(caught.value)
    )
    assert any(frame.name == "fail_load" for frame in traceback.extract_tb(caught.tb))
    assert analysis.working_output_dir.exists()
    assert not analysis.transaction.committed
    shutil.rmtree(analysis.working_output_dir, ignore_errors=True)


def test_workflow_status_and_rollback_failures_keep_primary_error(
    tmp_path, monkeypatch
):
    analysis = _analysis(tmp_path)
    original = RuntimeError("primary workflow failure")
    status_error = OSError("status reporting blocked")
    rollback_error = OSError("rollback blocked")

    def fail_load():
        raise original

    def fail_failed_status(status, error_info=None):
        if status == "failed":
            raise status_error

    def fail_rollback():
        raise rollback_error

    prepare_output = analysis.prepare_output
    monkeypatch.setattr(analysis, "load", fail_load)
    monkeypatch.setattr(analysis, "write_run_status", fail_failed_status)
    with pytest.raises(RuntimeError, match="primary workflow failure") as caught:
        monkeypatch.setattr(
            analysis,
            "prepare_output",
            lambda: _prepare_with_rollback_failure(
                analysis, prepare_output, monkeypatch, fail_rollback
            ),
        )
        analysis.run()

    assert caught.value is original
    notes = _exception_notes(caught.value)
    assert any("Failed-status recording also failed" in note for note in notes)
    assert any("Transaction rollback also failed" in note for note in notes)
    assert any(frame.name == "fail_load" for frame in traceback.extract_tb(caught.tb))
    assert analysis.working_output_dir.exists()
    assert not analysis.transaction.committed
    shutil.rmtree(analysis.working_output_dir, ignore_errors=True)


def _prepare_with_rollback_failure(
    analysis, prepare_output, monkeypatch, fail_rollback
):
    staging = prepare_output()
    monkeypatch.setattr(analysis.transaction, "rollback", fail_rollback)
    return staging


def test_running_status_failure_remains_primary_when_rollback_fails(
    tmp_path, monkeypatch
):
    analysis = _analysis(tmp_path)
    running_error = RuntimeError("running status failed")
    rollback_error = OSError("rollback blocked")
    original_write_status = analysis.write_run_status

    def fail_running_status(status, error_info=None):
        if status == "running":
            raise running_error
        return original_write_status(status, error_info)

    monkeypatch.setattr(analysis, "write_run_status", fail_running_status)

    original_prepare_output = analysis.prepare_output

    def prepare_with_failing_rollback():
        staging = original_prepare_output()

        def fail_rollback():
            raise rollback_error

        monkeypatch.setattr(analysis.transaction, "rollback", fail_rollback)
        return staging

    monkeypatch.setattr(analysis, "prepare_output", prepare_with_failing_rollback)
    with pytest.raises(RuntimeError, match="running status failed") as caught:
        analysis.run()

    assert caught.value is running_error
    assert any(
        "Transaction rollback also failed" in note
        for note in _exception_notes(caught.value)
    )
    assert analysis.working_output_dir.exists()
    assert not analysis.transaction.committed
    shutil.rmtree(analysis.working_output_dir, ignore_errors=True)


def test_run_analysis_stages_requires_explicit_preparation(tmp_path):
    with pytest.raises(RuntimeError, match=r"Call analysis.prepare_output\(\)"):
        run_analysis_stages(_analysis(tmp_path), stages=[])


class TestOutputTransaction:
    """Tests for OutputTransaction and CLI output policy enforcement."""

    def test_default_refusal_when_nonempty(self, tmp_path, minimal_grids):
        """Test default output-policy 'fail' refuses to run if final directory is nonempty."""
        data_file = tmp_path / "data.csv"
        data_file.write_text(
            "X1,X2,Y\n1,10,O\n2,20,F\n3,30,O\n4,40,F\n", encoding="utf-8"
        )

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
            cv_strategy="stratified-shuffled",
            cv_min_class_count=1,
            random_state=42,
            n_jobs=1,
            permutation_repeats=1,
            dpi=50,
            output_policy="fail",
        )

        with pytest.raises(RuntimeError, match="exists and is nonempty"):
            ClassificationAnalysis(cfg).run()

    def test_constructor_has_no_filesystem_side_effects(self, tmp_path):
        """Test constructor does not create directories or files."""
        data_file = tmp_path / "data.csv"
        data_file.write_text("X1,X2,Y\n1,10,O\n2,20,F\n", encoding="utf-8")
        output_dir = tmp_path / "should_not_exist_yet"

        cfg = Config(
            data=data_file,
            output=output_dir,
            target="Y",
            features=("X1", "X2"),
            class_order=None,
            test_fraction=0.25,
            cv_folds=2,
            cv_strategy="stratified-shuffled",
            cv_min_class_count=1,
            random_state=42,
            n_jobs=1,
            permutation_repeats=1,
            dpi=50,
            output_policy="fail",
        )

        txn = OutputTransaction(data_file, output_dir, policy="fail")
        assert not output_dir.exists()
        assert not txn.staging_path.exists()

        ClassificationAnalysis(cfg, transaction=txn)
        assert not output_dir.exists()
        assert not txn.staging_path.exists()
        assert len(list(tmp_path.glob(".*.staging-*"))) == 0

    def test_validate_staged_output_exact_set_check(self, tmp_path):
        """Test validate_staged_output enforces exact set equality with manifest."""
        data_file = tmp_path / "data.csv"
        data_file.write_text("X1,X2,Y\n1,10,O\n2,20,F\n", encoding="utf-8")
        output_dir = tmp_path / "out_test"

        txn = OutputTransaction(data_file, output_dir, policy="fail")
        staging = txn.prepare()

        (staging / "run_metadata.json").write_text("{}", encoding="utf-8")
        # Manifest missing file 'extra.txt' which is in generated_files set
        (staging / "generated_files.txt").write_text(
            "generated_files.txt\nrun_status.json\n", encoding="utf-8"
        )
        (staging / "run_status.json").write_text("{}", encoding="utf-8")
        (staging / "extra.txt").write_text("extra", encoding="utf-8")

        with pytest.raises(RuntimeError, match="Generated-file manifest mismatch"):
            txn.validate_staged_output({"extra.txt"})

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
            "X1,X2,Y\n1,10,O\n2,20,F\n3,30,O\n4,40,F\n5,50,O\n6,60,F\n7,70,O\n8,80,F\n",
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
            cv_strategy="stratified-shuffled",
            cv_min_class_count=1,
            random_state=42,
            n_jobs=1,
            permutation_repeats=1,
            dpi=50,
            output_policy="fail",
        )

        analysis = ClassificationAnalysis(cfg, grid_overrides=minimal_grids)
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
            "X1,X2,Y\n1,10,O\n2,20,F\n3,30,O\n4,40,F\n5,50,O\n6,60,F\n7,70,O\n8,80,F\n",
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
            cv_strategy="stratified-shuffled",
            cv_min_class_count=1,
            random_state=42,
            n_jobs=1,
            permutation_repeats=1,
            dpi=50,
            output_policy="fail",
        )

        analysis_good = ClassificationAnalysis(cfg_good, grid_overrides=minimal_grids)
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
            "X1,X2,Y\n1,10,O\n2,20,F\n3,30,O\n4,40,F\n5,50,O\n6,60,F\n7,70,O\n8,80,F\n",
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
            cv_strategy="stratified-shuffled",
            cv_min_class_count=1,
            random_state=42,
            n_jobs=1,
            permutation_repeats=1,
            dpi=50,
            output_policy="fail",
        )

        analysis1 = ClassificationAnalysis(cfg1, grid_overrides=minimal_grids)
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
            cv_strategy="stratified-shuffled",
            cv_min_class_count=1,
            random_state=42,
            n_jobs=1,
            permutation_repeats=1,
            dpi=50,
            output_policy="replace",
        )

        analysis2 = ClassificationAnalysis(cfg2, grid_overrides=minimal_grids)
        analysis2.run()

        assert output_dir.exists()
        assert not (output_dir / "stale_from_run_1.txt").exists()
        assert analysis2.transaction.committed
        assert not analysis2.transaction.prepared
        assert not analysis2.transaction.backup_path.exists()
        assert (output_dir / "run_metadata.json").exists()

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

    def test_unregistered_physical_files_rejected(self, tmp_path):
        """Test validate_staged_output rejects unlisted physical files at root and in nested dirs."""
        data_file = tmp_path / "data.csv"
        data_file.write_text("X1,X2,Y\n1,10,O\n2,20,F\n", encoding="utf-8")
        output_dir = tmp_path / "out_unregistered"

        txn = OutputTransaction(data_file, output_dir, policy="fail")
        staging = txn.prepare()

        (staging / "run_metadata.json").write_text("{}", encoding="utf-8")
        (staging / "run_status.json").write_text("{}", encoding="utf-8")
        (staging / "generated_files.txt").write_text(
            "generated_files.txt\nrun_status.json\nrun_metadata.json\n",
            encoding="utf-8",
        )

        # 1. Unregistered file at root
        (staging / "unregistered.txt").write_text("unregistered", encoding="utf-8")
        with pytest.raises(
            RuntimeError, match="Staged filesystem contents do not match the manifest"
        ):
            txn.validate_staged_output({"run_metadata.json"})

        # Remove root unregistered file and add nested unregistered file
        (staging / "unregistered.txt").unlink()
        nested_dir = staging / "nested"
        nested_dir.mkdir()
        (nested_dir / "unregistered.txt").write_text(
            "nested unregistered", encoding="utf-8"
        )

        with pytest.raises(
            RuntimeError, match="Staged filesystem contents do not match the manifest"
        ):
            txn.validate_staged_output({"run_metadata.json"})

        txn.rollback()

    def test_transaction_state_lifecycle_protections(self, tmp_path):
        """Test transaction state invariants and method guards."""
        data_file = tmp_path / "data.csv"
        data_file.write_text("X1,X2,Y\n1,10,O\n2,20,F\n", encoding="utf-8")
        output_dir = tmp_path / "out_state"

        txn = OutputTransaction(data_file, output_dir, policy="fail")

        # 1. Commit before prepare fails
        with pytest.raises(
            RuntimeError, match="Output transaction has not been prepared"
        ):
            txn.commit(set())

        # 2. Prepare transaction
        staging = txn.prepare()
        assert txn.prepared
        assert not txn.committed
        (staging / "run_metadata.json").write_text("{}", encoding="utf-8")
        (staging / "run_status.json").write_text("{}", encoding="utf-8")
        (staging / "generated_files.txt").write_text(
            "generated_files.txt\nrun_status.json\nrun_metadata.json\n",
            encoding="utf-8",
        )

        # 3. Repeated prepare fails
        with pytest.raises(
            RuntimeError, match="Output transaction has already been prepared"
        ):
            txn.prepare()

        # 4. Commit transaction
        txn.commit({"run_metadata.json"})
        assert not txn.prepared
        assert txn.committed

        # 5. Repeated commit fails
        with pytest.raises(
            RuntimeError, match="Output transaction has already been committed"
        ):
            txn.commit({"run_metadata.json"})

        # 6. Prepare after commit fails
        with pytest.raises(
            RuntimeError, match="A committed transaction cannot be prepared again"
        ):
            txn.prepare()

        # 7. Rollback after commit fails
        with pytest.raises(
            RuntimeError, match="A committed transaction cannot be rolled back"
        ):
            txn.rollback()

    def test_rollback_allows_clean_repreparation(self, tmp_path):
        data_file = tmp_path / "data.csv"
        data_file.write_text("X,Y\n1,A\n2,B\n", encoding="utf-8")
        txn = OutputTransaction(data_file, tmp_path / "out_reprepare")

        first_staging = txn.prepare()
        (first_staging / "old-attempt.txt").write_text("old", encoding="utf-8")
        txn.rollback()
        assert not first_staging.exists()
        assert not txn.prepared
        assert not txn.committed

        second_staging = txn.prepare()
        assert txn.prepared
        assert not txn.committed
        assert second_staging == first_staging
        assert list(second_staging.iterdir()) == []
        txn.rollback()
        assert not txn.prepared
        assert not txn.committed

    def test_failed_promotion_restores_previous_output(self, tmp_path, monkeypatch):
        data_file = tmp_path / "data.csv"
        data_file.write_text("X,Y\n1,A\n2,B\n", encoding="utf-8")
        output_dir = tmp_path / "existing"
        output_dir.mkdir()
        (output_dir / "old.txt").write_text("old output", encoding="utf-8")
        old_nested = output_dir / "archive" / "keep.bin"
        old_nested.parent.mkdir()
        old_nested.write_bytes(b"previous results")
        txn = OutputTransaction(data_file, output_dir, policy="replace")
        staging = txn.prepare()
        _write_valid_staging(staging)

        real_rename = __import__("viclassify_cli").os.rename
        promotion_error = OSError("promotion blocked")

        def fail_promotion(src, dst):
            if Path(src) == staging and Path(dst) == output_dir:
                raise promotion_error
            return real_rename(src, dst)

        monkeypatch.setattr("viclassify_cli.os.rename", fail_promotion)
        with pytest.raises(RuntimeError, match="promotion blocked") as caught:
            txn.commit(_valid_generated_files())

        assert caught.value.__cause__ is promotion_error
        assert (output_dir / "old.txt").read_text(encoding="utf-8") == "old output"
        assert old_nested.read_bytes() == b"previous results"
        assert {
            path.relative_to(output_dir).as_posix()
            for path in output_dir.rglob("*")
            if path.is_file()
        } == {"old.txt", "archive/keep.bin"}
        assert not staging.exists()
        assert not txn.backup_path.exists()
        assert not txn.prepared
        assert not txn.committed

    def test_failed_promotion_and_restore_resets_state(self, tmp_path, monkeypatch):
        data_file = tmp_path / "data.csv"
        data_file.write_text("X,Y\n1,A\n2,B\n", encoding="utf-8")
        output_dir = tmp_path / "existing"
        output_dir.mkdir()
        (output_dir / "old.txt").write_text("old output", encoding="utf-8")
        txn = OutputTransaction(data_file, output_dir, policy="replace")
        staging = txn.prepare()
        _write_valid_staging(staging)

        real_rename = __import__("viclassify_cli").os.rename
        promotion_error = OSError("promotion blocked")
        restore_error = OSError("restore blocked")

        def fail_promotion_and_restore(src, dst):
            if Path(src) == staging and Path(dst) == output_dir:
                raise promotion_error
            if Path(src) == txn.backup_path and Path(dst) == output_dir:
                raise restore_error
            return real_rename(src, dst)

        monkeypatch.setattr("viclassify_cli.os.rename", fail_promotion_and_restore)
        with pytest.raises(
            RuntimeError,
            match="Failed to promote staged output.*restore blocked.*manual recovery",
        ) as caught:
            txn.commit(_valid_generated_files())

        assert caught.value.__cause__ is promotion_error
        assert str(txn.backup_path) in str(caught.value)
        assert not staging.exists()
        assert txn.backup_path.exists()
        assert not txn.prepared
        assert not txn.committed

    def test_backup_rename_failure_resets_state_and_cleans_staging(
        self, tmp_path, monkeypatch
    ):
        data_file = tmp_path / "data.csv"
        data_file.write_text("X,Y\n1,A\n2,B\n", encoding="utf-8")
        output_dir = tmp_path / "existing"
        output_dir.mkdir()
        (output_dir / "old.txt").write_text("old output", encoding="utf-8")
        txn = OutputTransaction(data_file, output_dir, policy="replace")
        staging = txn.prepare()
        _write_valid_staging(staging)
        rename_error = OSError("backup rename blocked")
        real_rename = __import__("viclassify_cli").os.rename

        def fail_backup_rename(src, dst):
            if Path(src) == output_dir and Path(dst) == txn.backup_path:
                raise rename_error
            return real_rename(src, dst)

        monkeypatch.setattr("viclassify_cli.os.rename", fail_backup_rename)
        with pytest.raises(
            RuntimeError,
            match="Failed to prepare the final output location for promotion",
        ) as caught:
            txn.commit(_valid_generated_files())

        assert caught.value.__cause__ is rename_error
        assert (output_dir / "old.txt").read_text(encoding="utf-8") == "old output"
        assert not txn.backup_path.exists()
        assert not staging.exists()
        assert not txn.prepared
        assert not txn.committed

    def test_fail_policy_destination_removal_failure_resets_state(
        self, tmp_path, monkeypatch
    ):
        data_file = tmp_path / "data.csv"
        data_file.write_text("X,Y\n1,A\n2,B\n", encoding="utf-8")
        output_dir = tmp_path / "empty-existing"
        output_dir.mkdir()
        txn = OutputTransaction(data_file, output_dir, policy="fail")
        staging = txn.prepare()
        _write_valid_staging(staging)
        removal_error = OSError("empty destination removal blocked")

        def fail_remove_empty_destination():
            raise removal_error

        monkeypatch.setattr(
            txn,
            "_remove_empty_destination",
            fail_remove_empty_destination,
        )
        with pytest.raises(
            RuntimeError,
            match="Failed to prepare the final output location for promotion",
        ) as caught:
            txn.commit(_valid_generated_files())

        assert caught.value.__cause__ is removal_error
        assert output_dir.exists()
        assert list(output_dir.iterdir()) == []
        assert not txn.backup_path.exists()
        assert not staging.exists()
        assert not txn.prepared
        assert not txn.committed

    def test_stale_backup_is_preserved_and_commit_aborts_safely(self, tmp_path):
        data_file = tmp_path / "data.csv"
        data_file.write_text("X,Y\n1,A\n2,B\n", encoding="utf-8")
        output_dir = tmp_path / "existing"
        output_dir.mkdir()
        (output_dir / "old.txt").write_text("active output", encoding="utf-8")
        txn = OutputTransaction(data_file, output_dir, policy="replace")
        staging = txn.prepare()
        _write_valid_staging(staging)
        txn.backup_path.mkdir()
        (txn.backup_path / "recoverable.txt").write_text(
            "recovery copy", encoding="utf-8"
        )

        with pytest.raises(RuntimeError, match="Failed to prepare") as caught:
            txn.commit(_valid_generated_files())

        assert isinstance(caught.value.__cause__, FileExistsError)
        assert (output_dir / "old.txt").read_text(encoding="utf-8") == "active output"
        assert (txn.backup_path / "recoverable.txt").read_text(encoding="utf-8") == (
            "recovery copy"
        )
        assert not staging.exists()
        assert not txn.prepared
        assert not txn.committed

    def test_backup_cleanup_failure_restores_previous_output(
        self, tmp_path, monkeypatch
    ):
        data_file = tmp_path / "data.csv"
        data_file.write_text("X,Y\n1,A\n2,B\n", encoding="utf-8")
        output_dir = tmp_path / "existing"
        output_dir.mkdir()
        (output_dir / "old.txt").write_text("old output", encoding="utf-8")
        txn = OutputTransaction(data_file, output_dir, policy="replace")
        staging = txn.prepare()
        _write_valid_staging(staging)
        cleanup_error = OSError("backup cleanup blocked")
        real_rmtree = __import__("viclassify_cli").shutil.rmtree

        def fail_backup_cleanup(path, *args, **kwargs):
            if Path(path) == txn.backup_path:
                raise cleanup_error
            return real_rmtree(path, *args, **kwargs)

        monkeypatch.setattr("viclassify_cli.shutil.rmtree", fail_backup_cleanup)
        with pytest.raises(
            RuntimeError, match="remove the transaction backup"
        ) as caught:
            txn.commit(_valid_generated_files())

        assert caught.value.__cause__ is cleanup_error
        assert (output_dir / "old.txt").read_text(encoding="utf-8") == "old output"
        assert not txn.backup_path.exists()
        assert not staging.exists()
        assert not txn.prepared
        assert not txn.committed

    @pytest.mark.parametrize(
        "mutation, registered",
        [
            ("unregistered_root", {"run_metadata.json", "artifact.txt"}),
            ("unregistered_nested", {"run_metadata.json", "artifact.txt"}),
            ("listed_missing", {"run_metadata.json", "artifact.txt"}),
            ("registered_omitted", {"run_metadata.json", "artifact.txt", "extra.txt"}),
            ("manifest_unregistered", {"run_metadata.json", "artifact.txt"}),
            ("duplicate", {"run_metadata.json", "artifact.txt"}),
            ("absolute", {"run_metadata.json", "artifact.txt"}),
            ("traversal", {"run_metadata.json", "artifact.txt"}),
        ],
    )
    def test_exact_manifest_validation_preserves_existing_output(
        self, tmp_path, mutation, registered
    ):
        data_file = tmp_path / "data.csv"
        data_file.write_text("X,Y\n1,A\n2,B\n", encoding="utf-8")
        output_dir = tmp_path / "existing"
        output_dir.mkdir()
        (output_dir / "old.txt").write_text("old output", encoding="utf-8")
        txn = OutputTransaction(data_file, output_dir, policy="replace")
        staging = txn.prepare()
        _write_valid_staging(staging)
        manifest = [
            "generated_files.txt",
            "run_status.json",
            "run_metadata.json",
            "artifact.txt",
        ]

        if mutation == "unregistered_root":
            (staging / "stray.txt").write_text("stray", encoding="utf-8")
        elif mutation == "unregistered_nested":
            nested = staging / "nested"
            nested.mkdir()
            (nested / "stray.txt").write_text("stray", encoding="utf-8")
        elif mutation == "listed_missing":
            (staging / "artifact.txt").unlink()
        elif mutation == "registered_omitted":
            manifest.append("extra.txt")
        elif mutation == "manifest_unregistered":
            manifest.append("unregistered.txt")
            (staging / "unregistered.txt").write_text("listed", encoding="utf-8")
        elif mutation == "duplicate":
            manifest.append("artifact.txt")
        elif mutation == "absolute":
            manifest.append(str((tmp_path / "outside.txt").resolve()))
        elif mutation == "traversal":
            manifest.append("../outside.txt")

        (staging / "generated_files.txt").write_text(
            "\n".join(manifest) + "\n", encoding="utf-8"
        )
        try:
            with pytest.raises(RuntimeError):
                txn.commit(registered)
            assert (output_dir / "old.txt").read_text(encoding="utf-8") == "old output"
            assert not txn.backup_path.exists()
        finally:
            if not txn.committed:
                txn.rollback()

    def test_path_raises_runtime_error_if_not_prepared(self, tmp_path):
        """Test path() raises RuntimeError if output has not been prepared."""
        data_file = tmp_path / "data.csv"
        data_file.write_text("X1,X2,Y\n1,10,O\n2,20,F\n", encoding="utf-8")
        output_dir = tmp_path / "out_unprepared"

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
        with pytest.raises(
            RuntimeError, match="Output transaction has not been prepared"
        ):
            analysis.path("test.csv")
