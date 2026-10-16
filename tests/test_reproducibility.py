"""Packaging, provenance, and compatibility regression tests."""

import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest

import viclassify
from viclassify.cli import ClassificationAnalysis, Config
from viclassify.provenance import (
    MODEL_ARTIFACT_SCHEMA_VERSION,
    calculate_lock_file_sha256,
    check_artifact_compatibility,
    collect_environment_provenance,
    environment_package_snapshot,
    hash_source_tree,
)

ROOT = Path(__file__).resolve().parents[1]


def current_compatibility():
    environment = collect_environment_provenance(ROOT / "src" / "viclassify")
    packages = environment["package_versions"]
    return {
        "artifact_schema_version": MODEL_ARTIFACT_SCHEMA_VERSION,
        "viclassify_version": viclassify.__version__,
        "python_version": environment["python_version"],
        "python_major_minor": environment["python_major_minor"],
        "scikit_learn_version": packages["scikit_learn"],
        "numpy_version": packages["numpy"],
        "scipy_version": packages["scipy"],
        "pandas_version": packages["pandas"],
        "joblib_version": packages["joblib"],
        "source_sha256": environment["source_sha256"],
        "dependency_lock_sha256": environment["dependency_lock_sha256"],
    }


def test_version_is_consistent_across_package_and_distribution():
    assert version("viclassify") == viclassify.__version__


@pytest.mark.parametrize(
    "command",
    [
        ["viclassify", "--version"],
        [sys.executable, "-m", "viclassify", "--version"],
        [sys.executable, "viclassify_cli.py", "--version"],
    ],
)
def test_all_entry_points_report_same_version(command):
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == f"viclassify {viclassify.__version__}"


def test_environment_json_is_valid_and_excludes_private_machine_identity():
    result = subprocess.run(
        [sys.executable, "-m", "viclassify", "--show-environment", "--json"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    environment = json.loads(result.stdout)
    assert environment["viclassify_version"] == viclassify.__version__
    assert environment["source_sha256"]
    assert "git_status" in environment
    assert "dependency_lock_status" in environment
    assert "hostname" not in environment
    assert "username" not in environment
    assert "home" not in environment


def test_source_tree_hash_is_stable_and_content_sensitive(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.py").write_text("x = 1\n", encoding="utf-8")
    (source / "b.py").write_text("y = 2\n", encoding="utf-8")
    (source / "notes.txt").write_text("ignored", encoding="utf-8")
    initial = hash_source_tree(source)
    assert initial == hash_source_tree(source)
    (source / "notes.txt").write_text("changed", encoding="utf-8")
    assert initial == hash_source_tree(source)
    (source / "b.py").write_text("y = 3\n", encoding="utf-8")
    assert initial != hash_source_tree(source)
    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "cache.py").write_text("z = 1", encoding="utf-8")
    assert hash_source_tree(source) == hash_source_tree(source)


def test_lock_hash_available_and_changes_with_content(tmp_path):
    assert calculate_lock_file_sha256(tmp_path) is None
    lock = tmp_path / "uv.lock"
    lock.write_text("version = 1\n", encoding="utf-8")
    first = calculate_lock_file_sha256(tmp_path)
    assert first
    assert calculate_lock_file_sha256(tmp_path) == first
    lock.write_text("version = 2\n", encoding="utf-8")
    assert calculate_lock_file_sha256(tmp_path) != first


def test_environment_package_snapshot_is_sorted_and_newline_terminated():
    snapshot = environment_package_snapshot()
    lines = snapshot.splitlines()
    assert lines == sorted(lines, key=lambda line: line.split("==", 1)[0].lower())
    assert not snapshot or snapshot.endswith("\n")


def test_exact_artifact_environment_is_compatible():
    issues = check_artifact_compatibility(
        {"model": object(), "compatibility": current_compatibility()}
    )
    assert issues == []


def test_legacy_artifact_warns_without_crashing():
    issues = check_artifact_compatibility({"model": object()})
    assert any(issue.severity == "warning" for issue in issues)
    assert not any(issue.severity == "error" for issue in issues)


def test_future_schema_and_missing_model_are_errors():
    compat = current_compatibility()
    compat["artifact_schema_version"] = "2.0"
    issues = check_artifact_compatibility({"model": object(), "compatibility": compat})
    assert any(issue.severity == "error" for issue in issues)
    assert any(issue.field == "artifact_schema_version" for issue in issues)
    missing_model = check_artifact_compatibility(
        {"compatibility": current_compatibility()}
    )
    assert any(
        issue.field == "model" and issue.severity == "error" for issue in missing_model
    )


def test_major_version_source_and_lock_mismatches_are_classified():
    compat = current_compatibility()
    compat.update(
        {
            "python_version": "99.0.0",
            "scikit_learn_version": "99.0.0",
            "numpy_version": "99.0.0",
            "source_sha256": "different",
            "dependency_lock_sha256": "different",
        }
    )
    issues = check_artifact_compatibility({"model": object(), "compatibility": compat})
    severities = {issue.field: issue.severity for issue in issues}
    assert severities["python_version"] == "error"
    assert severities["scikit_learn_version"] == "error"
    assert severities["numpy_version"] == "warning"
    assert severities["source_sha256"] == "warning"
    assert severities["dependency_lock_sha256"] == "warning"


def test_model_inspection_strict_mode_rejects_a_warning(tmp_path):
    import joblib

    compatibility = current_compatibility()
    compatibility["source_sha256"] = "different-source"
    artifact_path = tmp_path / "model.joblib"
    joblib.dump(
        {
            "model": "trusted test fixture",
            "model_role": "test_fixture",
            "model_type": "fixture",
            "features": ["X1"],
            "provenance": {},
            "model_context": {},
            "compatibility": compatibility,
        },
        artifact_path,
    )
    default = subprocess.run(
        [sys.executable, "-m", "viclassify", "--inspect-model", str(artifact_path)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    strict = subprocess.run(
        [
            sys.executable,
            "-m",
            "viclassify",
            "--inspect-model",
            str(artifact_path),
            "--strict-compatibility",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert default.returncode == 0
    assert "WARNING: Source hashes differ" in default.stdout
    assert strict.returncode != 0


@pytest.mark.integration
def test_completed_run_exports_environment_and_artifact_compatibility(
    tmp_path, minimal_grids
):
    import joblib

    rows = [
        f"{index},{index * 0.5},{'A' if index % 2 == 0 else 'B'}" for index in range(40)
    ]
    data = tmp_path / "ordered.csv"
    data.write_text("X1,X2,Y\n" + "\n".join(rows) + "\n", encoding="utf-8")
    output = tmp_path / "run"
    config = Config(
        data=data,
        output=output,
        target="Y",
        features=("X1", "X2"),
        class_order=("A", "B"),
        test_fraction=0.2,
        cv_folds=2,
        random_state=31,
        n_jobs=1,
        permutation_repeats=1,
        dpi=50,
    )
    analysis = ClassificationAnalysis(config, grid_overrides=minimal_grids)
    analysis.run()

    metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    environment = json.loads((output / "environment.json").read_text(encoding="utf-8"))
    package_lines = (
        (output / "environment_packages.txt").read_text(encoding="utf-8").splitlines()
    )
    artifact = joblib.load(output / "evaluation_logistic_regression.joblib")
    assert environment["viclassify_version"] == viclassify.__version__
    assert environment["source_sha256"] == metadata["source_sha256"]
    assert environment["input_sha256"] == metadata["input_sha256"]
    assert environment["run_id"] == metadata["run_id"]
    assert metadata["dependency_lock_status"] == environment["dependency_lock_status"]
    assert metadata["git_commit"] == environment["git_commit"]
    assert environment["git_status"] in {"available", "unavailable"}
    assert package_lines == sorted(
        package_lines, key=lambda line: line.split("==", 1)[0].lower()
    )
    assert artifact["compatibility"]["source_sha256"] == metadata["source_sha256"]
    assert "environment.json" in (output / "generated_files.txt").read_text(
        encoding="utf-8"
    )


def test_deterministic_mode_forces_single_worker(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["viclassify", "--data", "example.csv", "--deterministic", "--n-jobs", "4"],
    )
    from viclassify.cli import parse_args

    with pytest.warns(
        UserWarning, match="forces viClassify estimator parallelism to one"
    ):
        args = parse_args()
    assert args.n_jobs == 1
    assert args.deterministic is True
