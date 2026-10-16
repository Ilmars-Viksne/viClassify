"""Environment, source, lock, and model compatibility provenance helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from viclassify.version import __version__

RUN_METADATA_SCHEMA_VERSION = "1.1"
MODEL_ARTIFACT_SCHEMA_VERSION = "1.1"
ENVIRONMENT_SCHEMA_VERSION = "1.0"
DIRECT_DISTRIBUTIONS = {
    "joblib": "joblib",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit_learn": "scikit-learn",
    "scipy": "scipy",
}
THREAD_ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class CompatibilityIssue:
    severity: str
    field: str
    expected: str | None
    actual: str | None
    message: str


def normalize_distribution_name(name: str) -> str:
    """Return a PEP 503 normalized distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_source_tree(source_root: Path) -> str:
    """Hash Python source deterministically using relative names and contents."""
    digest = hashlib.sha256()
    for path in sorted(source_root.rglob("*.py")):
        if any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def find_repository_root(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def calculate_lock_file_sha256(repository_root: Path | None) -> str | None:
    if repository_root is None:
        return None
    lock_path = repository_root / "uv.lock"
    if not lock_path.is_file():
        return None
    return hash_file(lock_path)


def collect_git_provenance(repository_root: Path | None) -> dict[str, Any]:
    unavailable = {
        "git_commit": None,
        "git_dirty": None,
        "git_describe": None,
        "git_status": "unavailable",
    }
    if repository_root is None or not (repository_root / ".git").exists():
        return unavailable
    try:

        def git(*args: str) -> str:
            result = subprocess.run(
                ["git", *args],
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()

        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain"))
        try:
            describe = git("describe", "--tags", "--always", "--dirty")
        except (OSError, subprocess.CalledProcessError):
            describe = commit[:12]
        return {
            "git_commit": commit,
            "git_dirty": dirty,
            "git_describe": describe,
            "git_status": "available",
        }
    except (OSError, subprocess.CalledProcessError):
        return unavailable


def _numpy_build_configuration() -> dict[str, str]:
    try:
        dependencies = np.__config__.CONFIG["Build Dependencies"]
        result: dict[str, str] = {}
        for name in ("blas", "lapack"):
            info = dependencies.get(name, {})
            library = info.get("name", "unavailable")
            version = info.get("version")
            result[name] = f"{library} {version}" if version else str(library)
        return result
    except (AttributeError, KeyError, TypeError):
        return {"blas": "unavailable", "lapack": "unavailable"}


def collect_environment_provenance(
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Collect controlled environment metadata without host/user identifiers."""
    package_versions: dict[str, str | None] = {}
    for key, distribution in sorted(DIRECT_DISTRIBUTIONS.items()):
        try:
            package_versions[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            package_versions[key] = None

    root = find_repository_root(source_root or Path(__file__).resolve())
    lock_hash = calculate_lock_file_sha256(root)
    source_hash = hash_source_tree(source_root or Path(__file__).resolve().parent)
    return {
        "environment_schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "viclassify_version": __version__,
        "python_version": platform.python_version(),
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "python_implementation": platform.python_implementation(),
        "python_executable": Path(sys.executable).name,
        "platform": platform.platform(),
        "operating_system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unavailable",
        "package_versions": package_versions,
        "numpy_build_configuration": _numpy_build_configuration(),
        "thread_environment": {
            key: os.environ.get(key) for key in THREAD_ENVIRONMENT_KEYS
        },
        "source_sha256": source_hash,
        "source_hash_scope": "all Python files under src/viclassify",
        "dependency_lock_file": "uv.lock" if lock_hash else None,
        "dependency_lock_sha256": lock_hash,
        "dependency_lock_status": "available" if lock_hash else "not_available",
        **collect_git_provenance(root),
    }


def environment_package_snapshot() -> str:
    """Return a stable complete snapshot of installed distributions."""
    entries: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        version = distribution.version
        if name and version:
            normalized = normalize_distribution_name(name)
            entries[normalized] = version
    return "".join(f"{name}=={entries[name]}\n" for name in sorted(entries))


def _major_minor(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    match = re.match(r"^(\d+)\.(\d+)", value)
    return (int(match.group(1)), int(match.group(2))) if match else None


def check_artifact_compatibility(
    artifact: dict[str, Any],
    current_environment: dict[str, Any] | None = None,
) -> list[CompatibilityIssue]:
    """Compare artifact structure and runtime versions with this environment."""
    current = current_environment or collect_environment_provenance()
    issues: list[CompatibilityIssue] = []
    if not isinstance(artifact, dict):
        return [
            CompatibilityIssue(
                "error", "artifact", None, None, "Artifact must be a dictionary."
            )
        ]
    if (
        not any(key in artifact for key in ("model", "estimator"))
        or artifact.get("model", artifact.get("estimator")) is None
    ):
        issues.append(
            CompatibilityIssue(
                "error", "model", None, None, "Artifact has no estimator object."
            )
        )

    compatibility = artifact.get("compatibility")
    if not isinstance(compatibility, dict):
        compatibility = {}
        issues.append(
            CompatibilityIssue(
                "warning",
                "compatibility",
                "schema 1.1",
                None,
                "Artifact has no compatibility metadata; it may predate schema 1.1.",
            )
        )

    schema = compatibility.get("artifact_schema_version")
    if schema:
        schema_major = str(schema).split(".", maxsplit=1)[0]
        if schema_major != "1":
            issues.append(
                CompatibilityIssue(
                    "error",
                    "artifact_schema_version",
                    "1.x",
                    str(schema),
                    "Artifact schema major version is unsupported.",
                )
            )
    elif compatibility:
        issues.append(
            CompatibilityIssue(
                "warning",
                "artifact_schema_version",
                "1.1",
                None,
                "Artifact schema version is missing.",
            )
        )

    artifact_version = compatibility.get("viclassify_version")
    if artifact_version and artifact_version != __version__:
        issues.append(
            CompatibilityIssue(
                "warning",
                "viclassify_version",
                str(artifact_version),
                __version__,
                "Fitting and current viClassify versions differ.",
            )
        )

    fields = (
        ("python_version", "python_version", "major_minor"),
        ("scikit_learn_version", "scikit_learn", "major_minor"),
        ("numpy_version", "numpy", "minor"),
        ("scipy_version", "scipy", "minor"),
        ("pandas_version", "pandas", "minor"),
        ("joblib_version", "joblib", "minor"),
    )
    current_versions = current.get("package_versions", {})
    for field, current_key, strictness in fields:
        expected = compatibility.get(field)
        actual = (
            current.get("python_version")
            if field == "python_version"
            else current_versions.get(current_key)
        )
        if expected is None or actual is None:
            issues.append(
                CompatibilityIssue(
                    "warning",
                    field,
                    expected,
                    actual,
                    f"Compatibility version {field} is unavailable.",
                )
            )
            continue
        expected_pair = _major_minor(str(expected))
        actual_pair = _major_minor(str(actual))
        mismatch = False
        severity = "warning"
        if strictness == "major":
            mismatch = (
                expected_pair is None
                or actual_pair is None
                or expected_pair[0] != actual_pair[0]
            )
            severity = "error" if mismatch else "warning"
        elif strictness == "major_minor":
            mismatch = expected_pair != actual_pair
            major_mismatch = bool(
                mismatch
                and expected_pair
                and actual_pair
                and expected_pair[0] != actual_pair[0]
            )
            severity = (
                "error"
                if major_mismatch
                and (current_key == "scikit_learn" or field == "python_version")
                else "warning"
            )
        else:
            mismatch = expected_pair != actual_pair
        if mismatch:
            issues.append(
                CompatibilityIssue(
                    severity,
                    field,
                    str(expected),
                    str(actual),
                    f"Fitting and current {field} differ.",
                )
            )

    source_expected = compatibility.get("source_sha256")
    source_actual = current.get("source_sha256")
    if source_expected != source_actual:
        issues.append(
            CompatibilityIssue(
                "warning",
                "source_sha256",
                source_expected,
                source_actual,
                "Source hashes differ or are unavailable.",
            )
        )
    lock_expected = compatibility.get("dependency_lock_sha256")
    lock_actual = current.get("dependency_lock_sha256")
    if not lock_expected or not lock_actual:
        issues.append(
            CompatibilityIssue(
                "warning",
                "dependency_lock_sha256",
                lock_expected,
                lock_actual,
                "Dependency lock hash is unavailable.",
            )
        )
    elif lock_expected != lock_actual:
        issues.append(
            CompatibilityIssue(
                "warning",
                "dependency_lock_sha256",
                lock_expected,
                lock_actual,
                "Dependency lock hashes differ.",
            )
        )
    return issues
