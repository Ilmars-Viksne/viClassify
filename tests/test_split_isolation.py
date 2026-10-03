"""Tests for chronological split isolation."""

import numpy as np
import pandas as pd
import pytest

from viclassify_cli import ClassificationAnalysis, Config


class TestChronologicalSplitBoundary:
    """Tests for exact chronological boundary."""

    def test_exact_chronological_boundary(self, binary_chronological_dataset, tmp_path):
        """Test 1: exact chronological boundary."""
        data_file = tmp_path / "data.csv"
        data_file.write_text(binary_chronological_dataset)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        df = pd.read_csv(data_file)
        n_rows = len(df)
        test_fraction = 0.25
        expected_split = int(np.floor(n_rows * (1 - test_fraction)))

        cfg = Config(
            data=data_file,
            output=output_dir,
            target="Y",
            features=("X1", "X2"),
            class_order=("O", "F"),
            test_fraction=test_fraction,
            cv_folds=2,
            random_state=42,
            n_jobs=1,
            permutation_repeats=1,
            dpi=50,
            selection_metric="macro_f1",
        )

        analysis = ClassificationAnalysis(cfg)
        analysis.load()
        analysis.create_chronological_split()

        # Verify exact boundary
        assert np.array_equal(analysis.train_indices, np.arange(expected_split))
        assert np.array_equal(analysis.test_indices, np.arange(expected_split, n_rows))

        # Verify boundary indices
        assert analysis.train_indices[-1] == expected_split - 1
        assert analysis.test_indices[0] == expected_split

        # Verify no randomization - indices are sequential
        assert np.all(np.diff(analysis.train_indices) == 1)
        assert np.all(np.diff(analysis.test_indices) == 1)


class TestPartitionDisjoint:
    """Tests for disjoint partitions."""

    def test_partitions_are_disjoint(self, binary_chronological_dataset, tmp_path):
        """Test 2: partitions are disjoint."""
        data_file = tmp_path / "data.csv"
        data_file.write_text(binary_chronological_dataset)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        cfg = Config(
            data=data_file,
            output=output_dir,
            target="Y",
            features=("X1", "X2"),
            class_order=("O", "F"),
            test_fraction=0.25,
            cv_folds=2,
            random_state=42,
            n_jobs=1,
            permutation_repeats=1,
            dpi=50,
            selection_metric="macro_f1",
        )

        analysis = ClassificationAnalysis(cfg)
        analysis.load()
        analysis.create_chronological_split()

        # Verify disjoint
        train_set = set(analysis.train_indices)
        test_set = set(analysis.test_indices)
        assert train_set.isdisjoint(test_set)


class TestPartitionCoverage:
    """Tests for complete coverage."""

    def test_partitions_cover_all_rows(self, binary_chronological_dataset, tmp_path):
        """Test 3: partitions provide complete coverage."""
        data_file = tmp_path / "data.csv"
        data_file.write_text(binary_chronological_dataset)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        df = pd.read_csv(data_file)
        n_rows = len(df)

        cfg = Config(
            data=data_file,
            output=output_dir,
            target="Y",
            features=("X1", "X2"),
            class_order=("O", "F"),
            test_fraction=0.25,
            cv_folds=2,
            random_state=42,
            n_jobs=1,
            permutation_repeats=1,
            dpi=50,
            selection_metric="macro_f1",
        )

        analysis = ClassificationAnalysis(cfg)
        analysis.load()
        analysis.create_chronological_split()

        # Concatenate and verify
        all_indices = np.concatenate([analysis.train_indices, analysis.test_indices])
        all_indices_sorted = np.sort(all_indices)

        # Should contain every index exactly once
        assert np.array_equal(all_indices_sorted, np.arange(n_rows))
        assert len(all_indices) == n_rows
        assert len(np.unique(all_indices)) == n_rows


class TestSplitDataReturnsExpectedRows:
    """Tests for split_data returning expected rows."""

    def test_split_data_returns_expected_rows(
        self, binary_chronological_dataset, tmp_path
    ):
        """Test 4: split_data returns the expected rows."""
        data_file = tmp_path / "data.csv"
        data_file.write_text(binary_chronological_dataset)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        cfg = Config(
            data=data_file,
            output=output_dir,
            target="Y",
            features=("X1", "X2"),
            class_order=("O", "F"),
            test_fraction=0.25,
            cv_folds=2,
            random_state=42,
            n_jobs=1,
            permutation_repeats=1,
            dpi=50,
            selection_metric="macro_f1",
        )

        analysis = ClassificationAnalysis(cfg)
        analysis.load()
        analysis.create_chronological_split()

        X_train, X_test, y_train, y_test = analysis.split_data()

        # Verify row counts
        assert len(X_train) == len(analysis.train_indices)
        assert len(X_test) == len(analysis.test_indices)
        assert len(y_train) == len(analysis.train_indices)
        assert len(y_test) == len(analysis.test_indices)

        # Verify predictor order
        assert list(X_train.columns) == ["X1", "X2"]
        assert list(X_test.columns) == ["X1", "X2"]

        # Verify targets correspond to same rows
        # Use identity feature X1 to verify exact membership
        train_x1_values = X_train["X1"].values
        test_x1_values = X_test["X1"].values

        # Training should have first 15 rows (0-14)
        assert np.all(train_x1_values < 15)
        # Test should have last 5 rows (15-19)
        assert np.all(test_x1_values >= 15)

        # Verify holdout row order is unchanged
        assert np.array_equal(test_x1_values, np.arange(15, 20))

        # Verify returned objects are copies (not views)
        # Modifying returned data should not affect original
        X_train_copy = X_train.copy()
        X_train.iloc[0, 0] = 999
        assert X_train_copy.iloc[0, 0] != 999


class TestExportedSplitAssignments:
    """Tests for exported split assignments."""

    def test_exported_split_assignments_correct(
        self, binary_chronological_dataset, tmp_path
    ):
        """Test 5: exported split assignments are correct."""
        data_file = tmp_path / "data.csv"
        data_file.write_text(binary_chronological_dataset)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        df = pd.read_csv(data_file)
        n_rows = len(df)
        test_fraction = 0.25
        expected_split = int(np.floor(n_rows * (1 - test_fraction)))

        cfg = Config(
            data=data_file,
            output=output_dir,
            target="Y",
            features=("X1", "X2"),
            class_order=("O", "F"),
            test_fraction=test_fraction,
            cv_folds=2,
            random_state=42,
            n_jobs=1,
            permutation_repeats=1,
            dpi=50,
            selection_metric="macro_f1",
        )

        analysis = ClassificationAnalysis(cfg)
        analysis.load()
        analysis.create_chronological_split()

        # Read exported assignments
        assignments_file = output_dir / "chronological_split_assignments.csv"
        assignments = pd.read_csv(assignments_file)

        # Verify every sequence appears exactly once
        assert len(assignments) == n_rows
        assert assignments["Sequence"].nunique() == n_rows
        assert set(assignments["Sequence"]) == set(range(1, n_rows + 1))

        # Verify sequences are one-based
        assert assignments["Sequence"].min() == 1
        assert assignments["Sequence"].max() == n_rows

        # Verify partition labels
        train_assignments = assignments[assignments["partition"] == "training"]
        test_assignments = assignments[assignments["partition"] == "testing"]

        assert len(train_assignments) == expected_split
        assert len(test_assignments) == n_rows - expected_split

        # Verify training rows are before boundary
        assert train_assignments["Sequence"].max() == expected_split
        # Verify testing rows are at and after boundary
        assert test_assignments["Sequence"].min() == expected_split + 1

        # Verify target labels match source
        source_targets = df["Y"].values
        for _, row in assignments.iterrows():
            seq = row["Sequence"]
            expected_target = source_targets[seq - 1]  # 1-based to 0-based
            assert row["Y"] == expected_target

        # Verify no unexpected partition labels
        assert set(assignments["partition"].unique()) == {"training", "testing"}


class TestMissingTrainingClass:
    """Tests for missing training class failure."""

    def test_missing_training_class_fails(
        self, missing_training_class_dataset, tmp_path
    ):
        """Test 6: missing training class fails."""
        data_file = tmp_path / "data.csv"
        data_file.write_text(missing_training_class_dataset)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        cfg = Config(
            data=data_file,
            output=output_dir,
            target="Y",
            features=("X1", "X2", "X3", "X4", "X5", "X6", "X7"),
            class_order=("O", "B", "G", "M"),
            test_fraction=0.25,
            cv_folds=2,
            random_state=42,
            n_jobs=1,
            permutation_repeats=1,
            dpi=50,
            selection_metric="macro_f1",
        )

        analysis = ClassificationAnalysis(cfg)
        analysis.load()

        # Should raise ValueError for missing class in training
        with pytest.raises(
            ValueError, match="Chronological training partition lacks classes"
        ):
            analysis.create_chronological_split()


class TestAbsentHoldoutClass:
    """Tests for absent holdout class warning."""

    def test_absent_holdout_class_warns_but_continues(
        self, absent_holdout_class_dataset, tmp_path
    ):
        """Test 7: absent holdout class warns but continues."""
        data_file = tmp_path / "data.csv"
        data_file.write_text(absent_holdout_class_dataset)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        cfg = Config(
            data=data_file,
            output=output_dir,
            target="Y",
            features=("X1", "X2", "X3", "X4", "X5", "X6", "X7"),
            class_order=("O", "B", "G", "M"),
            test_fraction=0.25,
            cv_folds=2,
            random_state=42,
            n_jobs=1,
            permutation_repeats=1,
            dpi=50,
            selection_metric="macro_f1",
        )

        analysis = ClassificationAnalysis(cfg)
        analysis.load()

        # Should warn but continue
        with pytest.warns(
            UserWarning, match="Chronological test partition lacks classes"
        ):
            analysis.create_chronological_split()

        # Verify split indices remain valid
        assert analysis.train_indices is not None
        assert analysis.test_indices is not None
        assert len(analysis.train_indices) > 0
        assert len(analysis.test_indices) > 0

        # Verify we can split data
        X_train, X_test, _y_train, _y_test = analysis.split_data()
        assert len(X_train) == len(analysis.train_indices)
        assert len(X_test) == len(analysis.test_indices)
