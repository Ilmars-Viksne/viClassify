"""Tests for evaluation export consistency."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from viclassify_cli import ClassificationAnalysis, Config


@pytest.mark.integration
class TestCombinedHoldoutMetrics:
    """Tests for combined holdout metrics consistency."""

    def test_combined_holdout_metrics_consistency(
        self, binary_chronological_dataset, tmp_path, minimal_grids
    ):
        """Test 29: combined holdout metrics match independently recomputed values."""
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
        analysis.grids = minimal_grids
        analysis.load()
        analysis.create_chronological_split()
        analysis.build_models()
        analysis.tune_on_training_only()
        analysis.evaluate_untouched_holdout()

        # Read combined metrics
        combined = pd.read_csv(output_dir / "chronological_holdout_metrics.csv")

        for _, row in combined.iterrows():
            model_name = row["model"]
            slug = analysis.slug(model_name)

            # Read predictions
            pred_file = output_dir / f"holdout_predictions_{slug}.csv"
            preds = pd.read_csv(pred_file)

            y_true = preds["Y"].values
            y_pred = preds["Predicted_Y"].values
            classes = analysis.classes

            # Get probability columns
            proba_cols = [c for c in preds.columns if c.startswith("Probability_")]
            y_proba = preds[proba_cols].values

            # Recompute metrics
            expected = {
                "accuracy": accuracy_score(y_true, y_pred),
                "balanced_accuracy": recall_score(
                    y_true, y_pred, labels=classes, average="macro", zero_division=0
                ),
                "macro_precision": precision_score(
                    y_true, y_pred, labels=classes, average="macro", zero_division=0
                ),
                "macro_recall": recall_score(
                    y_true, y_pred, labels=classes, average="macro", zero_division=0
                ),
                "macro_f1": f1_score(
                    y_true, y_pred, labels=classes, average="macro", zero_division=0
                ),
                "weighted_f1": f1_score(
                    y_true, y_pred, labels=classes, average="weighted", zero_division=0
                ),
                "mcc": matthews_corrcoef(y_true, y_pred),
            }

            try:
                expected["log_loss"] = log_loss(y_true, y_proba, labels=classes)
            except ValueError:
                expected["log_loss"] = np.nan

            # Compare
            for metric, expected_val in expected.items():
                actual_val = row[metric]
                if np.isnan(expected_val):
                    assert np.isnan(actual_val), (
                        f"{model_name}: {metric} expected NaN but got {actual_val}"
                    )
                else:
                    assert actual_val == pytest.approx(expected_val, rel=1e-6), (
                        f"{model_name}: {metric} mismatch: expected {expected_val}, got {actual_val}"
                    )


@pytest.mark.integration
class TestPerClassClassificationReport:
    """Tests for per-class classification report consistency."""

    def test_per_class_report_consistency(
        self, binary_chronological_dataset, tmp_path, minimal_grids
    ):
        """Test 30: per-class classification report matches independent calculations."""
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
        analysis.grids = minimal_grids
        analysis.load()
        analysis.create_chronological_split()
        analysis.build_models()
        analysis.tune_on_training_only()
        analysis.evaluate_untouched_holdout()

        for name in analysis.models:
            slug = analysis.slug(name)
            report_file = output_dir / f"holdout_report_{slug}.csv"
            report = pd.read_csv(report_file, index_col=0)

            # Read predictions
            pred_file = output_dir / f"holdout_predictions_{slug}.csv"
            preds = pd.read_csv(pred_file)

            y_true = preds["Y"].values
            y_pred = preds["Predicted_Y"].values
            classes = analysis.classes

            # Check per-class metrics
            for cls in classes:
                if cls in report.index:
                    row = report.loc[cls]
                    # Support
                    expected_support = np.sum(y_true == cls)
                    assert row["support"] == expected_support

                    # Precision, recall, F1
                    from sklearn.metrics import precision_recall_fscore_support

                    prec, rec, f1, _supp = precision_recall_fscore_support(
                        y_true, y_pred, labels=[cls], average=None, zero_division=0
                    )
                    assert row["precision"] == pytest.approx(prec[0])
                    assert row["recall"] == pytest.approx(rec[0])
                    assert row["f1-score"] == pytest.approx(f1[0])

            # Check macro avg
            if "macro avg" in report.index:
                macro_row = report.loc["macro avg"]
                assert macro_row["precision"] == pytest.approx(
                    precision_score(
                        y_true, y_pred, labels=classes, average="macro", zero_division=0
                    )
                )
                assert macro_row["recall"] == pytest.approx(
                    recall_score(
                        y_true, y_pred, labels=classes, average="macro", zero_division=0
                    )
                )
                assert macro_row["f1-score"] == pytest.approx(
                    f1_score(
                        y_true, y_pred, labels=classes, average="macro", zero_division=0
                    )
                )

            # Check weighted avg
            if "weighted avg" in report.index:
                weighted_row = report.loc["weighted avg"]
                assert weighted_row["precision"] == pytest.approx(
                    precision_score(
                        y_true,
                        y_pred,
                        labels=classes,
                        average="weighted",
                        zero_division=0,
                    )
                )
                assert weighted_row["recall"] == pytest.approx(
                    recall_score(
                        y_true,
                        y_pred,
                        labels=classes,
                        average="weighted",
                        zero_division=0,
                    )
                )
                assert weighted_row["f1-score"] == pytest.approx(
                    f1_score(
                        y_true,
                        y_pred,
                        labels=classes,
                        average="weighted",
                        zero_division=0,
                    )
                )

            # Ensure absent reporting classes remain represented
            # (if a class is absent from holdout, it should still appear in report with support=0)
            for cls in classes:
                assert cls in report.index or cls in [
                    "accuracy",
                    "macro avg",
                    "weighted avg",
                ]


@pytest.mark.integration
class TestConfusionMatrixConsistency:
    """Tests for confusion matrix consistency."""

    def test_confusion_matrix_consistency(
        self, binary_chronological_dataset, tmp_path, minimal_grids
    ):
        """Test 31: confusion matrix consistency."""
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
        analysis.grids = minimal_grids
        analysis.load()
        analysis.create_chronological_split()
        analysis.build_models()
        analysis.tune_on_training_only()
        analysis.evaluate_untouched_holdout()

        for name in analysis.models:
            slug = analysis.slug(name)

            # Read predictions
            pred_file = output_dir / f"holdout_predictions_{slug}.csv"
            preds = pd.read_csv(pred_file)

            y_true = preds["Y"].values
            y_pred = preds["Predicted_Y"].values
            classes = analysis.classes

            # Check counts matrix
            counts_file = output_dir / f"holdout_confusion_matrix_{slug}.csv"
            counts = pd.read_csv(counts_file, index_col=0)

            expected_counts = confusion_matrix(y_true, y_pred, labels=classes)
            assert counts.index.tolist() == [f"Actual_{c}" for c in classes]
            assert counts.columns.tolist() == [f"Predicted_{c}" for c in classes]
            np.testing.assert_array_equal(counts.values, expected_counts)

            # Check normalized matrix
            norm_file = output_dir / f"holdout_confusion_matrix_normalized_{slug}.csv"
            normalized = pd.read_csv(norm_file, index_col=0)

            expected_norm = confusion_matrix(
                y_true, y_pred, labels=classes, normalize="true"
            )
            assert normalized.index.tolist() == [f"Actual_{c}" for c in classes]
            assert normalized.columns.tolist() == [f"Predicted_{c}" for c in classes]
            np.testing.assert_allclose(normalized.values, expected_norm, rtol=1e-6)

            # Handle absent true classes - rows should still exist with NaN or 0
            for i, cls in enumerate(classes):
                if cls not in y_true:
                    # Row should exist but be NaN (normalize="true" gives NaN for absent classes)
                    assert np.all(np.isnan(expected_norm[i, :])) or np.all(
                        expected_norm[i, :] == 0
                    )


@pytest.mark.integration
class TestModelComparisonConsistency:
    """Tests for model comparison consistency."""

    def test_model_comparison_consistency(
        self, binary_chronological_dataset, tmp_path, minimal_grids
    ):
        """Test 32: model comparison consistency."""
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
        analysis.grids = minimal_grids
        analysis.load()
        analysis.create_chronological_split()
        analysis.build_models()
        analysis.tune_on_training_only()
        analysis.evaluate_untouched_holdout()
        analysis.refit_and_export_production_models()

        # Read model comparison
        comparison = pd.read_csv(output_dir / "model_comparison.csv")

        # Read CV summaries
        cv_summary = pd.DataFrame(analysis.search_summaries)

        # Read holdout metrics
        holdout = pd.read_csv(output_dir / "chronological_holdout_metrics.csv")

        for _, row in comparison.iterrows():
            model = row["model"]

            # Find corresponding CV row
            cv_row = cv_summary[cv_summary["model"] == model].iloc[0]
            holdout_row = holdout[holdout["model"] == model].iloc[0]

            # Check gap calculation
            selection_metric = cfg.selection_metric
            cv_col = f"cv_{selection_metric}_mean"
            holdout_col = selection_metric
            gap_col = f"cv_holdout_{selection_metric}_gap"

            expected_gap = cv_row[cv_col] - holdout_row[holdout_col]
            actual_gap = row[gap_col]

            assert actual_gap == pytest.approx(expected_gap, rel=1e-6), (
                f"{model}: gap mismatch: expected {expected_gap}, got {actual_gap}"
            )

            # When selection_metric is balanced_accuracy, verify holdout uses corrected value
            if selection_metric == "balanced_accuracy":
                assert holdout_row[holdout_col] == pytest.approx(
                    holdout_row["macro_recall"]
                ), "Holdout balanced_accuracy should equal macro_recall"
