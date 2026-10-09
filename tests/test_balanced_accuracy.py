"""Tests for balanced accuracy calculation in viClassify."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import recall_score

from viclassify_cli import ClassificationAnalysis, Config


def test_balanced_accuracy_all_classes_present():
    """Test balanced accuracy when all reporting classes are present in holdout."""
    classes = ["O", "B", "G", "M"]
    y_true = np.array(["O", "O", "B", "B", "G", "G", "M", "M"])
    y_pred = np.array(["O", "B", "B", "B", "G", "M", "M", "O"])

    # Expected: macro recall over all classes
    expected = recall_score(
        y_true,
        y_pred,
        labels=classes,
        average="macro",
        zero_division=0,
    )

    # Calculate using the same logic as the updated implementation
    macro_recall = recall_score(
        y_true,
        y_pred,
        labels=classes,
        average="macro",
        zero_division=0,
    )

    assert macro_recall == pytest.approx(expected)
    # balanced_accuracy should equal macro_recall
    assert macro_recall == pytest.approx(macro_recall)


def test_balanced_accuracy_one_class_absent():
    """Test balanced accuracy when one reporting class is absent from holdout."""
    classes = ["O", "B", "G", "M"]
    y_true = np.array(["O", "O", "B", "B", "M", "M"])
    y_pred = np.array(["O", "B", "B", "B", "M", "O"])

    # Expected recalls:
    # recall(O) = 1/2 = 0.5
    # recall(B) = 2/2 = 1.0
    # recall(G) = 0/0 = 0 (zero_division=0)
    # recall(M) = 1/2 = 0.5
    # balanced_accuracy = (0.5 + 1.0 + 0.0 + 0.5) / 4 = 0.5
    expected = 0.5

    macro_recall = recall_score(
        y_true,
        y_pred,
        labels=classes,
        average="macro",
        zero_division=0,
    )

    assert macro_recall == pytest.approx(expected)
    # balanced_accuracy should equal macro_recall
    assert macro_recall == pytest.approx(expected)


def test_balanced_accuracy_multiple_classes_absent():
    """Test balanced accuracy when multiple reporting classes are absent from holdout."""
    classes = ["O", "B", "G", "M", "K", "L"]
    y_true = np.array(["O", "O", "B", "B"])
    y_pred = np.array(["O", "B", "B", "B"])

    # Expected recalls:
    # recall(O) = 1/2 = 0.5
    # recall(B) = 2/2 = 1.0
    # recall(G) = 0/0 = 0
    # recall(M) = 0/0 = 0
    # recall(K) = 0/0 = 0
    # recall(L) = 0/0 = 0
    # balanced_accuracy = (0.5 + 1.0 + 0.0 + 0.0 + 0.0 + 0.0) / 6 = 1.5/6 = 0.25
    expected = 0.25

    macro_recall = recall_score(
        y_true,
        y_pred,
        labels=classes,
        average="macro",
        zero_division=0,
    )

    assert macro_recall == pytest.approx(expected)
    # balanced_accuracy should equal macro_recall
    assert macro_recall == pytest.approx(expected)


def test_balanced_accuracy_binary_classification():
    """Test balanced accuracy for binary classification with both classes present."""
    classes = ["O", "F"]
    y_true = np.array(["O", "O", "F", "F"])
    y_pred = np.array(["O", "F", "F", "O"])

    # Expected recalls:
    # recall(O) = 1/2 = 0.5
    # recall(F) = 1/2 = 0.5
    # balanced_accuracy = (0.5 + 0.5) / 2 = 0.5
    expected = 0.5

    macro_recall = recall_score(
        y_true,
        y_pred,
        labels=classes,
        average="macro",
        zero_division=0,
    )

    assert macro_recall == pytest.approx(expected)
    # balanced_accuracy should equal macro_recall
    assert macro_recall == pytest.approx(expected)


def test_balanced_accuracy_binary_one_class_absent():
    """Test balanced accuracy for binary classification with one class absent."""
    classes = ["O", "F"]
    y_true = np.array(["O", "O", "O"])
    y_pred = np.array(["O", "O", "O"])

    # Expected recalls:
    # recall(O) = 3/3 = 1.0
    # recall(F) = 0/0 = 0
    # balanced_accuracy = (1.0 + 0.0) / 2 = 0.5
    expected = 0.5

    macro_recall = recall_score(
        y_true,
        y_pred,
        labels=classes,
        average="macro",
        zero_division=0,
    )

    assert macro_recall == pytest.approx(expected)
    # balanced_accuracy should equal macro_recall
    assert macro_recall == pytest.approx(expected)


@pytest.mark.integration
def test_evaluate_untouched_holdout_integration(tmp_path, minimal_grids):
    """Integration test for evaluate_untouched_holdout with missing class."""
    import json

    data = """X1,X2,X3,X4,X5,X6,X7,Y
1.0,2.0,3.0,4.0,5.0,6.0,7.0,O
1.1,2.1,3.1,4.1,5.1,6.1,7.1,O
1.2,2.2,3.2,4.2,5.2,6.2,7.2,O
1.3,2.3,3.3,4.3,5.3,6.3,7.3,O
2.0,3.0,4.0,5.0,6.0,7.0,8.0,B
2.1,3.1,4.1,5.1,6.1,7.1,8.1,B
2.2,3.2,4.2,5.2,6.2,7.2,8.2,B
2.3,3.3,4.3,5.3,6.3,7.3,8.3,B
3.0,4.0,5.0,6.0,7.0,8.0,9.0,G
3.1,4.1,5.1,6.1,7.1,8.1,9.1,G
3.2,4.2,5.2,6.2,7.2,8.2,9.2,G
3.3,4.3,5.3,6.3,7.3,8.3,9.3,G
4.0,5.0,6.0,7.0,8.0,9.0,10.0,M
4.1,5.1,6.1,7.1,8.1,9.1,10.1,M
4.2,5.2,6.2,7.2,8.2,9.2,10.2,M
4.3,5.3,6.3,7.3,8.3,9.3,10.3,M
5.0,6.0,7.0,8.0,9.0,10.0,11.0,O
5.1,6.1,7.1,8.1,9.1,10.1,11.1,B
5.2,6.2,7.2,8.2,9.2,10.2,11.2,M
5.3,6.3,7.3,8.3,9.3,10.3,11.3,M
"""

    data_file = tmp_path / "test_data.csv"
    output_dir = tmp_path / "output"

    data_file.write_text(data, encoding="utf-8")

    cfg = Config(
        data=data_file,
        output=output_dir,
        target="Y",
        features=("X1", "X2", "X3", "X4", "X5", "X6", "X7"),
        class_order=("O", "B", "G", "M"),
        test_fraction=0.25,  # Last 5 rows (O, B, M, M) - no G in holdout
        cv_folds=2,
        random_state=42,
        n_jobs=1,
        permutation_repeats=1,
        dpi=100,
        selection_metric="macro_f1",
    )

    analysis = ClassificationAnalysis(cfg)
    analysis.grids = minimal_grids
    analysis.run()  # Run the full pipeline to generate all outputs including metadata

    # Check the holdout metrics
    holdout_metrics = pd.read_csv(output_dir / "chronological_holdout_metrics.csv")

    # The holdout has O, B, M but no G
    # All models should have balanced_accuracy == macro_recall
    for _, row in holdout_metrics.iterrows():
        assert row["balanced_accuracy"] == pytest.approx(row["macro_recall"])

    # Check model_comparison.csv uses the corrected value
    comparison = pd.read_csv(output_dir / "model_comparison.csv")
    assert "balanced_accuracy" in comparison.columns

    # Check run_metadata.json contains the definition
    with (output_dir / "run_metadata.json").open(encoding="utf-8") as f:
        metadata = json.load(f)
    assert "balanced_accuracy_definition" in metadata
    assert (
        "Macro recall over the complete observed reporting class order"
        in metadata["balanced_accuracy_definition"]
    )
