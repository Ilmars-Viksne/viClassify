"""Tests for hyperparameter tuning isolation."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from viclassify_cli import ClassificationAnalysis, Config


@pytest.mark.integration
class TestGridSearchCVReceivesTrainingOnly:
    """Tests for GridSearchCV receiving training rows only."""

    def test_gridsearchcv_receives_training_only(
        self, binary_chronological_dataset, tmp_path, minimal_grids
    ):
        """Test 8: GridSearchCV receives training rows only."""
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

        # Spy on GridSearchCV.fit
        original_fit = None
        captured_fits = []

        def spy_fit(self, X, y, **kwargs):
            captured_fits.append(
                {
                    "X": X.copy(),
                    "y": y.copy(),
                    "model_name": getattr(self, "_spy_model_name", "unknown"),
                }
            )
            return original_fit(self, X, y, **kwargs)

        from sklearn.model_selection import GridSearchCV

        original_fit = GridSearchCV.fit

        with patch.object(GridSearchCV, "fit", spy_fit):
            analysis.tune_on_training_only()

        # Verify each model's fit received only training data
        train_size = len(analysis.train_indices)
        test_start = analysis.test_indices[0]

        for fit_info in captured_fits:
            X_fit = fit_info["X"]
            y_fit = fit_info["y"]

            # Number of fitted rows equals training size
            assert len(X_fit) == train_size
            assert len(y_fit) == train_size

            # No holdout identity values present
            if "X1" in X_fit.columns:
                max_x1 = X_fit["X1"].max()
                assert max_x1 < test_start, (
                    f"Holdout row found in training for {fit_info['model_name']}"
                )

            # Target values equal y_train
            _, _, y_train, _ = analysis.split_data()
            pd.testing.assert_series_equal(
                y_fit.reset_index(drop=True), y_train.reset_index(drop=True)
            )


class TestPreprocessingFittedWithinCV:
    """Tests for preprocessing fitted within CV."""

    def test_pipeline_structure(
        self, binary_chronological_dataset, tmp_path, minimal_grids
    ):
        """Test 9: preprocessing is fitted within CV - pipeline structure."""
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

        # Verify each model pipeline has preprocessor and classifier
        for name, pipeline in analysis.models.items():
            assert hasattr(pipeline, "named_steps")
            assert "preprocessor" in pipeline.named_steps
            assert "classifier" in pipeline.named_steps

            preprocessor = pipeline.named_steps["preprocessor"]

            # Access unfitted transformers via .transformers attribute or named_transformers_ if fitted
            transformers = getattr(preprocessor, "named_transformers_", None)
            if transformers is None:
                transformers = {
                    name: trans for name, trans, _ in preprocessor.transformers
                }

            if name == "Logistic Regression":
                # Should have imputer and scaler
                assert "numeric" in transformers
                numeric_pipe = transformers["numeric"]
                assert "imputer" in numeric_pipe.named_steps
                assert "scaler" in numeric_pipe.named_steps
            else:
                # Tree models should have imputer
                assert "numeric" in transformers
                numeric_pipe = transformers["numeric"]
                assert "imputer" in numeric_pipe.named_steps


@pytest.mark.integration
class TestSelectedConfigCVTrainingOnly:
    """Tests for selected-configuration CV remaining training-only."""

    def test_selected_config_cv_training_only(
        self, binary_chronological_dataset, tmp_path, minimal_grids
    ):
        """Test 10: selected-configuration CV remains training-only."""
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

        # Spy on cross_validate
        captured_cv = []
        original_cross_validate = None

        def spy_cross_validate(estimator, X, y, **kwargs):
            captured_cv.append({"X": X.copy(), "y": y.copy(), "estimator": estimator})
            return original_cross_validate(estimator, X, y, **kwargs)

        from viclassify_cli import cross_validate

        original_cross_validate = cross_validate

        with patch("viclassify_cli.cross_validate", spy_cross_validate):
            analysis.tune_on_training_only()

        # Verify cross_validate received only training data
        train_size = len(analysis.train_indices)
        test_start = analysis.test_indices[0]

        for cv_info in captured_cv:
            X_cv = cv_info["X"]
            y_cv = cv_info["y"]

            assert len(X_cv) == train_size
            assert len(y_cv) == train_size

            if "X1" in X_cv.columns:
                max_x1 = X_cv["X1"].max()
                assert max_x1 < test_start, "Holdout row found in selected-config CV"

            # Verify estimator has selected hyperparameters
            estimator = cv_info["estimator"]
            assert hasattr(estimator, "named_steps")
            assert "classifier" in estimator.named_steps


@pytest.mark.integration
class TestCVSplitterFromTrainingTargets:
    """Tests for CV splitter created from training targets."""

    def test_cv_splitter_uses_training_class_counts(self, tmp_path, minimal_grids):
        """Test 11: CV splitter is created from training targets."""
        # Create dataset where complete data supports more folds than training
        # 30 rows, test_fraction=0.5 -> split=15, holdout=15
        # Training: 15 rows with 3 classes (5 each) -> max 5 folds
        # Complete: 30 rows with 3 classes (10 each) -> max 10 folds
        # Request 8 folds - should be reduced to 5 based on training
        rows = []
        for i in range(30):
            x1 = float(i)
            x2 = float(i * 10)
            y = ["O", "B", "G"][i % 3]
            rows.append(f"{x1},{x2},{y}")
        data = "X1,X2,Y\n" + "\n".join(rows)

        data_file = tmp_path / "data.csv"
        data_file.write_text(data)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        cfg = Config(
            data=data_file,
            output=output_dir,
            target="Y",
            features=("X1", "X2"),
            class_order=("O", "B", "G"),
            test_fraction=0.5,
            cv_folds=8,  # Request more than training can support
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

        # Capture warning about fold reduction
        with pytest.warns(UserWarning, match="Reducing CV folds from 8 to 5"):
            analysis.tune_on_training_only()

        # Verify effective folds is 5 (based on training class count of 5)
        # This is implicitly tested by the warning above


@pytest.mark.integration
class TestEvaluationModelFitUsesTrainingOnly:
    """Tests for evaluation model fit using training rows only."""

    def test_evaluation_model_fit_training_only(
        self, binary_chronological_dataset, tmp_path, minimal_grids, monkeypatch
    ):
        """Test 12: evaluation model fit uses training rows only."""
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

        # Spy on pipeline.fit during evaluation
        captured_fits = []
        from sklearn.pipeline import Pipeline

        original_fit = Pipeline.fit

        def spy_fit(self, X, y=None, **kwargs):
            captured_fits.append(
                {
                    "X": X.copy(),
                    "y": y.copy() if y is not None else None,
                    "model_name": getattr(self, "_spy_model_name", "unknown"),
                }
            )
            return original_fit(self, X, y, **kwargs)

        monkeypatch.setattr(Pipeline, "fit", spy_fit)
        analysis.evaluate_untouched_holdout()

        # Verify each evaluation model fit used only training data
        train_size = len(analysis.train_indices)
        len(analysis.test_indices)
        test_start = analysis.test_indices[0]

        for fit_info in captured_fits:
            X_fit = fit_info["X"]
            y_fit = fit_info["y"]

            assert len(X_fit) == train_size
            assert len(y_fit) == train_size

            if "X1" in X_fit.columns:
                max_x1 = X_fit["X1"].max()
                assert max_x1 < test_start, (
                    f"Holdout row in evaluation fit for {fit_info['model_name']}"
                )

        # Verify prediction uses exactly X_test
        # This is implicitly tested by the holdout prediction exports


@pytest.mark.integration
class TestEvaluationArtifactsTrainingOnly:
    """Tests for evaluation artifacts being training-only."""

    def test_evaluation_artifacts_training_only(
        self, binary_chronological_dataset, tmp_path, minimal_grids
    ):
        """Test 13: evaluation artifacts are training-only artifacts."""
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

        # Load each evaluation artifact
        import joblib

        for name in analysis.models:
            slug = analysis.slug(name)
            artifact_path = output_dir / f"evaluation_{slug}.joblib"
            assert artifact_path.exists()

            artifact = joblib.load(artifact_path)

            # Verify artifact contents
            assert artifact["model_role"] == "chronological_training_evaluation_model"
            assert artifact["features"] == list(cfg.features)
            assert artifact["reporting_class_order"] == analysis.classes
            assert "estimator_class_order" in artifact
            assert artifact["best_parameters"] == analysis.best_params[name]

            # Verify predictions match exported holdout predictions
            model = artifact["model"]
            X_test = analysis.data.iloc[analysis.test_indices][list(cfg.features)]
            preds = model.predict(X_test)

            pred_file = output_dir / f"holdout_predictions_{slug}.csv"
            exported_preds = pd.read_csv(pred_file)["Predicted_Y"].values

            np.testing.assert_array_equal(preds, exported_preds)


@pytest.mark.integration
class TestHoldoutPredictionExports:
    """Tests for holdout prediction exports containing only holdout rows."""

    def test_holdout_predictions_only_holdout_rows(
        self, binary_chronological_dataset, tmp_path, minimal_grids
    ):
        """Test 14: holdout prediction exports contain only holdout rows."""
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

        test_indices = analysis.test_indices
        test_sequences = analysis.data.iloc[test_indices]["Sequence"].values

        for name in analysis.models:
            slug = analysis.slug(name)
            pred_file = output_dir / f"holdout_predictions_{slug}.csv"
            preds = pd.read_csv(pred_file)

            # Number of rows equals holdout size
            assert len(preds) == len(test_indices)

            # Every exported Sequence belongs to holdout
            assert set(preds["Sequence"].values) == set(test_sequences)

            # No training sequence present
            train_sequences = analysis.data.iloc[analysis.train_indices][
                "Sequence"
            ].values
            assert not set(preds["Sequence"].values).intersection(set(train_sequences))

            # Sequence order matches source holdout order
            assert np.array_equal(preds["Sequence"].values, test_sequences)

            # True target values match y_test
            _, _, _, y_test = analysis.split_data()
            np.testing.assert_array_equal(preds["Y"].values, y_test.values)

            # Correct is equivalent to actual == Predicted_Y
            expected_correct = (preds["Y"] == preds["Predicted_Y"]).values
            np.testing.assert_array_equal(preds["Correct"].values, expected_correct)

            # Probability columns align with estimator class order
            # (verified by checking column names exist)
            proba_cols = [c for c in preds.columns if c.startswith("Probability_")]
            assert len(proba_cols) == 2  # binary

            # Each row's probabilities sum to 1
            proba_sums = preds[proba_cols].sum(axis=1)
            np.testing.assert_allclose(proba_sums, 1.0, rtol=1e-6)

            # Prediction_Confidence equals maximum probability
            max_proba = preds[proba_cols].max(axis=1)
            np.testing.assert_allclose(
                preds["Prediction_Confidence"].values, max_proba, rtol=1e-6
            )

            # Prediction_Margin equals difference between two largest
            sorted_proba = np.sort(preds[proba_cols].values, axis=1)
            expected_margin = sorted_proba[:, -1] - sorted_proba[:, -2]
            np.testing.assert_allclose(
                preds["Prediction_Margin"].values, expected_margin, rtol=1e-6
            )


@pytest.mark.integration
class TestMisclassificationExports:
    """Tests for misclassification exports being exact subset."""

    def test_misclassification_exports_exact_subset(
        self, binary_chronological_dataset, tmp_path, minimal_grids
    ):
        """Test 15: misclassification exports are an exact subset."""
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
            pred_file = output_dir / f"holdout_predictions_{slug}.csv"
            misc_file = output_dir / f"holdout_misclassifications_{slug}.csv"

            preds = pd.read_csv(pred_file)
            miscs = pd.read_csv(misc_file)

            # Misclassifications should be exact subset of incorrect predictions
            incorrect_preds = preds[~preds["Correct"]]

            # Same number of rows
            assert len(miscs) == len(incorrect_preds)

            # Same sequences
            assert set(miscs["Sequence"].values) == set(
                incorrect_preds["Sequence"].values
            )

            # Same columns (misclassifications should have all columns)
            assert list(miscs.columns) == list(preds.columns)

            # All misclassified rows have Correct=False
            assert all(miscs["Correct"] == False)
