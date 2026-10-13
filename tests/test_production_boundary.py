"""Tests for production refit boundary."""

from unittest.mock import patch

import pytest
from sklearn.pipeline import Pipeline

from viclassify_cli import ClassificationAnalysis, Config


@pytest.mark.integration
class TestProductionModelsUseAllRows:
    """Tests for production models using all rows."""

    def test_production_models_use_all_rows(
        self, binary_chronological_dataset, tmp_path, minimal_grids, monkeypatch
    ):
        """Test 16: production models use all rows."""
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

        analysis = ClassificationAnalysis(cfg, grid_overrides=minimal_grids)

        # Spy on production model fit
        captured_fits = []
        original_fit = Pipeline.fit

        def spy_fit(self, X, y=None, **fit_params):
            classifier_step = self.named_steps.get("classifier")
            classifier_name = (
                classifier_step.__class__.__name__
                if classifier_step is not None
                else "unknown"
            )
            if len(X) == 20:
                captured_fits.append(
                    {
                        "X": X.copy(),
                        "y": y.copy() if y is not None else None,
                        "classifier_name": classifier_name,
                    }
                )
            return original_fit(self, X, y, **fit_params)

        monkeypatch.setattr(Pipeline, "fit", spy_fit)
        analysis.run()

        total_rows = len(analysis.data)
        # Should capture exactly 4 top-level production fits
        assert len(captured_fits) == 4

        expected_classifiers = {
            "LogisticRegression",
            "RandomForestClassifier",
            "ExtraTreesClassifier",
            "HistGradientBoostingClassifier",
        }
        captured_classifiers = {
            fit_info["classifier_name"] for fit_info in captured_fits
        }
        assert captured_classifiers == expected_classifiers

        for fit_info in captured_fits:
            X_fit = fit_info["X"]
            y_fit = fit_info["y"]

            # Total fitted row count equals complete dataset size
            assert len(X_fit) == total_rows
            assert len(y_fit) == total_rows

            # Training rows included
            train_x1 = analysis.data.iloc[analysis.train_indices]["X1"].values
            assert set(train_x1).issubset(set(X_fit["X1"].values))

            # Former holdout rows included
            test_x1 = analysis.data.iloc[analysis.test_indices]["X1"].values
            assert set(test_x1).issubset(set(X_fit["X1"].values))


@pytest.mark.integration
class TestProductionFittingAfterEvaluation:
    """Tests for production fitting occurring after evaluation."""

    def test_production_fitting_after_evaluation(
        self, binary_chronological_dataset, tmp_path, minimal_grids
    ):
        """Test 17: production fitting occurs after evaluation."""
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

        # Create fresh instance to run run() cleanly
        fresh_analysis = ClassificationAnalysis(cfg, grid_overrides=minimal_grids)

        stage_order = []

        original_evaluate = fresh_analysis.evaluate_untouched_holdout
        original_refit = fresh_analysis.refit_and_export_production_models

        def track_evaluate():
            stage_order.append("evaluate_start")
            result = original_evaluate()
            stage_order.append("evaluate_end")
            return result

        def track_refit():
            stage_order.append("refit_start")
            result = original_refit()
            stage_order.append("refit_end")
            return result

        with (
            patch.object(fresh_analysis, "evaluate_untouched_holdout", track_evaluate),
            patch.object(
                fresh_analysis, "refit_and_export_production_models", track_refit
            ),
        ):
            fresh_analysis.run()

        # Verify order: evaluate completes before refit begins
        eval_end_idx = stage_order.index("evaluate_end")
        refit_start_idx = stage_order.index("refit_start")
        assert eval_end_idx < refit_start_idx, (
            "Evaluation must complete before production refit begins"
        )


@pytest.mark.integration
class TestArtifactRolesDistinct:
    """Tests for artifact roles remaining distinct."""

    def test_artifact_roles_distinct(
        self, binary_chronological_dataset, tmp_path, minimal_grids
    ):
        """Test 18: artifact roles remain distinct."""
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

        analysis = ClassificationAnalysis(cfg, grid_overrides=minimal_grids)
        analysis.run()

        import joblib

        out_dir = analysis.final_output_path
        # Check evaluation artifacts
        for name in analysis.models:
            slug = analysis.slug(name)
            eval_path = out_dir / f"evaluation_{slug}.joblib"
            assert eval_path.exists()

            eval_artifact = joblib.load(eval_path)
            assert (
                eval_artifact["model_role"] == "chronological_training_evaluation_model"
            )

        # Check production artifacts
        for name in analysis.models:
            slug = analysis.slug(name)
            prod_path = out_dir / f"production_{slug}.joblib"
            assert prod_path.exists()

            prod_artifact = joblib.load(prod_path)
            assert prod_artifact["model_role"] == "production_full_data_refit"

        # Check CV-selected production artifact
        cv_selected_path = out_dir / "cv_selected_production_model.joblib"
        assert cv_selected_path.exists()

        cv_selected_artifact = joblib.load(cv_selected_path)
        assert (
            cv_selected_artifact["model_role"]
            == "cv_selected_production_full_data_refit"
        )

        # Verify filenames don't overwrite
        eval_files = list(out_dir.glob("evaluation_*.joblib"))
        prod_files = list(out_dir.glob("production_*.joblib"))
        assert len(eval_files) == 4
        assert len(prod_files) == 4
        assert len(set(eval_files).intersection(set(prod_files))) == 0
