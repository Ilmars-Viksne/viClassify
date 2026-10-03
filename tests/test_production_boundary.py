"""Tests for production refit boundary."""

from unittest.mock import patch

from viclassify_cli import ClassificationAnalysis, Config


class TestProductionModelsUseAllRows:
    """Tests for production models using all rows."""

    def test_production_models_use_all_rows(
        self, binary_chronological_dataset, tmp_path, minimal_grids
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

        analysis = ClassificationAnalysis(cfg)
        analysis.grids = minimal_grids
        analysis.load()
        analysis.create_chronological_split()
        analysis.build_models()
        analysis.tune_on_training_only()
        analysis.evaluate_untouched_holdout()

        # Spy on production model fit
        captured_fits = []

        def spy_fit(self, X, y, **kwargs):
            captured_fits.append(
                {
                    "X": X.copy(),
                    "y": y.copy(),
                    "model_name": getattr(self, "_spy_model_name", "unknown"),
                }
            )
            return self._original_fit(X, y, **kwargs)

        from sklearn.pipeline import Pipeline

        with patch.object(Pipeline, "fit", spy_fit):
            analysis.refit_and_export_production_models()

        # Verify production models fitted on all rows
        total_rows = len(analysis.data)
        len(analysis.train_indices)
        len(analysis.test_indices)

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

        analysis = ClassificationAnalysis(cfg)
        analysis.grids = minimal_grids
        analysis.load()
        analysis.create_chronological_split()
        analysis.build_models()
        analysis.tune_on_training_only()

        # Track stage order
        stage_order = []

        original_evaluate = analysis.evaluate_untouched_holdout
        original_refit = analysis.refit_and_export_production_models

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
            patch.object(analysis, "evaluate_untouched_holdout", track_evaluate),
            patch.object(analysis, "refit_and_export_production_models", track_refit),
        ):
            analysis.run()

        # Verify order: evaluate completes before refit begins
        eval_end_idx = stage_order.index("evaluate_end")
        refit_start_idx = stage_order.index("refit_start")
        assert eval_end_idx < refit_start_idx, (
            "Evaluation must complete before production refit begins"
        )


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

        analysis = ClassificationAnalysis(cfg)
        analysis.grids = minimal_grids
        analysis.load()
        analysis.create_chronological_split()
        analysis.build_models()
        analysis.tune_on_training_only()
        analysis.evaluate_untouched_holdout()
        analysis.refit_and_export_production_models()

        import joblib

        # Check evaluation artifacts
        for name in analysis.models:
            slug = analysis.slug(name)
            eval_path = output_dir / f"evaluation_{slug}.joblib"
            assert eval_path.exists()

            eval_artifact = joblib.load(eval_path)
            assert (
                eval_artifact["model_role"] == "chronological_training_evaluation_model"
            )

        # Check production artifacts
        for name in analysis.models:
            slug = analysis.slug(name)
            prod_path = output_dir / f"production_{slug}.joblib"
            assert prod_path.exists()

            prod_artifact = joblib.load(prod_path)
            assert prod_artifact["model_role"] == "production_full_data_refit"

        # Check best overall production artifact
        best_path = output_dir / "best_overall_model.joblib"
        assert best_path.exists()

        best_artifact = joblib.load(best_path)
        assert best_artifact["model_role"] == "best_overall_production_full_data_refit"

        # Verify filenames don't overwrite
        eval_files = list(output_dir.glob("evaluation_*.joblib"))
        prod_files = list(output_dir.glob("production_*.joblib"))
        assert len(eval_files) == 4
        assert len(prod_files) == 4
        assert len(set(eval_files).intersection(set(prod_files))) == 0
