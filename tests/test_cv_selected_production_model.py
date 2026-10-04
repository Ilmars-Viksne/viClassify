"""Tests for CV-selected production model selection, artifact, metadata, and terminology."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from viclassify_cli import ClassificationAnalysis, Config


def create_analysis(tmp_path, minimal_grids, dataset_text):
    data_file = tmp_path / "data.csv"
    data_file.write_text(dataset_text)
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

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
    return analysis, output_dir


@pytest.mark.integration
class TestCVSelectedProductionModel:
    def test_filename_existence(
        self, binary_chronological_dataset, tmp_path, minimal_grids
    ):
        """Test 1: cv_selected_production_model.joblib exists and best_overall_model.joblib does not."""
        analysis, output_dir = create_analysis(
            tmp_path, minimal_grids, binary_chronological_dataset
        )
        analysis.run()

        assert (output_dir / "cv_selected_production_model.joblib").exists()
        assert not (output_dir / "best_overall_model.joblib").exists()

    def test_artifact_metadata(
        self, binary_chronological_dataset, tmp_path, minimal_grids
    ):
        """Test 2: check exported artifact metadata fields."""
        analysis, output_dir = create_analysis(
            tmp_path, minimal_grids, binary_chronological_dataset
        )
        analysis.run()

        artifact = joblib.load(output_dir / "cv_selected_production_model.joblib")

        assert artifact["model_role"] == "cv_selected_production_full_data_refit"
        assert (
            "Highest mean non-nested tuning-CV selection score"
            in artifact["selection_basis"]
        )
        assert artifact["holdout_used_for_selection"] is False
        assert artifact["production_refit_data"] == "all_available_rows"
        assert artifact["selection_score_mean"] == analysis.cv_selected_score_mean
        assert artifact["selection_score_std"] == analysis.cv_selected_score_std

    def test_selection_uses_cv_not_holdout(
        self, binary_chronological_dataset, tmp_path, minimal_grids
    ):
        """Test 3: CV score drives selection even if holdout score favors another model."""
        analysis, output_dir = create_analysis(
            tmp_path, minimal_grids, binary_chronological_dataset
        )
        analysis.load()
        analysis.create_chronological_split()
        analysis.build_models()

        # Mock CV summaries: Random Forest has higher CV mean, Logistic Regression lower
        analysis.search_summaries = [
            {
                "model": "Logistic Regression",
                "cv_macro_f1_mean": 0.70,
                "cv_macro_f1_std": 0.05,
                "mean_fit_time": 0.1,
            },
            {
                "model": "Random Forest",
                "cv_macro_f1_mean": 0.90,
                "cv_macro_f1_std": 0.05,
                "mean_fit_time": 0.2,
            },
            {
                "model": "Extra Trees",
                "cv_macro_f1_mean": 0.80,
                "cv_macro_f1_std": 0.05,
                "mean_fit_time": 0.2,
            },
            {
                "model": "Histogram Gradient Boosting",
                "cv_macro_f1_mean": 0.60,
                "cv_macro_f1_std": 0.05,
                "mean_fit_time": 0.1,
            },
        ]

        for name in analysis.models:
            analysis.best_params[name] = {}

        analysis.refit_and_export_production_models()

        assert analysis.cv_selected_model_name == "Random Forest"
        artifact = joblib.load(output_dir / "cv_selected_production_model.joblib")
        assert artifact["model_type"] == "Random Forest"

    def test_deterministic_tie_handling(
        self, binary_chronological_dataset, tmp_path, minimal_grids
    ):
        """Test 4: verify tie-breaking order (std, fit time, alphabetical)."""
        analysis, _ = create_analysis(
            tmp_path, minimal_grids, binary_chronological_dataset
        )
        analysis.load()
        analysis.create_chronological_split()
        analysis.build_models()

        # Helper to build full search summaries for all 4 models
        def make_summaries(overrides):
            defaults = {
                "Logistic Regression": {
                    "cv_macro_f1_mean": 0.50,
                    "cv_macro_f1_std": 0.10,
                    "mean_fit_time": 0.1,
                },
                "Random Forest": {
                    "cv_macro_f1_mean": 0.50,
                    "cv_macro_f1_std": 0.10,
                    "mean_fit_time": 0.1,
                },
                "Extra Trees": {
                    "cv_macro_f1_mean": 0.50,
                    "cv_macro_f1_std": 0.10,
                    "mean_fit_time": 0.1,
                },
                "Histogram Gradient Boosting": {
                    "cv_macro_f1_mean": 0.50,
                    "cv_macro_f1_std": 0.10,
                    "mean_fit_time": 0.1,
                },
            }
            for model_name, props in overrides.items():
                defaults[model_name].update(props)
            return [{"model": m, **vals} for m, vals in defaults.items()]

        for name in analysis.models:
            analysis.best_params[name] = {}

        # Case A: Tied mean score, lower std wins
        analysis.search_summaries = make_summaries(
            {
                "Extra Trees": {
                    "cv_macro_f1_mean": 0.85,
                    "cv_macro_f1_std": 0.10,
                    "mean_fit_time": 0.2,
                },
                "Random Forest": {
                    "cv_macro_f1_mean": 0.85,
                    "cv_macro_f1_std": 0.02,
                    "mean_fit_time": 0.3,
                },
            }
        )
        analysis.refit_and_export_production_models()
        assert analysis.cv_selected_model_name == "Random Forest"

        # Case B: Tied mean score and std, lower mean_fit_time wins
        analysis.search_summaries = make_summaries(
            {
                "Random Forest": {
                    "cv_macro_f1_mean": 0.85,
                    "cv_macro_f1_std": 0.05,
                    "mean_fit_time": 0.3,
                },
                "Extra Trees": {
                    "cv_macro_f1_mean": 0.85,
                    "cv_macro_f1_std": 0.05,
                    "mean_fit_time": 0.1,
                },
            }
        )
        analysis.refit_and_export_production_models()
        assert analysis.cv_selected_model_name == "Extra Trees"

        # Case C: Tied mean score, std, and fit time -> alphabetical model name
        analysis.search_summaries = make_summaries(
            {
                "Random Forest": {
                    "cv_macro_f1_mean": 0.85,
                    "cv_macro_f1_std": 0.05,
                    "mean_fit_time": 0.2,
                },
                "Logistic Regression": {
                    "cv_macro_f1_mean": 0.85,
                    "cv_macro_f1_std": 0.05,
                    "mean_fit_time": 0.2,
                },
            }
        )
        analysis.refit_and_export_production_models()
        assert analysis.cv_selected_model_name == "Logistic Regression"

    def test_full_data_refit(
        self, binary_chronological_dataset, tmp_path, minimal_grids
    ):
        """Test 5: selected production model pipeline is refitted on all available rows."""
        analysis, output_dir = create_analysis(
            tmp_path, minimal_grids, binary_chronological_dataset
        )
        analysis.run()

        cv_selected_artifact = joblib.load(
            output_dir / "cv_selected_production_model.joblib"
        )
        selected_pipeline = cv_selected_artifact["model"]

        # Number of samples in fitted preprocessor scaler / imputer
        imputer = (
            selected_pipeline.named_steps["preprocessor"]
            .named_transformers_["numeric"]
            .named_steps["imputer"]
        )
        assert imputer.n_features_in_ == 2

    def test_family_consistency(
        self, binary_chronological_dataset, tmp_path, minimal_grids
    ):
        """Test 6: selected model matches highest-ranked CV family and parameters, and gives identical predictions as family artifact."""
        analysis, output_dir = create_analysis(
            tmp_path, minimal_grids, binary_chronological_dataset
        )
        analysis.run()

        selected_name = analysis.cv_selected_model_name
        slug = analysis.slug(selected_name)

        family_artifact = joblib.load(output_dir / f"production_{slug}.joblib")
        cv_selected_artifact = joblib.load(
            output_dir / "cv_selected_production_model.joblib"
        )

        assert cv_selected_artifact["model_type"] == selected_name
        assert (
            cv_selected_artifact["best_parameters"]
            == family_artifact["best_parameters"]
        )

        X_sample = analysis.data[list(analysis.cfg.features)].iloc[:5]
        pred_family = family_artifact["model"].predict(X_sample)
        pred_cv_selected = cv_selected_artifact["model"].predict(X_sample)

        np.testing.assert_array_equal(pred_family, pred_cv_selected)

        proba_family = family_artifact["model"].predict_proba(X_sample)
        proba_cv_selected = cv_selected_artifact["model"].predict_proba(X_sample)

        np.testing.assert_allclose(proba_family, proba_cv_selected)

    def test_manifest_consistency(
        self, binary_chronological_dataset, tmp_path, minimal_grids
    ):
        """Test 7: model_export_manifest.csv structure and flags."""
        analysis, output_dir = create_analysis(
            tmp_path, minimal_grids, binary_chronological_dataset
        )
        analysis.run()

        manifest_df = pd.read_csv(output_dir / "model_export_manifest.csv")

        # Check explicit row for cv_selected_production_model.joblib
        alias_rows = manifest_df.loc[
            manifest_df["filename"] == "cv_selected_production_model.joblib"
        ]
        assert len(alias_rows) == 1
        alias_row = alias_rows.iloc[0]
        assert alias_row["model_role"] == "cv_selected_production_full_data_refit"
        assert alias_row["artifact_kind"] == "cv_selected_production_alias"
        assert alias_row["cv_selected"] is True or alias_row["cv_selected"] == "True"
        assert alias_row["cv_selection_rank"] == 1

        # Check family production rows
        family_rows = manifest_df.loc[
            manifest_df["artifact_kind"] == "family_production_model"
        ]
        assert len(family_rows) == 4

        # Exactly one family row has cv_selected True
        selected_family_rows = family_rows.loc[
            (family_rows["cv_selected"] == True)
            | (family_rows["cv_selected"] == "True")
        ]
        assert len(selected_family_rows) == 1

        # Unique sequential ranks 1..4
        ranks = sorted(family_rows["cv_selection_rank"].astype(int).tolist())
        assert ranks == [1, 2, 3, 4]

    def test_generated_files_consistency(
        self, binary_chronological_dataset, tmp_path, minimal_grids
    ):
        """Test 8: generated_files.txt includes cv_selected_production_model.joblib and excludes legacy."""
        analysis, output_dir = create_analysis(
            tmp_path, minimal_grids, binary_chronological_dataset
        )
        analysis.run()

        files_text = (output_dir / "generated_files.txt").read_text()
        file_list = files_text.splitlines()

        assert "cv_selected_production_model.joblib" in file_list
        assert "best_overall_model.joblib" not in file_list

    def test_metadata_consistency(
        self, binary_chronological_dataset, tmp_path, minimal_grids
    ):
        """Test 9: run_metadata.json contains cv_selected_production_model section."""
        analysis, output_dir = create_analysis(
            tmp_path, minimal_grids, binary_chronological_dataset
        )
        analysis.run()

        metadata_path = output_dir / "run_metadata.json"
        with metadata_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        assert "cv_selected_production_model" in data
        cv_meta = data["cv_selected_production_model"]

        assert cv_meta["selected_model_type"] == analysis.cv_selected_model_name
        assert cv_meta["selection_partition"] == "chronological_training_partition"
        assert cv_meta["selection_metric"] == "macro_f1"
        assert cv_meta["selection_score_mean"] == analysis.cv_selected_score_mean
        assert cv_meta["selection_score_std"] == analysis.cv_selected_score_std
        assert cv_meta["holdout_used_for_selection"] is False
        assert cv_meta["nested_cv"] is False
        assert cv_meta["production_refit_partition"] == "all_available_rows"
        assert cv_meta["artifact_filename"] == "cv_selected_production_model.joblib"

    def test_no_ambiguous_user_facing_terminology(
        self, binary_chronological_dataset, tmp_path, minimal_grids
    ):
        """Test 10: active source code and README do not contain unqualified 'best overall model' / 'best_overall_model.joblib'."""
        from pathlib import Path

        repo_root = Path(__file__).parent.parent
        readme_text = (repo_root / "README.md").read_text(encoding="utf-8")
        src_text = (repo_root / "viclassify_cli.py").read_text(encoding="utf-8")

        assert "best_overall_model.joblib" not in readme_text
        assert "best_overall_model.joblib" not in src_text
        assert "best_overall_production_full_data_refit" not in src_text
