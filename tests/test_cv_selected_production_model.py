"""Tests for CV-selected production model selection, artifact, metadata, ranking, and tie-breaking rules."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from viclassify_cli import ClassificationAnalysis, Config


def make_mock_analysis(metric="macro_f1"):
    cfg = Config(
        data=Path("dummy.csv"),
        output=Path("dummy_out"),
        target="Y",
        features=("X1", "X2"),
        class_order=None,
        test_fraction=0.2,
        cv_folds=2,
        random_state=42,
        n_jobs=1,
        permutation_repeats=1,
        dpi=150,
        selection_metric=metric,
    )
    return ClassificationAnalysis(cfg)


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


# --- Unit Tests for rank_model_families ---


def test_1_rank_highest_mean_score_wins():
    """Test 1: highest mean score wins."""
    analysis = make_mock_analysis()
    df = pd.DataFrame(
        [
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
        ]
    )
    ranked = analysis.rank_model_families(df)
    assert ranked.iloc[0]["model"] == "Random Forest"
    assert ranked.iloc[0]["cv_selection_rank"] == 1
    assert ranked.iloc[1]["model"] == "Extra Trees"
    assert ranked.iloc[1]["cv_selection_rank"] == 2
    assert ranked.iloc[2]["model"] == "Logistic Regression"
    assert ranked.iloc[2]["cv_selection_rank"] == 3


def test_2_rank_lower_std_breaks_mean_tie():
    """Test 2: lower standard deviation breaks a mean tie."""
    analysis = make_mock_analysis()
    df = pd.DataFrame(
        [
            {
                "model": "Extra Trees",
                "cv_macro_f1_mean": 0.85,
                "cv_macro_f1_std": 0.10,
                "mean_fit_time": 0.1,
            },
            {
                "model": "Random Forest",
                "cv_macro_f1_mean": 0.85,
                "cv_macro_f1_std": 0.02,
                "mean_fit_time": 0.5,
            },
        ]
    )
    ranked = analysis.rank_model_families(df)
    assert ranked.iloc[0]["model"] == "Random Forest"
    assert ranked.iloc[0]["cv_selection_rank"] == 1


def test_3_rank_lower_fit_time_breaks_mean_and_std_ties():
    """Test 3: lower mean fit time breaks mean and standard-deviation ties."""
    analysis = make_mock_analysis()
    df = pd.DataFrame(
        [
            {
                "model": "Random Forest",
                "cv_macro_f1_mean": 0.85,
                "cv_macro_f1_std": 0.05,
                "mean_fit_time": 0.3,
            },
            {
                "model": "Extra Trees",
                "cv_macro_f1_mean": 0.85,
                "cv_macro_f1_std": 0.05,
                "mean_fit_time": 0.1,
            },
        ]
    )
    ranked = analysis.rank_model_families(df)
    assert ranked.iloc[0]["model"] == "Extra Trees"


def test_4_rank_alphabetical_name_is_final_tie_breaker():
    """Test 4: alphabetical model name is the final tie-breaker."""
    analysis = make_mock_analysis()
    df = pd.DataFrame(
        [
            {
                "model": "Random Forest",
                "cv_macro_f1_mean": 0.85,
                "cv_macro_f1_std": 0.05,
                "mean_fit_time": 0.2,
            },
            {
                "model": "Logistic Regression",
                "cv_macro_f1_mean": 0.85,
                "cv_macro_f1_std": 0.05,
                "mean_fit_time": 0.2,
            },
            {
                "model": "Extra Trees",
                "cv_macro_f1_mean": 0.85,
                "cv_macro_f1_std": 0.05,
                "mean_fit_time": 0.2,
            },
        ]
    )
    ranked = analysis.rank_model_families(df)
    assert ranked["model"].tolist() == [
        "Extra Trees",
        "Logistic Regression",
        "Random Forest",
    ]


def test_5_rank_primary_criterion_overrides_all_tie_breakers():
    """Test 5: primary criterion overrides all tie-breakers."""
    analysis = make_mock_analysis()
    df = pd.DataFrame(
        [
            {
                "model": "Extra Trees",
                "cv_macro_f1_mean": 0.800,
                "cv_macro_f1_std": 0.01,
                "mean_fit_time": 0.1,
            },
            {
                "model": "Random Forest",
                "cv_macro_f1_mean": 0.801,
                "cv_macro_f1_std": 0.15,
                "mean_fit_time": 2.0,
            },
        ]
    )
    ranked = analysis.rank_model_families(df)
    assert ranked.iloc[0]["model"] == "Random Forest"


def test_6_rank_std_overrides_speed_and_name():
    """Test 6: standard deviation overrides speed and name."""
    analysis = make_mock_analysis()
    df = pd.DataFrame(
        [
            {
                "model": "Extra Trees",
                "cv_macro_f1_mean": 0.80,
                "cv_macro_f1_std": 0.05,
                "mean_fit_time": 0.1,
            },
            {
                "model": "Random Forest",
                "cv_macro_f1_mean": 0.80,
                "cv_macro_f1_std": 0.01,
                "mean_fit_time": 0.5,
            },
        ]
    )
    ranked = analysis.rank_model_families(df)
    assert ranked.iloc[0]["model"] == "Random Forest"


def test_7_rank_fit_time_overrides_alphabetical_name():
    """Test 7: fit time overrides alphabetical name."""
    analysis = make_mock_analysis()
    df = pd.DataFrame(
        [
            {
                "model": "Extra Trees",
                "cv_macro_f1_mean": 0.80,
                "cv_macro_f1_std": 0.05,
                "mean_fit_time": 0.3,
            },
            {
                "model": "Random Forest",
                "cv_macro_f1_mean": 0.80,
                "cv_macro_f1_std": 0.05,
                "mean_fit_time": 0.1,
            },
        ]
    )
    ranked = analysis.rank_model_families(df)
    assert ranked.iloc[0]["model"] == "Random Forest"


def test_8_rank_input_row_order_independence():
    """Test 8: dictionary and input-row order do not affect ranking."""
    analysis = make_mock_analysis()
    df = pd.DataFrame(
        [
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
    )
    ranked_orig = analysis.rank_model_families(df)
    ranked_rev = analysis.rank_model_families(df.iloc[::-1].reset_index(drop=True))
    ranked_shuffled = analysis.rank_model_families(
        df.iloc[[2, 0, 3, 1]].reset_index(drop=True)
    )

    assert (
        ranked_orig["model"].tolist()
        == ranked_rev["model"].tolist()
        == ranked_shuffled["model"].tolist()
    )
    assert (
        ranked_orig["cv_selection_rank"].tolist()
        == ranked_rev["cv_selection_rank"].tolist()
        == ranked_shuffled["cv_selection_rank"].tolist()
    )


def test_9_rank_values_validity():
    """Test 9: rank values are valid."""
    analysis = make_mock_analysis()
    df = pd.DataFrame(
        [
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
    )
    ranked = analysis.rank_model_families(df)
    ranks = ranked["cv_selection_rank"].tolist()
    assert ranks == [1, 2, 3, 4]
    assert all(isinstance(r, (int, np.integer)) for r in ranks)
    assert len(set(ranks)) == len(df)


def test_10_configured_metric_controls_ranking():
    """Test 10: configured metric controls ranking."""
    analysis_f1 = make_mock_analysis("macro_f1")
    analysis_acc = make_mock_analysis("balanced_accuracy")

    df_f1 = pd.DataFrame(
        [
            {
                "model": "Logistic Regression",
                "cv_macro_f1_mean": 0.90,
                "cv_macro_f1_std": 0.05,
                "mean_fit_time": 0.1,
            },
            {
                "model": "Random Forest",
                "cv_macro_f1_mean": 0.70,
                "cv_macro_f1_std": 0.05,
                "mean_fit_time": 0.1,
            },
        ]
    )
    df_acc = pd.DataFrame(
        [
            {
                "model": "Logistic Regression",
                "cv_balanced_accuracy_mean": 0.60,
                "cv_balanced_accuracy_std": 0.05,
                "mean_fit_time": 0.1,
            },
            {
                "model": "Random Forest",
                "cv_balanced_accuracy_mean": 0.85,
                "cv_balanced_accuracy_std": 0.05,
                "mean_fit_time": 0.1,
            },
        ]
    )

    ranked_f1 = analysis_f1.rank_model_families(df_f1)
    ranked_acc = analysis_acc.rank_model_families(df_acc)

    assert ranked_f1.iloc[0]["model"] == "Logistic Regression"
    assert ranked_acc.iloc[0]["model"] == "Random Forest"


@pytest.mark.integration
def test_11_holdout_results_cannot_change_cv_rank(
    binary_chronological_dataset, tmp_path, minimal_grids
):
    """Test 11: holdout results cannot change CV rank."""
    analysis, _ = create_analysis(tmp_path, minimal_grids, binary_chronological_dataset)
    analysis.load()
    analysis.create_chronological_split()
    analysis.build_models()

    analysis.search_summaries = [
        {
            "model": "Logistic Regression",
            "cv_macro_f1_mean": 0.50,
            "cv_macro_f1_std": 0.05,
            "mean_fit_time": 0.1,
        },
        {
            "model": "Random Forest",
            "cv_macro_f1_mean": 0.70,
            "cv_macro_f1_std": 0.05,
            "mean_fit_time": 0.1,
        },
        {
            "model": "Extra Trees",
            "cv_macro_f1_mean": 0.85,
            "cv_macro_f1_std": 0.05,
            "mean_fit_time": 0.1,
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

    analysis.holdout_rows = [
        {"model": "Logistic Regression", "macro_f1": 0.50},
        {"model": "Random Forest", "macro_f1": 0.99},
        {"model": "Extra Trees", "macro_f1": 0.10},
        {"model": "Histogram Gradient Boosting", "macro_f1": 0.60},
    ]

    analysis.refit_and_export_production_models()
    assert analysis.cv_selected_model_name == "Extra Trees"


def test_12_rank_exact_stored_values_not_rounded():
    """Test 12: exact stored values are not rounded into a tie."""
    analysis = make_mock_analysis()
    df = pd.DataFrame(
        [
            {
                "model": "Random Forest",
                "cv_macro_f1_mean": 0.80003,
                "cv_macro_f1_std": 0.05,
                "mean_fit_time": 0.1,
            },
            {
                "model": "Extra Trees",
                "cv_macro_f1_mean": 0.80004,
                "cv_macro_f1_std": 0.05,
                "mean_fit_time": 0.1,
            },
        ]
    )
    ranked = analysis.rank_model_families(df)
    assert ranked.iloc[0]["model"] == "Extra Trees"
    assert ranked.iloc[1]["model"] == "Random Forest"


@pytest.mark.parametrize(
    "missing_col",
    ["model", "cv_macro_f1_mean", "cv_macro_f1_std", "mean_fit_time"],
)
def test_13_rank_missing_required_column_fails(missing_col):
    """Test 13: missing required column fails clearly."""
    analysis = make_mock_analysis()
    df = pd.DataFrame(
        [
            {
                "model": "Random Forest",
                "cv_macro_f1_mean": 0.80,
                "cv_macro_f1_std": 0.05,
                "mean_fit_time": 0.1,
            },
        ]
    )
    df = df.drop(columns=[missing_col])
    with pytest.raises(ValueError, match="missing required columns"):
        analysis.rank_model_families(df)


def test_14_rank_empty_summary_fails():
    """Test 14: empty summary fails clearly."""
    analysis = make_mock_analysis()
    df = pd.DataFrame(
        columns=["model", "cv_macro_f1_mean", "cv_macro_f1_std", "mean_fit_time"]
    )
    with pytest.raises(ValueError, match="empty"):
        analysis.rank_model_families(df)


def test_15_rank_duplicate_model_names_fail():
    """Test 15: duplicate model names fail clearly."""
    analysis = make_mock_analysis()
    df = pd.DataFrame(
        [
            {
                "model": "Random Forest",
                "cv_macro_f1_mean": 0.80,
                "cv_macro_f1_std": 0.05,
                "mean_fit_time": 0.1,
            },
            {
                "model": "Random Forest",
                "cv_macro_f1_mean": 0.85,
                "cv_macro_f1_std": 0.05,
                "mean_fit_time": 0.1,
            },
        ]
    )
    with pytest.raises(ValueError, match="duplicate model-family rows"):
        analysis.rank_model_families(df)


@pytest.mark.parametrize(
    "col", ["cv_macro_f1_mean", "cv_macro_f1_std", "mean_fit_time"]
)
@pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf, "invalid_text"])
def test_16_rank_invalid_numeric_values_fail(col, bad_val):
    """Test 16: invalid numeric values fail clearly."""
    analysis = make_mock_analysis()
    df = pd.DataFrame(
        [
            {
                "model": "Random Forest",
                "cv_macro_f1_mean": 0.80,
                "cv_macro_f1_std": 0.05,
                "mean_fit_time": 0.1,
            },
        ]
    )
    df.loc[0, col] = bad_val
    with pytest.raises(ValueError):
        analysis.rank_model_families(df)


def test_17_rank_negative_std_fails():
    """Test 17: negative standard deviation fails."""
    analysis = make_mock_analysis()
    df = pd.DataFrame(
        [
            {
                "model": "Random Forest",
                "cv_macro_f1_mean": 0.80,
                "cv_macro_f1_std": -0.05,
                "mean_fit_time": 0.1,
            },
        ]
    )
    with pytest.raises(ValueError, match="negative standard deviation"):
        analysis.rank_model_families(df)


def test_18_rank_negative_fit_time_fails():
    """Test 18: negative fit time fails."""
    analysis = make_mock_analysis()
    df = pd.DataFrame(
        [
            {
                "model": "Random Forest",
                "cv_macro_f1_mean": 0.80,
                "cv_macro_f1_std": 0.05,
                "mean_fit_time": -0.1,
            },
        ]
    )
    with pytest.raises(ValueError, match="negative fit time"):
        analysis.rank_model_families(df)


@pytest.mark.parametrize("bad_name", ["", "   ", None])
def test_19_rank_empty_model_name_fails(bad_name):
    """Test 19: empty model name fails."""
    analysis = make_mock_analysis()
    df = pd.DataFrame(
        [
            {
                "model": bad_name,
                "cv_macro_f1_mean": 0.80,
                "cv_macro_f1_std": 0.05,
                "mean_fit_time": 0.1,
            },
        ]
    )
    with pytest.raises(ValueError):
        analysis.rank_model_families(df)


# --- Integration Tests for Artifacts, Manifests, and Metadata ---


@pytest.mark.integration
def test_20_manifest_matches_ranking(
    binary_chronological_dataset, tmp_path, minimal_grids
):
    """Test 20: manifest matches ranking."""
    analysis, output_dir = create_analysis(
        tmp_path, minimal_grids, binary_chronological_dataset
    )
    analysis.run()

    manifest_df = pd.read_csv(output_dir / "model_export_manifest.csv")

    family_rows = manifest_df.loc[
        manifest_df["artifact_kind"] == "family_production_model"
    ].sort_values("cv_selection_rank")
    assert len(family_rows) == 4
    assert family_rows["cv_selection_rank"].tolist() == [1, 2, 3, 4]

    selected_family = family_rows.iloc[0]
    assert (
        selected_family["cv_selected"] is True
        or selected_family["cv_selected"] == "True"
        or selected_family["cv_selected"] == True
    )

    for _, row in family_rows.iloc[1:].iterrows():
        assert (
            row["cv_selected"] is False
            or row["cv_selected"] == "False"
            or row["cv_selected"] == False
        )

    assert all(
        row["tie_breaking_rule"]
        == ClassificationAnalysis.CV_SELECTION_TIE_BREAKING_RULE
        for _, row in manifest_df.iterrows()
    )

    alias_row = manifest_df.loc[
        manifest_df["artifact_kind"] == "cv_selected_production_alias"
    ].iloc[0]
    assert alias_row["model_type"] == selected_family["model_type"]
    assert alias_row["cv_selection_rank"] == 1
    assert alias_row["selection_score_mean"] == selected_family["selection_score_mean"]


@pytest.mark.integration
def test_21_artifact_metadata_matches_ranking(
    binary_chronological_dataset, tmp_path, minimal_grids
):
    """Test 21: artifact metadata matches ranking."""
    analysis, output_dir = create_analysis(
        tmp_path, minimal_grids, binary_chronological_dataset
    )
    analysis.run()

    artifact = joblib.load(output_dir / "cv_selected_production_model.joblib")
    assert artifact["model_type"] == analysis.cv_selected_model_name
    assert artifact["cv_selection_rank"] == 1
    assert artifact["selection_score_mean"] == analysis.cv_selected_score_mean
    assert artifact["selection_score_std"] == analysis.cv_selected_score_std
    assert (
        artifact["tie_breaking_rule"]
        == ClassificationAnalysis.CV_SELECTION_TIE_BREAKING_RULE
    )
    assert artifact["holdout_used_for_selection"] is False
    assert artifact["nested_cv"] is False


@pytest.mark.integration
def test_22_run_metadata_contains_complete_ranking(
    binary_chronological_dataset, tmp_path, minimal_grids
):
    """Test 22: run metadata contains complete ranking."""
    analysis, output_dir = create_analysis(
        tmp_path, minimal_grids, binary_chronological_dataset
    )
    analysis.run()

    with (output_dir / "run_metadata.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)

    cv_meta = meta["cv_selected_production_model"]
    assert cv_meta["cv_selection_rank"] == 1
    assert (
        cv_meta["tie_breaking_rule"]
        == ClassificationAnalysis.CV_SELECTION_TIE_BREAKING_RULE
    )
    assert cv_meta["holdout_used_for_selection"] is False

    ranking = cv_meta["model_family_cv_ranking"]
    assert len(ranking) == 4
    ranks = [item["cv_selection_rank"] for item in ranking]
    assert ranks == [1, 2, 3, 4]

    models = [item["model"] for item in ranking]
    assert len(set(models)) == 4


@pytest.mark.integration
def test_23_family_and_alias_artifacts_agree(
    binary_chronological_dataset, tmp_path, minimal_grids
):
    """Test 23: family-specific and alias artifacts agree."""
    analysis, output_dir = create_analysis(
        tmp_path, minimal_grids, binary_chronological_dataset
    )
    analysis.run()

    selected_name = analysis.cv_selected_model_name
    slug = analysis.slug(selected_name)

    family_artifact = joblib.load(output_dir / f"production_{slug}.joblib")
    alias_artifact = joblib.load(output_dir / "cv_selected_production_model.joblib")

    assert family_artifact["model_type"] == alias_artifact["model_type"]
    assert family_artifact["best_parameters"] == alias_artifact["best_parameters"]
    assert (
        family_artifact["estimator_class_order"]
        == alias_artifact["estimator_class_order"]
    )

    X_sample = analysis.data[list(analysis.cfg.features)].iloc[:5]
    pred_fam = family_artifact["model"].predict(X_sample)
    pred_alias = alias_artifact["model"].predict(X_sample)
    np.testing.assert_array_equal(pred_fam, pred_alias)

    proba_fam = family_artifact["model"].predict_proba(X_sample)
    proba_alias = alias_artifact["model"].predict_proba(X_sample)
    np.testing.assert_allclose(proba_fam, proba_alias)


def test_24_no_ambiguous_user_facing_terminology():
    """Test 24: active source code and README do not contain unqualified legacy artifacts/terms."""
    repo_root = Path(__file__).parent.parent
    readme_text = (repo_root / "README.md").read_text(encoding="utf-8")
    src_text = (repo_root / "viclassify_cli.py").read_text(encoding="utf-8")

    assert "best_overall_model.joblib" not in readme_text
    assert "best_overall_model.joblib" not in src_text
    assert "best_overall_production_full_data_refit" not in src_text
