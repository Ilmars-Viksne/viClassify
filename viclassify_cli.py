from __future__ import annotations

import argparse
import json
import platform
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Config:
    data: Path
    output: Path
    target: str
    features: tuple[str, ...]
    class_order: tuple[str, ...] | None
    test_fraction: float
    cv_folds: int
    random_state: int
    n_jobs: int
    permutation_repeats: int
    dpi: int
    near_constant_threshold: float = 0.99
    selection_metric: str = "macro_f1"


class ClassificationAnalysis:
    MODEL_NAMES = (
        "Logistic Regression",
        "Random Forest",
        "Extra Trees",
        "Histogram Gradient Boosting",
    )

    SCORING = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "macro_precision": make_scorer(
            precision_score,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": make_scorer(
            recall_score,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": make_scorer(
            f1_score,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": make_scorer(
            f1_score,
            average="weighted",
            zero_division=0,
        ),
    }

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.cfg.output.mkdir(parents=True, exist_ok=True)

        self.df: pd.DataFrame | None = None
        self.classes: list[str] = []

        self.models: dict[str, Pipeline] = {}
        self.grids: dict[str, dict[str, list[Any]]] = {}
        self.best_params: dict[str, dict[str, Any]] = {}

        self.evaluation_models: dict[str, Pipeline] = {}
        self.production_models: dict[str, Pipeline] = {}

        self.search_summaries: list[dict[str, Any]] = []
        self.holdout_rows: list[dict[str, Any]] = []

        self.train_indices: np.ndarray | None = None
        self.test_indices: np.ndarray | None = None

        self.generated_files: set[str] = set()

    @property
    def data(self) -> pd.DataFrame:
        if self.df is None:
            raise RuntimeError("Data have not been loaded")
        return self.df

    @property
    def selection_mean_column(self) -> str:
        return f"cv_{self.cfg.selection_metric}_mean"

    @property
    def selection_std_column(self) -> str:
        return f"cv_{self.cfg.selection_metric}_std"

    def path(self, name: str) -> Path:
        self.generated_files.add(name)
        return self.cfg.output / name

    @staticmethod
    def slug(name: str) -> str:
        value = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
        return value.strip("_")

    @staticmethod
    def heading(text: str) -> None:
        print("\n" + "=" * 78)
        print(text)
        print("=" * 78)

    def savefig(self, name: str) -> None:
        plt.tight_layout()
        plt.savefig(
            self.path(name),
            dpi=self.cfg.dpi,
            bbox_inches="tight",
        )
        plt.close()

    def split_data(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        if self.train_indices is None or self.test_indices is None:
            raise RuntimeError("Chronological split has not been created")

        X = self.data[list(self.cfg.features)]
        y = self.data[self.cfg.target]

        return (
            X.iloc[self.train_indices].copy(),
            X.iloc[self.test_indices].copy(),
            y.iloc[self.train_indices].copy(),
            y.iloc[self.test_indices].copy(),
        )

    def load(self) -> None:
        self.heading("1. Loading and validating data")

        if not self.cfg.data.exists():
            raise FileNotFoundError(self.cfg.data.resolve())

        df = pd.read_csv(
            self.cfg.data,
            skipinitialspace=True,
        )
        df.columns = df.columns.astype(str).str.strip()

        duplicated_columns = df.columns[df.columns.duplicated()].tolist()
        if duplicated_columns:
            raise ValueError(
                f"Input CSV contains duplicate column names: "
                f"{sorted(set(duplicated_columns))}"
            )

        required = [*self.cfg.features, self.cfg.target]
        missing_columns = sorted(set(required) - set(df.columns))

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        if df.empty:
            raise ValueError("Dataset is empty")

        if "Sequence" in df.columns:
            raise ValueError(
                "Input already contains reserved column 'Sequence'"
            )

        unused = sorted(set(df.columns) - set(required))

        pd.DataFrame(
            {
                "column": unused,
                "reason": "not selected as feature or target",
            }
        ).to_csv(
            self.path("unused_columns.csv"),
            index=False,
        )

        if unused:
            warnings.warn(
                f"Unused input columns: {unused}",
                stacklevel=2,
            )

        df = df.copy()
        df.insert(
            0,
            "Sequence",
            np.arange(1, len(df) + 1),
        )

        conversion_rows: list[dict[str, Any]] = []
        infinity_rows: list[dict[str, Any]] = []

        for col in self.cfg.features:
            original_missing = int(df[col].isna().sum())
            numeric = pd.to_numeric(df[col], errors="coerce")
            after_conversion = int(numeric.isna().sum())

            positive_inf = int(np.isposinf(numeric).sum())
            negative_inf = int(np.isneginf(numeric).sum())

            conversion_rows.append(
                {
                    "feature": col,
                    "original_missing": original_missing,
                    "missing_after_conversion": after_conversion,
                    "new_missing_from_conversion": (
                        after_conversion - original_missing
                    ),
                }
            )

            infinity_rows.append(
                {
                    "feature": col,
                    "positive_infinity_count": positive_inf,
                    "negative_infinity_count": negative_inf,
                    "total_infinity_count": positive_inf + negative_inf,
                }
            )

            df[col] = numeric.replace(
                [np.inf, -np.inf],
                np.nan,
            )

        pd.DataFrame(conversion_rows).to_csv(
            self.path("numeric_conversion_problems.csv"),
            index=False,
        )

        pd.DataFrame(infinity_rows).to_csv(
            self.path("infinite_values.csv"),
            index=False,
        )

        all_missing_features = [
            feature
            for feature in self.cfg.features
            if df[feature].isna().all()
        ]

        if all_missing_features:
            raise ValueError(
                "The following selected features contain no usable numeric "
                f"values: {all_missing_features}"
            )

        y = df[self.cfg.target].astype("string").str.strip()

        if y.isna().any() or y.eq("").any():
            raise ValueError(
                "Target contains missing or empty labels"
            )

        df[self.cfg.target] = y
        observed = sorted(y.unique().tolist())

        if len(observed) < 2:
            raise ValueError(
                "At least two target classes are required"
            )

        if self.cfg.class_order:
            unknown = sorted(
                set(self.cfg.class_order) - set(observed)
            )

            if unknown:
                warnings.warn(
                    f"Requested labels not observed: {unknown}",
                    stacklevel=2,
                )

            self.classes = [
                label
                for label in self.cfg.class_order
                if label in observed
            ]

            self.classes += [
                label
                for label in observed
                if label not in self.classes
            ]
        else:
            self.classes = observed

        self.df = df

        print(
            f"Rows: {len(df):,}; "
            f"predictors: {len(self.cfg.features)}; "
            f"classes: {self.classes}"
        )

    def audit_and_describe(self) -> None:
        self.heading("2. Data audit and descriptive statistics")

        df = self.data
        features = list(self.cfg.features)

        missing = df.isna().sum().to_frame("missing_count")
        missing["missing_percentage"] = (
            100 * missing["missing_count"] / len(df)
        )
        missing.to_csv(
            self.path("missing_values.csv"),
            index_label="column",
        )

        duplicate_columns = [*features, self.cfg.target]
        duplicate_mask = df.duplicated(
            duplicate_columns,
            keep=False,
        )

        df.loc[duplicate_mask].to_csv(
            self.path("duplicate_rows.csv"),
            index=False,
        )

        counts = (
            df[self.cfg.target]
            .value_counts()
            .reindex(self.classes, fill_value=0)
        )

        distribution = pd.DataFrame(
            {
                "count": counts,
                "percentage": 100 * counts / len(df),
            }
        )

        distribution.to_csv(
            self.path("class_distribution.csv"),
            index_label="class",
        )

        summary = df[features].describe(
            percentiles=[
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
                0.99,
            ]
        ).T

        summary["missing"] = df[features].isna().sum()
        summary["unique_values"] = df[features].nunique(
            dropna=True
        )
        summary["skewness"] = df[features].skew()
        summary["kurtosis"] = df[features].kurtosis()

        summary.to_csv(
            self.path("numerical_summary.csv"),
            index_label="feature",
        )

        grouped = (
            df.groupby(
                self.cfg.target,
                observed=True,
            )[features]
            .agg(
                [
                    "count",
                    "mean",
                    "median",
                    "std",
                    "min",
                    "max",
                ]
            )
        )

        grouped.to_csv(
            self.path("numerical_summary_by_class.csv")
        )

        audit = pd.DataFrame(index=features)
        audit["non_missing"] = df[features].count()
        audit["missing"] = df[features].isna().sum()
        audit["unique_values"] = df[features].nunique(
            dropna=True
        )
        audit["variance"] = df[features].var()

        audit["most_common_fraction"] = [
            (
                df[column]
                .value_counts(
                    normalize=True,
                    dropna=True,
                )
                .iloc[0]
            )
            if df[column].notna().any()
            else np.nan
            for column in features
        ]

        audit["constant"] = audit["unique_values"] <= 1
        audit["near_constant"] = (
            audit["most_common_fraction"]
            >= self.cfg.near_constant_threshold
        )

        audit.to_csv(
            self.path("feature_audit.csv"),
            index_label="feature",
        )

        constant_features = audit.index[
            audit["constant"]
        ].tolist()

        if constant_features:
            warnings.warn(
                f"Constant features detected: {constant_features}",
                stacklevel=2,
            )

    def correlations(self) -> None:
        self.heading("3. Correlation analysis")

        x = self.data[list(self.cfg.features)]

        for method in ("pearson", "spearman"):
            corr = x.corr(method=method)

            corr.to_csv(
                self.path(
                    f"{method}_correlation_matrix.csv"
                )
            )

            pairs: list[dict[str, Any]] = []

            for i, first in enumerate(corr.columns):
                for j in range(i + 1, len(corr.columns)):
                    second = corr.columns[j]
                    value = corr.iloc[i, j]

                    pairs.append(
                        {
                            "feature_1": first,
                            "feature_2": second,
                            "correlation": value,
                            "absolute_correlation": (
                                abs(value)
                                if pd.notna(value)
                                else np.nan
                            ),
                        }
                    )

            pairs_frame = pd.DataFrame(
                pairs,
                columns=[
                    "feature_1",
                    "feature_2",
                    "correlation",
                    "absolute_correlation",
                ],
            )

            if not pairs_frame.empty:
                pairs_frame = pairs_frame.sort_values(
                    "absolute_correlation",
                    ascending=False,
                    na_position="last",
                )

            pairs_frame.to_csv(
                self.path(
                    f"{method}_correlation_pairs.csv"
                ),
                index=False,
            )

            self.plot_correlation(corr, method)

    def plot_correlation(
        self,
        corr: pd.DataFrame,
        method: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(9, 8))

        im = ax.imshow(
            corr,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
        )

        ax.set_xticks(
            range(len(corr)),
            labels=corr.columns,
            rotation=45,
            ha="right",
        )

        ax.set_yticks(
            range(len(corr)),
            labels=corr.index,
        )

        for i in range(len(corr)):
            for j in range(len(corr)):
                value = corr.iloc[i, j]

                ax.text(
                    j,
                    i,
                    (
                        "NA"
                        if pd.isna(value)
                        else f"{value:.2f}"
                    ),
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if pd.notna(value)
                        and abs(value) > 0.65
                        else "black"
                    ),
                )

        fig.colorbar(
            im,
            ax=ax,
            label=f"{method.title()} correlation",
        )

        ax.set_title(
            f"{method.title()} Correlation Matrix"
        )

        self.savefig(
            f"{method}_correlation_matrix.png"
        )

    def exploratory_plots(self) -> None:
        self.heading("4. Exploratory diagnostic plots")

        df = self.data
        target = self.cfg.target
        features = list(self.cfg.features)

        counts = (
            df[target]
            .value_counts()
            .reindex(self.classes, fill_value=0)
        )

        plt.figure(figsize=(8, 5))

        bars = plt.bar(
            counts.index,
            counts.values,
            color="steelblue",
            edgecolor="black",
        )

        for bar, count in zip(bars, counts.values):
            percentage = 100 * count / len(df)

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{count:,}\n({percentage:.1f}%)",
                ha="center",
                va="bottom",
            )

        plt.title("Class Distribution")
        plt.xlabel("Class")
        plt.ylabel("Observations")
        plt.grid(axis="y", alpha=0.3)

        self.savefig("class_distribution.png")

        ncols = 2
        nrows = int(np.ceil(len(features) / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(14, 4 * nrows),
        )
        axes = np.atleast_1d(axes).ravel()

        for ax, feature in zip(axes, features):
            values = df[feature].dropna()

            ax.hist(
                values,
                bins=40,
                color="skyblue",
                edgecolor="black",
            )

            if not values.empty:
                ax.axvline(
                    values.mean(),
                    color="darkred",
                    linestyle="--",
                    label="Mean",
                )
                ax.axvline(
                    values.median(),
                    color="darkgreen",
                    linestyle=":",
                    label="Median",
                )
                ax.legend()

            ax.set_title(
                f"Distribution of {feature}"
            )
            ax.set_xlabel(feature)
            ax.set_ylabel("Frequency")
            ax.grid(alpha=0.3)

        for ax in axes[len(features):]:
            fig.delaxes(ax)

        self.savefig("feature_histograms.png")

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(14, 4.5 * nrows),
        )
        axes = np.atleast_1d(axes).ravel()

        for ax, feature in zip(axes, features):
            groups = [
                df.loc[
                    df[target] == label,
                    feature,
                ].dropna()
                for label in self.classes
            ]

            ax.boxplot(
                groups,
                showfliers=True,
            )
            ax.set_xticks(
                range(1, len(self.classes) + 1),
                labels=self.classes,
            )
            ax.set_title(
                f"{feature} by Class"
            )
            ax.set_xlabel("Class")
            ax.set_ylabel(feature)
            ax.grid(axis="y", alpha=0.3)

        for ax in axes[len(features):]:
            fig.delaxes(ax)

        self.savefig(
            "feature_boxplots_by_class.png"
        )

    def create_chronological_split(self) -> None:
        self.heading(
            "5. Creating chronological holdout"
        )

        n = len(self.data)
        split = int(
            np.floor(
                n * (1 - self.cfg.test_fraction)
            )
        )

        if not 0 < split < n:
            raise ValueError(
                "test-fraction creates an empty partition"
            )

        self.train_indices = np.arange(split)
        self.test_indices = np.arange(split, n)

        _, _, y_train, y_test = self.split_data()

        missing_train = sorted(
            set(self.classes) - set(y_train)
        )

        if missing_train:
            raise ValueError(
                "Chronological training partition lacks "
                f"classes: {missing_train}"
            )

        missing_test = sorted(
            set(self.classes) - set(y_test)
        )

        if missing_test:
            warnings.warn(
                "Chronological test partition lacks "
                f"classes: {missing_test}",
                stacklevel=2,
            )

        if len(y_test) < 10:
            warnings.warn(
                "The chronological holdout contains only "
                f"{len(y_test)} observations. Holdout metrics "
                "and permutation importance may be unstable.",
                stacklevel=2,
            )

        train_counts = (
            y_train.value_counts()
            .reindex(self.classes, fill_value=0)
        )
        test_counts = (
            y_test.value_counts()
            .reindex(self.classes, fill_value=0)
        )

        split_df = pd.DataFrame(
            {
                "training_count": train_counts,
                "training_percentage": (
                    100 * train_counts / len(y_train)
                ),
                "testing_count": test_counts,
                "testing_percentage": (
                    100 * test_counts / len(y_test)
                ),
            }
        )

        split_df.to_csv(
            self.path(
                "chronological_split_distribution.csv"
            ),
            index_label="class",
        )

        assignments = self.data[
            ["Sequence", self.cfg.target]
        ].copy()

        assignments["partition"] = np.where(
            assignments.index < split,
            "training",
            "testing",
        )

        assignments.to_csv(
            self.path(
                "chronological_split_assignments.csv"
            ),
            index=False,
        )

        positions = np.arange(len(self.classes))
        width = 0.38

        plt.figure(figsize=(9, 5))

        plt.bar(
            positions - width / 2,
            train_counts,
            width,
            label="Training",
        )
        plt.bar(
            positions + width / 2,
            test_counts,
            width,
            label="Testing",
        )

        plt.xticks(
            positions,
            self.classes,
        )
        plt.xlabel("Class")
        plt.ylabel("Observations")
        plt.title(
            "Chronological Split Distribution"
        )
        plt.grid(axis="y", alpha=0.3)
        plt.legend()

        self.savefig(
            "chronological_split_distribution.png"
        )

        figure_rows = len(self.cfg.features) + 1

        fig, axes = plt.subplots(
            figure_rows,
            1,
            figsize=(17, 2.4 * figure_rows),
            sharex=True,
        )

        axes = np.atleast_1d(axes)

        for ax, feature in zip(
            axes,
            self.cfg.features,
        ):
            ax.plot(
                self.data["Sequence"],
                self.data[feature],
                linewidth=0.7,
            )
            ax.axvline(
                split + 0.5,
                color="red",
                linestyle="--",
                label="Holdout boundary",
            )
            ax.set_ylabel(feature)
            ax.grid(alpha=0.3)

        class_to_int = {
            label: index
            for index, label in enumerate(self.classes)
        }

        axes[-1].scatter(
            self.data["Sequence"],
            self.data[self.cfg.target].map(
                class_to_int
            ),
            s=8,
        )

        axes[-1].axvline(
            split + 0.5,
            color="red",
            linestyle="--",
        )

        axes[-1].set_yticks(
            range(len(self.classes)),
            labels=self.classes,
        )
        axes[-1].set_ylabel("Class")
        axes[-1].set_xlabel(
            "Observation Sequence"
        )
        axes[-1].grid(alpha=0.3)

        axes[0].set_title(
            "Predictors and Classes by Sequence"
        )

        self.savefig(
            "predictors_and_classes_by_sequence.png"
        )

    def build_models(self) -> None:
        self.heading("6. Building model families")

        numeric = ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline(
                        [
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="median"
                                ),
                            ),
                            (
                                "scaler",
                                StandardScaler(),
                            ),
                        ]
                    ),
                    list(self.cfg.features),
                )
            ],
            remainder="drop",
        )

        tree = ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline(
                        [
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="median"
                                ),
                            ),
                        ]
                    ),
                    list(self.cfg.features),
                )
            ],
            remainder="drop",
        )

        random_state = self.cfg.random_state
        n_jobs = self.cfg.n_jobs

        self.models = {
            "Logistic Regression": Pipeline(
                [
                    (
                        "preprocessor",
                        clone(numeric),
                    ),
                    (
                        "classifier",
                        LogisticRegression(
                            class_weight="balanced",
                            solver="lbfgs",
                            max_iter=10000,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            "Random Forest": Pipeline(
                [
                    (
                        "preprocessor",
                        clone(tree),
                    ),
                    (
                        "classifier",
                        RandomForestClassifier(
                            class_weight=(
                                "balanced_subsample"
                            ),
                            random_state=random_state,
                            n_jobs=n_jobs,
                        ),
                    ),
                ]
            ),
            "Extra Trees": Pipeline(
                [
                    (
                        "preprocessor",
                        clone(tree),
                    ),
                    (
                        "classifier",
                        ExtraTreesClassifier(
                            class_weight="balanced",
                            random_state=random_state,
                            n_jobs=n_jobs,
                        ),
                    ),
                ]
            ),
            "Histogram Gradient Boosting": Pipeline(
                [
                    (
                        "preprocessor",
                        clone(tree),
                    ),
                    (
                        "classifier",
                        HistGradientBoostingClassifier(
                            random_state=random_state
                        ),
                    ),
                ]
            ),
        }

        self.grids = {
            "Logistic Regression": {
                "classifier__C": [
                    0.1,
                    1.0,
                    10.0,
                ],
            },
            "Random Forest": {
                "classifier__n_estimators": [
                    300,
                    600,
                ],
                "classifier__min_samples_leaf": [
                    1,
                    2,
                    5,
                ],
                "classifier__max_features": [
                    "sqrt",
                    None,
                ],
            },
            "Extra Trees": {
                "classifier__n_estimators": [
                    300,
                    600,
                ],
                "classifier__min_samples_leaf": [
                    1,
                    2,
                    5,
                ],
                "classifier__max_features": [
                    "sqrt",
                    None,
                ],
            },
            "Histogram Gradient Boosting": {
                "classifier__learning_rate": [
                    0.05,
                    0.1,
                ],
                "classifier__max_iter": [
                    200,
                    400,
                ],
                "classifier__max_leaf_nodes": [
                    15,
                    31,
                ],
                "classifier__l2_regularization": [
                    0.0,
                    1.0,
                ],
            },
        }

    def cv(self, y: pd.Series) -> StratifiedKFold:
        minimum_class_count = int(
            y.value_counts().min()
        )

        folds = min(
            self.cfg.cv_folds,
            minimum_class_count,
        )

        if folds < 2:
            raise ValueError(
                "The rarest chronological-training class "
                "needs at least two observations"
            )

        if folds < self.cfg.cv_folds:
            warnings.warn(
                f"Reducing CV folds from {self.cfg.cv_folds} "
                f"to {folds}, based on the rarest class "
                "in the chronological training partition.",
                stacklevel=2,
            )

        return StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=self.cfg.random_state,
        )

    def tune_on_training_only(self) -> None:
        self.heading(
            "7. Hyperparameter tuning on chronological "
            "training data only"
        )

        X_train, _, y_train, _ = self.split_data()
        cv = self.cv(y_train)

        for name, pipeline in self.models.items():
            print(f"Tuning {name} ...")

            search = GridSearchCV(
                estimator=pipeline,
                param_grid=self.grids[name],
                scoring=self.SCORING,
                refit=self.cfg.selection_metric,
                cv=cv,
                n_jobs=self.cfg.n_jobs,
                return_train_score=False,
                error_score="raise",
            )

            search.fit(X_train, y_train)

            slug = self.slug(name)
            results = pd.DataFrame(
                search.cv_results_
            )

            results.to_csv(
                self.path(
                    f"tuning_cv_results_{slug}.csv"
                ),
                index=False,
            )

            self.best_params[name] = (
                search.best_params_
            )

            best_index = int(search.best_index_)

            row: dict[str, Any] = {
                "model": name,
                "selection_metric": (
                    self.cfg.selection_metric
                ),
                "best_parameters": json.dumps(
                    search.best_params_,
                    sort_keys=True,
                ),
            }

            for metric in self.SCORING:
                row[f"cv_{metric}_mean"] = (
                    results.loc[
                        best_index,
                        f"mean_test_{metric}",
                    ]
                )
                row[f"cv_{metric}_std"] = (
                    results.loc[
                        best_index,
                        f"std_test_{metric}",
                    ]
                )

            row["mean_fit_time"] = results.loc[
                best_index,
                "mean_fit_time",
            ]
            row["mean_score_time"] = results.loc[
                best_index,
                "mean_score_time",
            ]

            self.search_summaries.append(row)

            selected = clone(pipeline).set_params(
                **search.best_params_
            )

            fold_scores = cross_validate(
                selected,
                X_train,
                y_train,
                cv=cv,
                scoring=self.SCORING,
                n_jobs=self.cfg.n_jobs,
                error_score="raise",
            )

            fold_rows: list[dict[str, Any]] = []

            for fold_index in range(cv.n_splits):
                fold_row: dict[str, Any] = {
                    "model": name,
                    "fold": fold_index + 1,
                    "note": (
                        "Selected-configuration tuning-CV "
                        "score; not a nested-CV estimate"
                    ),
                    "fit_time": fold_scores[
                        "fit_time"
                    ][fold_index],
                    "score_time": fold_scores[
                        "score_time"
                    ][fold_index],
                }

                for metric in self.SCORING:
                    fold_row[metric] = fold_scores[
                        f"test_{metric}"
                    ][fold_index]

                fold_rows.append(fold_row)

            pd.DataFrame(fold_rows).to_csv(
                self.path(
                    "selected_configuration_"
                    f"training_cv_{slug}.csv"
                ),
                index=False,
            )

            self.plot_search(
                results=results,
                name=name,
                best_index=best_index,
            )

    def plot_search(
        self,
        results: pd.DataFrame,
        name: str,
        best_index: int,
    ) -> None:
        metric = self.cfg.selection_metric
        rank_column = f"rank_test_{metric}"
        mean_column = f"mean_test_{metric}"
        std_column = f"std_test_{metric}"

        ordered = (
            results
            .reset_index(names="original_index")
            .sort_values(rank_column)
            .reset_index(drop=True)
        )

        selected_matches = ordered.index[
            ordered["original_index"] == best_index
        ].tolist()

        x = np.arange(len(ordered))

        plt.figure(figsize=(11, 6))

        plt.errorbar(
            x,
            ordered[mean_column],
            yerr=ordered[std_column],
            marker="o",
            capsize=3,
            label=metric.replace("_", " ").title(),
        )

        if metric != "balanced_accuracy":
            plt.errorbar(
                x,
                ordered[
                    "mean_test_balanced_accuracy"
                ],
                yerr=ordered[
                    "std_test_balanced_accuracy"
                ],
                marker="s",
                capsize=3,
                label="Balanced Accuracy",
            )

        if selected_matches:
            selected_position = selected_matches[0]

            plt.axvline(
                selected_position,
                color="darkred",
                linestyle="--",
                linewidth=1.2,
                label="Selected configuration",
            )

        plt.xlabel(
            "Hyperparameter configuration ordered by "
            f"{metric.replace('_', ' ')} rank"
        )
        plt.ylabel("Cross-validated score")
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.title(
            f"Hyperparameter Search: {name}"
        )

        self.savefig(
            f"hyperparameter_search_{self.slug(name)}.png"
        )

    def evaluate_untouched_holdout(self) -> None:
        self.heading(
            "8. Evaluating untouched chronological holdout"
        )

        X_train, X_test, y_train, y_test = (
            self.split_data()
        )

        for name, pipeline in self.models.items():
            model = (
                clone(pipeline)
                .set_params(**self.best_params[name])
                .fit(X_train, y_train)
            )

            self.evaluation_models[name] = model
            self.export_evaluation_model(name, model)

            pred = model.predict(X_test)
            proba = model.predict_proba(X_test)

            model_classes = list(
                model.named_steps[
                    "classifier"
                ].classes_
            )

            metrics = {
                "model": name,
                "accuracy": accuracy_score(
                    y_test,
                    pred,
                ),
                "balanced_accuracy": (
                    balanced_accuracy_score(
                        y_test,
                        pred,
                    )
                ),
                "macro_precision": precision_score(
                    y_test,
                    pred,
                    labels=self.classes,
                    average="macro",
                    zero_division=0,
                ),
                "macro_recall": recall_score(
                    y_test,
                    pred,
                    labels=self.classes,
                    average="macro",
                    zero_division=0,
                ),
                "macro_f1": f1_score(
                    y_test,
                    pred,
                    labels=self.classes,
                    average="macro",
                    zero_division=0,
                ),
                "weighted_f1": f1_score(
                    y_test,
                    pred,
                    labels=self.classes,
                    average="weighted",
                    zero_division=0,
                ),
                "mcc": matthews_corrcoef(
                    y_test,
                    pred,
                ),
            }

            try:
                metrics["log_loss"] = log_loss(
                    y_test,
                    proba,
                    labels=model_classes,
                )
            except ValueError:
                metrics["log_loss"] = np.nan

            self.holdout_rows.append(metrics)

            slug = self.slug(name)

            report = classification_report(
                y_test,
                pred,
                labels=self.classes,
                output_dict=True,
                zero_division=0,
            )

            pd.DataFrame(report).T.to_csv(
                self.path(
                    f"holdout_report_{slug}.csv"
                ),
                index_label="class_or_average",
            )

            if self.test_indices is None:
                raise RuntimeError(
                    "Chronological test indices are missing"
                )

            out = self.data.iloc[
                self.test_indices
            ][
                [
                    "Sequence",
                    *self.cfg.features,
                    self.cfg.target,
                ]
            ].copy()

            out["Predicted_Y"] = pred
            out["Correct"] = (
                out[self.cfg.target].to_numpy()
                == pred
            )

            sorted_proba = np.sort(
                proba,
                axis=1,
            )

            out["Prediction_Confidence"] = (
                sorted_proba[:, -1]
            )

            if sorted_proba.shape[1] >= 2:
                out["Prediction_Margin"] = (
                    sorted_proba[:, -1]
                    - sorted_proba[:, -2]
                )
            else:
                out["Prediction_Margin"] = np.nan

            for column_index, label in enumerate(
                model_classes
            ):
                out[f"Probability_{label}"] = (
                    proba[:, column_index]
                )

            out.to_csv(
                self.path(
                    f"holdout_predictions_{slug}.csv"
                ),
                index=False,
            )

            out.loc[~out["Correct"]].to_csv(
                self.path(
                    "holdout_misclassifications_"
                    f"{slug}.csv"
                ),
                index=False,
            )

            self.export_confusion_matrices(
                y_true=y_test,
                pred=pred,
                name=name,
            )

            self.plot_prediction_diagnostics(
                out=out,
                name=name,
            )

            self.holdout_importance(
                model=model,
                X_test=X_test,
                y_test=y_test,
                name=name,
            )

        holdout = (
            pd.DataFrame(self.holdout_rows)
            .sort_values(
                self.cfg.selection_metric,
                ascending=False,
            )
        )

        holdout.to_csv(
            self.path(
                "chronological_holdout_metrics.csv"
            ),
            index=False,
        )

        cv_summary = pd.DataFrame(
            self.search_summaries
        )

        comparison = cv_summary.merge(
            holdout,
            on="model",
            suffixes=("_cv", "_holdout"),
        )

        holdout_metric_column = (
            self.cfg.selection_metric
        )
        cv_metric_column = (
            f"cv_{self.cfg.selection_metric}_mean"
        )

        comparison[
            f"cv_holdout_{self.cfg.selection_metric}_gap"
        ] = (
            comparison[cv_metric_column]
            - comparison[holdout_metric_column]
        )

        comparison.sort_values(
            cv_metric_column,
            ascending=False,
        ).to_csv(
            self.path("model_comparison.csv"),
            index=False,
        )

        self.plot_model_comparison(comparison)

    def export_evaluation_model(
        self,
        name: str,
        model: Pipeline,
    ) -> None:
        classifier = model.named_steps["classifier"]

        artifact = {
            "model": model,
            "model_type": name,
            "model_role": (
                "chronological_training_evaluation_model"
            ),
            "features": list(self.cfg.features),
            "reporting_class_order": self.classes,
            "estimator_class_order": list(
                classifier.classes_
            ),
            "selection_metric": (
                self.cfg.selection_metric
            ),
            "best_parameters": (
                self.best_params[name]
            ),
        }

        joblib.dump(
            artifact,
            self.path(
                "evaluation_"
                f"{self.slug(name)}.joblib"
            ),
            compress=3,
        )

    def export_confusion_matrices(
        self,
        y_true: pd.Series,
        pred: np.ndarray,
        name: str,
    ) -> None:
        slug = self.slug(name)

        matrix = confusion_matrix(
            y_true,
            pred,
            labels=self.classes,
        )

        normalized = confusion_matrix(
            y_true,
            pred,
            labels=self.classes,
            normalize="true",
        )

        index = [
            f"Actual_{label}"
            for label in self.classes
        ]
        columns = [
            f"Predicted_{label}"
            for label in self.classes
        ]

        pd.DataFrame(
            matrix,
            index=index,
            columns=columns,
        ).to_csv(
            self.path(
                "holdout_confusion_matrix_"
                f"{slug}.csv"
            )
        )

        pd.DataFrame(
            normalized,
            index=index,
            columns=columns,
        ).to_csv(
            self.path(
                "holdout_confusion_matrix_"
                f"normalized_{slug}.csv"
            )
        )

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(13, 5),
        )

        ConfusionMatrixDisplay.from_predictions(
            y_true,
            pred,
            labels=self.classes,
            cmap="Blues",
            values_format="d",
            colorbar=False,
            ax=axes[0],
        )

        axes[0].set_title(
            f"{name}\nCounts"
        )

        ConfusionMatrixDisplay.from_predictions(
            y_true,
            pred,
            labels=self.classes,
            cmap="Blues",
            normalize="true",
            values_format=".2f",
            colorbar=False,
            ax=axes[1],
        )

        axes[1].set_title(
            f"{name}\nRecall-Normalized"
        )

        self.savefig(
            f"holdout_confusion_matrix_{slug}.png"
        )

    def plot_prediction_diagnostics(
        self,
        out: pd.DataFrame,
        name: str,
    ) -> None:
        slug = self.slug(name)

        plt.figure(figsize=(10, 5))

        correct = out.loc[
            out["Correct"],
            "Prediction_Confidence",
        ]
        incorrect = out.loc[
            ~out["Correct"],
            "Prediction_Confidence",
        ]

        plt.hist(
            correct,
            bins=20,
            alpha=0.7,
            label="Correct",
        )

        if not incorrect.empty:
            plt.hist(
                incorrect,
                bins=20,
                alpha=0.7,
                label="Incorrect",
            )

        plt.xlabel(
            "Maximum predicted probability"
        )
        plt.ylabel("Observations")
        plt.title(
            f"Prediction Confidence: {name}"
        )
        plt.grid(alpha=0.3)
        plt.legend()

        self.savefig(
            "holdout_prediction_confidence_"
            f"{slug}.png"
        )

        plt.figure(figsize=(12, 4))

        plt.scatter(
            out["Sequence"],
            out["Prediction_Confidence"],
            c=np.where(
                out["Correct"],
                "steelblue",
                "red",
            ),
            s=18,
        )

        plt.xlabel("Observation Sequence")
        plt.ylabel("Prediction Confidence")
        plt.title(
            f"Holdout Predictions by Sequence: {name}"
        )
        plt.grid(alpha=0.3)

        self.savefig(
            f"holdout_errors_by_sequence_{slug}.png"
        )

    def holdout_importance(
        self,
        model: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        name: str,
    ) -> None:
        result = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="f1_macro",
            n_repeats=self.cfg.permutation_repeats,
            random_state=self.cfg.random_state,
            n_jobs=self.cfg.n_jobs,
        )

        importance = pd.DataFrame(
            {
                "model": name,
                "feature": self.cfg.features,
                "importance_mean": (
                    result.importances_mean
                ),
                "importance_std": (
                    result.importances_std
                ),
                "interpretation_partition": (
                    "chronological_holdout"
                ),
            }
        ).sort_values(
            "importance_mean",
            ascending=False,
        )

        slug = self.slug(name)

        importance.to_csv(
            self.path(
                "holdout_permutation_importance_"
                f"{slug}.csv"
            ),
            index=False,
        )

        plot_data = importance.sort_values(
            "importance_mean"
        )

        plt.figure(figsize=(9, 6))

        plt.barh(
            plot_data["feature"],
            plot_data["importance_mean"],
            xerr=plot_data["importance_std"],
            color="steelblue",
            alpha=0.85,
        )

        plt.axvline(
            0,
            color="black",
            linewidth=0.8,
        )

        plt.xlabel(
            "Decrease in holdout macro F1 "
            "after permutation"
        )
        plt.title(
            f"Holdout Permutation Importance: {name}"
        )
        plt.grid(axis="x", alpha=0.3)

        self.savefig(
            "holdout_permutation_importance_"
            f"{slug}.png"
        )

    def plot_model_comparison(
        self,
        comparison: pd.DataFrame,
    ) -> None:
        metric = self.cfg.selection_metric
        cv_column = f"cv_{metric}_mean"
        cv_std_column = f"cv_{metric}_std"
        holdout_column = metric

        plot = comparison.sort_values(cv_column)

        y = np.arange(len(plot))
        height = 0.35

        fig, ax = plt.subplots(figsize=(11, 6))

        ax.barh(
            y - height / 2,
            plot[cv_column],
            height,
            xerr=plot[cv_std_column],
            label=(
                "Training-partition tuning CV "
                f"{metric.replace('_', ' ').title()}"
            ),
            color="steelblue",
        )

        ax.barh(
            y + height / 2,
            plot[holdout_column],
            height,
            label=(
                "Chronological holdout "
                f"{metric.replace('_', ' ').title()}"
            ),
            color="darkorange",
        )

        ax.set_yticks(
            y,
            labels=plot["model"],
        )
        ax.set_xlim(0, 1)
        ax.set_xlabel("Score")
        ax.set_title(
            "Model Comparison"
        )
        ax.grid(axis="x", alpha=0.3)
        ax.legend()

        self.savefig("model_comparison.png")

    def export_logistic_parameters(
        self,
        pipeline: Pipeline,
        prefix: str,
        model_role: str,
        create_plot: bool = False,
    ) -> None:
        classifier = pipeline.named_steps[
            "classifier"
        ]

        numeric = pipeline.named_steps[
            "preprocessor"
        ].named_transformers_["numeric"]

        imputer = numeric.named_steps["imputer"]
        scaler = numeric.named_steps["scaler"]

        preprocessing = pd.DataFrame(
            {
                "feature": self.cfg.features,
                "imputation_median": (
                    imputer.statistics_
                ),
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
            }
        )

        preprocessing.to_csv(
            self.path(
                f"{prefix}_logistic_"
                "preprocessing_parameters.csv"
            ),
            index=False,
        )

        classes = list(classifier.classes_)
        coefficients = classifier.coef_
        intercepts = classifier.intercept_

        if len(classes) == 2:
            equation_labels = [
                f"{classes[1]} relative to {classes[0]}"
            ]
        else:
            equation_labels = classes

        standardized_rows: list[
            dict[str, Any]
        ] = []

        original_rows: list[
            dict[str, Any]
        ] = []

        for index, label in enumerate(
            equation_labels
        ):
            standardized_rows.append(
                {
                    "equation_for": label,
                    "beta_0": intercepts[index],
                    **dict(
                        zip(
                            self.cfg.features,
                            coefficients[index],
                        )
                    ),
                }
            )

            beta_original = (
                coefficients[index]
                / scaler.scale_
            )

            beta0_original = (
                intercepts[index]
                - np.sum(
                    coefficients[index]
                    * scaler.mean_
                    / scaler.scale_
                )
            )

            original_rows.append(
                {
                    "equation_for": label,
                    "beta_0_original_units": (
                        beta0_original
                    ),
                    **dict(
                        zip(
                            self.cfg.features,
                            beta_original,
                        )
                    ),
                }
            )

        standardized_frame = pd.DataFrame(
            standardized_rows
        )

        standardized_frame.to_csv(
            self.path(
                f"{prefix}_logistic_"
                "coefficients_and_intercepts.csv"
            ),
            index=False,
        )

        pd.DataFrame(original_rows).to_csv(
            self.path(
                f"{prefix}_logistic_"
                "coefficients_original_units.csv"
            ),
            index=False,
        )

        usage = {
            "source_model": model_role,
            "classes": classes,
            "reporting_class_order": self.classes,
            "estimator_class_order": classes,
            "binary": len(classes) == 2,
            "standardized_equation": (
                "eta_k = beta_0_k + "
                "sum(beta_kj * z_j)"
            ),
            "standardization": (
                "z_j = (imputed_x_j - "
                "scaler_mean_j) / scaler_scale_j"
            ),
            "original_units_equation": (
                "eta_k = beta_0_original_k + "
                "sum(beta_original_kj * imputed_x_j)"
            ),
            "missing_value_rule": (
                "Before using either equation, replace "
                "each missing predictor with its exported "
                "imputation median."
            ),
            "binary_probability": (
                "P(classes[1]) = sigmoid(eta); "
                "P(classes[0]) = 1 - sigmoid(eta)"
            ),
            "multiclass_probability": (
                "P(k) = exp(eta_k - max(eta)) / "
                "sum_l exp(eta_l - max(eta))"
            ),
            "recommended_usage": (
                "Use the exported joblib pipeline to "
                "preserve imputation, feature order, "
                "scaling, estimator class order, and "
                "probability-column order."
            ),
        }

        with self.path(
            f"{prefix}_logistic_equation_usage.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                usage,
                file,
                indent=2,
                ensure_ascii=False,
            )

        if create_plot:
            self.plot_logistic_coefficients(
                standardized_frame.set_index(
                    "equation_for"
                ),
                prefix=prefix,
            )

    def export_evaluation_logistic_outputs(
        self,
    ) -> None:
        self.heading(
            "9. Evaluation logistic equations "
            "and preprocessing"
        )

        pipeline = self.evaluation_models[
            "Logistic Regression"
        ]

        self.export_logistic_parameters(
            pipeline=pipeline,
            prefix="evaluation",
            model_role=(
                "chronological_training_evaluation_model"
            ),
            create_plot=True,
        )

    def plot_logistic_coefficients(
        self,
        frame: pd.DataFrame,
        prefix: str,
    ) -> None:
        coefficients = frame.drop(
            columns="beta_0"
        )

        if len(coefficients) == 1:
            values = (
                coefficients.iloc[0]
                .sort_values()
            )

            colors = [
                "darkorange"
                if value < 0
                else "steelblue"
                for value in values
            ]

            plt.figure(figsize=(9, 6))

            plt.barh(
                values.index,
                values.values,
                color=colors,
            )

            plt.axvline(
                0,
                color="black",
                linewidth=0.8,
            )

            plt.xlabel(
                "Standardized coefficient"
            )

            plt.title(
                "Logistic Coefficients: "
                f"{coefficients.index[0]}"
            )
        else:
            limit = max(
                float(
                    np.abs(
                        coefficients.to_numpy()
                    ).max()
                ),
                1e-12,
            )

            fig, ax = plt.subplots(
                figsize=(
                    11,
                    max(
                        5,
                        0.8 * len(coefficients),
                    ),
                )
            )

            image = ax.imshow(
                coefficients,
                cmap="coolwarm",
                aspect="auto",
                vmin=-limit,
                vmax=limit,
            )

            ax.set_xticks(
                range(len(coefficients.columns)),
                labels=coefficients.columns,
                rotation=45,
                ha="right",
            )

            ax.set_yticks(
                range(len(coefficients.index)),
                labels=coefficients.index,
            )

            fig.colorbar(
                image,
                ax=ax,
                label="Standardized coefficient",
            )

            ax.set_title(
                "Multiclass Logistic Coefficients"
            )

        self.savefig(
            f"{prefix}_logistic_coefficients.png"
        )

    def refit_and_export_production_models(
        self,
    ) -> None:
        self.heading(
            "10. Refitting production models on all data"
        )

        X = self.data[list(self.cfg.features)]
        y = self.data[self.cfg.target]

        cv_summary = pd.DataFrame(
            self.search_summaries
        )

        overall_name = str(
            cv_summary.sort_values(
                self.selection_mean_column,
                ascending=False,
            ).iloc[0]["model"]
        )

        manifest: list[dict[str, Any]] = []

        for name, pipeline in self.models.items():
            production = (
                clone(pipeline)
                .set_params(**self.best_params[name])
                .fit(X, y)
            )

            self.production_models[name] = (
                production
            )

            filename = (
                f"production_{self.slug(name)}.joblib"
            )

            estimator_classes = list(
                production.named_steps[
                    "classifier"
                ].classes_
            )

            artifact = {
                "model": production,
                "model_type": name,
                "model_role": (
                    "production_full_data_refit"
                ),
                "features": list(self.cfg.features),
                "reporting_class_order": self.classes,
                "estimator_class_order": (
                    estimator_classes
                ),
                "selection_metric": (
                    self.cfg.selection_metric
                ),
                "best_parameters": (
                    self.best_params[name]
                ),
            }

            joblib.dump(
                artifact,
                self.path(filename),
                compress=3,
            )

            cv_row = cv_summary.loc[
                cv_summary["model"] == name
            ].iloc[0]

            manifest.append(
                {
                    "model_type": name,
                    "filename": filename,
                    "model_role": (
                        "production_full_data_refit"
                    ),
                    "selection_metric": (
                        self.cfg.selection_metric
                    ),
                    "selection_score_mean": (
                        cv_row[
                            self.selection_mean_column
                        ]
                    ),
                    "selection_score_std": (
                        cv_row[
                            self.selection_std_column
                        ]
                    ),
                    "reporting_class_order": (
                        json.dumps(self.classes)
                    ),
                    "estimator_class_order": (
                        json.dumps(estimator_classes)
                    ),
                    "feature_count": len(
                        self.cfg.features
                    ),
                    "best_parameters": json.dumps(
                        self.best_params[name],
                        sort_keys=True,
                    ),
                }
            )

        overall_model = self.production_models[
            overall_name
        ]

        overall_classes = list(
            overall_model.named_steps[
                "classifier"
            ].classes_
        )

        overall_artifact = {
            "model": overall_model,
            "model_type": overall_name,
            "model_role": (
                "best_overall_production_"
                "full_data_refit"
            ),
            "features": list(self.cfg.features),
            "reporting_class_order": self.classes,
            "estimator_class_order": (
                overall_classes
            ),
            "selection_metric": (
                self.cfg.selection_metric
            ),
            "best_parameters": (
                self.best_params[overall_name]
            ),
        }

        joblib.dump(
            overall_artifact,
            self.path("best_overall_model.joblib"),
            compress=3,
        )

        pd.DataFrame(manifest).to_csv(
            self.path("model_export_manifest.csv"),
            index=False,
        )

        self.export_logistic_parameters(
            pipeline=self.production_models[
                "Logistic Regression"
            ],
            prefix="production",
            model_role=(
                "production_full_data_refit"
            ),
            create_plot=True,
        )

    def metadata(self) -> None:
        configuration = asdict(self.cfg)

        configuration["data"] = str(
            self.cfg.data.resolve()
        )
        configuration["output"] = str(
            self.cfg.output.resolve()
        )
        configuration["features"] = list(
            self.cfg.features
        )
        configuration["class_order"] = (
            list(self.cfg.class_order)
            if self.cfg.class_order
            else None
        )

        payload = {
            "configuration": configuration,
            "observed_reporting_class_order": (
                self.classes
            ),
            "validation_design": (
                "A chronological holdout is created "
                "before tuning. Grid search and all "
                "preprocessing during tuning use only "
                "the chronological training partition."
            ),
            "cv_interpretation": (
                "Reported training-partition CV scores "
                "are hyperparameter-tuning scores and "
                "selected-configuration CV scores. They "
                "are not nested-CV generalization estimates."
            ),
            "final_evaluation": (
                "The chronological holdout is used for "
                "final evaluation after hyperparameter "
                "selection."
            ),
            "holdout_interpretation": (
                "Permutation importance is calculated "
                "post-evaluation on the chronological "
                "holdout and should be treated as holdout "
                "interpretation, not independent training data."
            ),
            "production_design": (
                "Selected configurations are refitted on "
                "all available data only after holdout "
                "evaluation."
            ),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        }

        with self.path(
            "run_metadata.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                indent=2,
                ensure_ascii=False,
            )

    def write_generated_file_manifest(
        self,
    ) -> None:
        manifest_name = "generated_files.txt"

        files = sorted(
            self.generated_files | {manifest_name}
        )

        manifest_path = (
            self.cfg.output / manifest_name
        )

        manifest_path.write_text(
            "\n".join(files) + "\n",
            encoding="utf-8",
        )

        self.generated_files.add(manifest_name)

    def run(self) -> None:
        self.load()
        self.audit_and_describe()
        self.correlations()
        self.exploratory_plots()
        self.create_chronological_split()
        self.build_models()
        self.tune_on_training_only()
        self.evaluate_untouched_holdout()
        self.export_evaluation_logistic_outputs()
        self.refit_and_export_production_models()
        self.metadata()
        self.write_generated_file_manifest()

        self.heading("Analysis completed")

        print(
            f"Results saved to: "
            f"{self.cfg.output.resolve()}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "viClassify: binary or multiclass "
            "tabular classification"
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Input CSV path",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("viclassify_results"),
        help="Output directory",
    )

    parser.add_argument(
        "--target",
        default="Y",
        help="Target-column name",
    )

    parser.add_argument(
        "--features",
        nargs="+",
        default=[
            f"X{index}"
            for index in range(1, 8)
        ],
        help="Ordered predictor-column names",
    )

    parser.add_argument(
        "--class-order",
        nargs="+",
        default=None,
        help="Optional reporting order for class labels",
    )

    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.20,
        help=(
            "Fraction of final chronological observations "
            "reserved for holdout evaluation"
        ),
    )

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Maximum number of stratified CV folds",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="-1 for all processors or a positive integer",
    )

    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
    )

    parser.add_argument(
        "--near-constant-threshold",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "--selection-metric",
        choices=[
            "accuracy",
            "balanced_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_f1",
        ],
        default="macro_f1",
        help=(
            "Metric used for hyperparameter selection "
            "and overall production-model selection"
        ),
    )

    args = parser.parse_args()

    args.target = args.target.strip()
    args.features = [
        feature.strip()
        for feature in args.features
    ]

    if args.class_order:
        args.class_order = [
            label.strip()
            for label in args.class_order
        ]

    if not args.target:
        parser.error(
            "--target must not be empty"
        )

    if not args.features:
        parser.error(
            "--features must contain at least one feature"
        )

    if any(
        not feature
        for feature in args.features
    ):
        parser.error(
            "--features must not contain empty names"
        )

    if (
        len(set(args.features))
        != len(args.features)
    ):
        parser.error(
            "--features must not contain duplicate "
            "column names"
        )

    if args.target in args.features:
        parser.error(
            "--target must not also appear in --features; "
            "this would cause target leakage"
        )

    if args.class_order:
        if any(
            not label
            for label in args.class_order
        ):
            parser.error(
                "--class-order must not contain "
                "empty labels"
            )

        if (
            len(set(args.class_order))
            != len(args.class_order)
        ):
            parser.error(
                "--class-order must not contain "
                "duplicate labels"
            )

    if not 0 < args.test_fraction < 1:
        parser.error(
            "--test-fraction must be between 0 and 1"
        )

    if args.cv_folds < 2:
        parser.error(
            "--cv-folds must be at least 2"
        )

    if (
        args.n_jobs == 0
        or args.n_jobs < -1
    ):
        parser.error(
            "--n-jobs must be -1 or a positive integer"
        )

    if args.permutation_repeats < 1:
        parser.error(
            "--permutation-repeats must be positive"
        )

    if args.dpi < 50:
        parser.error(
            "--dpi must be at least 50"
        )

    if not (
        0
        < args.near_constant_threshold
        <= 1
    ):
        parser.error(
            "--near-constant-threshold must be in (0, 1]"
        )

    return args


def main() -> None:
    args = parse_args()

    cfg = Config(
        data=args.data,
        output=args.output,
        target=args.target,
        features=tuple(args.features),
        class_order=(
            tuple(args.class_order)
            if args.class_order
            else None
        ),
        test_fraction=args.test_fraction,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        permutation_repeats=(
            args.permutation_repeats
        ),
        dpi=args.dpi,
        near_constant_threshold=(
            args.near_constant_threshold
        ),
        selection_metric=(
            args.selection_metric
        ),
    )

    ClassificationAnalysis(cfg).run()


if __name__ == "__main__":
    main()
