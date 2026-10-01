from __future__ import annotations

import argparse
import json
import platform
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
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
                             classification_report, f1_score, log_loss, matthews_corrcoef,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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
    selection_metric: str = "f1_macro"


class ClassificationAnalysis:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.cfg.output.mkdir(parents=True, exist_ok=True)
        self.df: pd.DataFrame | None = None
        self.classes: list[str] = []
        self.models: dict[str, Pipeline] = {}
        self.grids: dict[str, dict[str, list[Any]]] = {}
        self.best_estimators: dict[str, Pipeline] = {}
        self.cv_summary: pd.DataFrame | None = None

    @property
    def data(self) -> pd.DataFrame:
        if self.df is None:
            raise RuntimeError("Data not loaded")
        return self.df

    def path(self, name: str) -> Path:
        return self.cfg.output / name

    @staticmethod
    def slug(name: str) -> str:
        return name.lower().replace(" ", "_")

    def savefig(self, name: str) -> None:
        plt.tight_layout()
        plt.savefig(self.path(name), dpi=self.cfg.dpi, bbox_inches="tight")
        plt.close()

    def load(self) -> None:
        if not self.cfg.data.exists():
            raise FileNotFoundError(self.cfg.data.resolve())
        df = pd.read_csv(self.cfg.data, skipinitialspace=True)
        df.columns = df.columns.astype(str).str.strip()
        required = [*self.cfg.features, self.cfg.target]
        missing = sorted(set(required) - set(df.columns))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if df.empty:
            raise ValueError("Dataset is empty")
        df = df.copy()
        df.insert(0, "Sequence", np.arange(1, len(df) + 1))
        for col in self.cfg.features:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        y = df[self.cfg.target].astype("string").str.strip()
        if y.isna().any() or y.eq("").any():
            raise ValueError("Target contains missing or empty labels")
        df[self.cfg.target] = y
        observed = sorted(y.unique().tolist())
        if len(observed) < 2:
            raise ValueError("At least two target classes are required")
        if self.cfg.class_order:
            unknown = sorted(set(self.cfg.class_order) - set(observed))
            if unknown:
                warnings.warn(f"Requested class labels not observed: {unknown}")
            self.classes = [c for c in self.cfg.class_order if c in observed]
            self.classes += [c for c in observed if c not in self.classes]
        else:
            self.classes = observed
        self.df = df
        print(f"Rows: {len(df):,}; features: {len(self.cfg.features)}; classes: {self.classes}")

    def audit_and_describe(self) -> None:
        df = self.data
        missing = df.isna().sum().to_frame("missing_count")
        missing["missing_percentage"] = 100 * missing["missing_count"] / len(df)
        missing.to_csv(self.path("missing_values.csv"), index_label="column")
        dup_cols = [*self.cfg.features, self.cfg.target]
        df.loc[df.duplicated(dup_cols, keep=False)].to_csv(self.path("duplicate_rows.csv"), index=False)
        counts = df[self.cfg.target].value_counts().reindex(self.classes, fill_value=0)
        pd.DataFrame({"count": counts, "percentage": 100 * counts / len(df)}).to_csv(
            self.path("class_distribution.csv"), index_label="class")
        summary = df[list(self.cfg.features)].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
        summary["missing"] = df[list(self.cfg.features)].isna().sum()
        summary["unique_values"] = df[list(self.cfg.features)].nunique()
        summary["skewness"] = df[list(self.cfg.features)].skew()
        summary["kurtosis"] = df[list(self.cfg.features)].kurtosis()
        summary.to_csv(self.path("numerical_summary.csv"), index_label="feature")

    def correlations(self) -> None:
        x = self.data[list(self.cfg.features)]
        for method in ("pearson", "spearman"):
            corr = x.corr(method=method)
            corr.to_csv(self.path(f"{method}_correlation_matrix.csv"))
            fig, ax = plt.subplots(figsize=(9, 8))
            im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr)), labels=corr.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(corr)), labels=corr.index)
            for i in range(len(corr)):
                for j in range(len(corr)):
                    value = corr.iloc[i, j]
                    ax.text(j, i, "NA" if pd.isna(value) else f"{value:.2f}", ha="center", va="center",
                            color="white" if pd.notna(value) and abs(value) > .65 else "black")
            fig.colorbar(im, ax=ax, label=f"{method.title()} correlation")
            ax.set_title(f"{method.title()} Correlation Matrix")
            self.savefig(f"{method}_correlation_matrix.png")

    def build_models(self) -> None:
        numeric = ColumnTransformer([("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), list(self.cfg.features))])
        tree = ColumnTransformer([("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median"))]), list(self.cfg.features))])
        rs, jobs = self.cfg.random_state, self.cfg.n_jobs
        self.models = {
            "Logistic Regression": Pipeline([("preprocessor", clone(numeric)), ("classifier", LogisticRegression(
                class_weight="balanced", solver="lbfgs", max_iter=10000, random_state=rs))]),
            "Random Forest": Pipeline([("preprocessor", clone(tree)), ("classifier", RandomForestClassifier(
                class_weight="balanced_subsample", random_state=rs, n_jobs=jobs))]),
            "Extra Trees": Pipeline([("preprocessor", clone(tree)), ("classifier", ExtraTreesClassifier(
                class_weight="balanced", random_state=rs, n_jobs=jobs))]),
            "Histogram Gradient Boosting": Pipeline([("preprocessor", clone(tree)),
                ("classifier", HistGradientBoostingClassifier(random_state=rs))]),
        }
        self.grids = {
            "Logistic Regression": {"classifier__C": [0.1, 1.0, 10.0]},
            "Random Forest": {"classifier__n_estimators": [300, 600], "classifier__min_samples_leaf": [1, 2, 5],
                              "classifier__max_features": ["sqrt", None]},
            "Extra Trees": {"classifier__n_estimators": [300, 600], "classifier__min_samples_leaf": [1, 2, 5],
                            "classifier__max_features": ["sqrt", None]},
            "Histogram Gradient Boosting": {"classifier__learning_rate": [0.05, 0.1],
                "classifier__max_iter": [200, 400], "classifier__max_leaf_nodes": [15, 31],
                "classifier__l2_regularization": [0.0, 1.0]},
        }

    def cv(self) -> StratifiedKFold:
        minimum = int(self.data[self.cfg.target].value_counts().min())
        folds = min(self.cfg.cv_folds, minimum)
        if folds < 2:
            raise ValueError("The rarest class needs at least two observations")
        return StratifiedKFold(folds, shuffle=True, random_state=self.cfg.random_state)

    def tune_and_export_each_type(self) -> None:
        X, y = self.data[list(self.cfg.features)], self.data[self.cfg.target]
        cv = self.cv()
        rows: list[dict[str, Any]] = []
        for name, pipeline in self.models.items():
            print(f"Tuning {name} ...")
            search = GridSearchCV(pipeline, self.grids[name], scoring=self.cfg.selection_metric, cv=cv,
                                  n_jobs=self.cfg.n_jobs, refit=True, return_train_score=False, error_score="raise")
            search.fit(X, y)
            best = search.best_estimator_
            self.best_estimators[name] = best
            slug = self.slug(name)
            pd.DataFrame(search.cv_results_).to_csv(self.path(f"cv_results_{slug}.csv"), index=False)
            artifact = {"model": best, "model_type": name, "features": list(self.cfg.features),
                        "classes": list(best.named_steps["classifier"].classes_),
                        "selection_metric": self.cfg.selection_metric, "best_cv_score": float(search.best_score_),
                        "best_parameters": search.best_params_}
            joblib.dump(artifact, self.path(f"best_{slug}.joblib"), compress=3)
            rows.append({"model": name, "best_cv_macro_f1": search.best_score_,
                         "best_parameters": json.dumps(search.best_params_, sort_keys=True)})
        summary = pd.DataFrame(rows).sort_values("best_cv_macro_f1", ascending=False)
        summary.to_csv(self.path("model_comparison.csv"), index=False)
        self.cv_summary = summary
        overall_name = str(summary.iloc[0]["model"])
        joblib.dump({"model": self.best_estimators[overall_name], "model_type": overall_name,
                     "features": list(self.cfg.features), "classes": self.classes},
                    self.path("best_overall_model.joblib"), compress=3)
        print(summary.to_string(index=False))

    def chronological_holdout(self) -> None:
        n = len(self.data)
        split = int(np.floor(n * (1 - self.cfg.test_fraction)))
        if not 0 < split < n:
            raise ValueError("test-fraction creates an empty partition")
        X = self.data[list(self.cfg.features)]
        y = self.data[self.cfg.target]
        Xtr, Xte, ytr, yte = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
        missing_train = sorted(set(self.classes) - set(ytr))
        if missing_train:
            raise ValueError(f"Chronological training partition lacks classes: {missing_train}")
        rows = []
        for name, tuned_full_model in self.best_estimators.items():
            model = clone(tuned_full_model).fit(Xtr, ytr)
            pred = model.predict(Xte)
            row = {"model": name, "accuracy": accuracy_score(yte, pred),
                   "balanced_accuracy": balanced_accuracy_score(yte, pred),
                   "macro_precision": precision_score(yte, pred, labels=self.classes, average="macro", zero_division=0),
                   "macro_recall": recall_score(yte, pred, labels=self.classes, average="macro", zero_division=0),
                   "macro_f1": f1_score(yte, pred, labels=self.classes, average="macro", zero_division=0),
                   "weighted_f1": f1_score(yte, pred, labels=self.classes, average="weighted", zero_division=0),
                   "mcc": matthews_corrcoef(yte, pred)}
            proba = model.predict_proba(Xte)
            try: row["log_loss"] = log_loss(yte, proba, labels=model.named_steps["classifier"].classes_)
            except ValueError: row["log_loss"] = np.nan
            rows.append(row)
            slug = self.slug(name)
            pd.DataFrame(classification_report(yte, pred, labels=self.classes, output_dict=True,
                                               zero_division=0)).T.to_csv(self.path(f"holdout_report_{slug}.csv"))
            out = self.data.iloc[split:][["Sequence", *self.cfg.features, self.cfg.target]].copy()
            out["Predicted_Y"] = pred
            for j, label in enumerate(model.named_steps["classifier"].classes_):
                out[f"Probability_{label}"] = proba[:, j]
            out.to_csv(self.path(f"holdout_predictions_{slug}.csv"), index=False)
            ConfusionMatrixDisplay.from_predictions(yte, pred, labels=self.classes, cmap="Blues")
            plt.title(f"{name}: chronological holdout")
            self.savefig(f"holdout_confusion_matrix_{slug}.png")
        pd.DataFrame(rows).sort_values("macro_f1", ascending=False).to_csv(
            self.path("chronological_holdout_metrics.csv"), index=False)

    def export_logistic_equations(self) -> None:
        pipeline = self.best_estimators["Logistic Regression"]
        clf = pipeline.named_steps["classifier"]
        classes = list(clf.classes_)
        coef, intercept = clf.coef_, clf.intercept_
        rows: list[dict[str, Any]] = []
        if len(classes) == 2:
            rows.append({"equation_for": f"{classes[1]} relative to {classes[0]}", "beta_0": intercept[0],
                         **dict(zip(self.cfg.features, coef[0]))})
            usage = {"model": "binary logistic regression", "classes": classes,
                     "equation": "eta = beta_0 + sum(beta_j * z_j)",
                     "probability": f"P({classes[1]}) = 1 / (1 + exp(-eta)); P({classes[0]}) = 1 - P({classes[1]})",
                     "z_definition": "z_j is produced by the saved pipeline: median imputation followed by StandardScaler",
                     "beta_0": float(intercept[0])}
        else:
            for i, label in enumerate(classes):
                rows.append({"equation_for": label, "beta_0": intercept[i], **dict(zip(self.cfg.features, coef[i]))})
            usage = {"model": "multinomial logistic regression", "classes": classes,
                     "equation": "eta_k = beta_0_k + sum(beta_kj * z_j)",
                     "probability": "P(class k) = exp(eta_k) / sum_l exp(eta_l) (use a numerically stable softmax)",
                     "z_definition": "z_j is produced by the saved pipeline: median imputation followed by StandardScaler",
                     "beta_0_by_class": dict(zip(classes, map(float, intercept)))}
        pd.DataFrame(rows).to_csv(self.path("logistic_coefficients_and_intercepts.csv"), index=False)
        with self.path("logistic_equation_usage.json").open("w", encoding="utf-8") as f:
            json.dump(usage, f, indent=2, ensure_ascii=False)
        print("Logistic intercept(s):", dict(zip([r["equation_for"] for r in rows], map(float, intercept))))

    def importance(self) -> None:
        if self.cv_summary is None:
            return
        best_name = str(self.cv_summary.iloc[0]["model"])
        model = self.best_estimators[best_name]
        X, y = self.data[list(self.cfg.features)], self.data[self.cfg.target]
        result = permutation_importance(model, X, y, scoring=self.cfg.selection_metric,
                                        n_repeats=self.cfg.permutation_repeats,
                                        random_state=self.cfg.random_state, n_jobs=self.cfg.n_jobs)
        importance_df = pd.DataFrame({"feature": self.cfg.features,
                                      "importance_mean": result.importances_mean,
                                      "importance_std": result.importances_std})
        importance_df.insert(0, "model", best_name)
        importance_df.sort_values("importance_mean", ascending=False).to_csv(
            self.path("permutation_importance_full_data.csv"), index=False)

    def metadata(self) -> None:
        configuration = asdict(self.cfg)
        configuration["data"] = str(self.cfg.data.resolve())
        configuration["output"] = str(self.cfg.output.resolve())
        configuration["features"] = list(self.cfg.features)
        configuration["class_order"] = (
            list(self.cfg.class_order) if self.cfg.class_order else None
        )
        payload = {"configuration": configuration,
                   "classes": self.classes, "python": platform.python_version(), "numpy": np.__version__,
                   "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}
        with self.path("run_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def run(self) -> None:
        self.load(); self.audit_and_describe(); self.correlations(); self.build_models()
        self.tune_and_export_each_type(); self.chronological_holdout()
        self.export_logistic_equations(); self.importance(); self.metadata()
        print(f"Results saved to: {self.cfg.output.resolve()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Binary or multiclass tabular classification analysis")
    p.add_argument("--data", type=Path, required=True, help="Input CSV path")
    p.add_argument("--output", type=Path, default=Path("classification_results"))
    p.add_argument("--target", default="Y")
    p.add_argument("--features", nargs="+", default=[f"X{i}" for i in range(1, 8)])
    p.add_argument("--class-order", nargs="+", default=None)
    p.add_argument("--test-fraction", type=float, default=.20)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--permutation-repeats", type=int, default=20)
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()
    if not 0 < args.test_fraction < 1: p.error("--test-fraction must be between 0 and 1")
    if args.cv_folds < 2: p.error("--cv-folds must be at least 2")
    if args.permutation_repeats < 1: p.error("--permutation-repeats must be positive")
    if args.dpi < 50: p.error("--dpi must be at least 50")
    return args


def main() -> None:
    a = parse_args()
    cfg = Config(a.data, a.output, a.target, tuple(a.features),
                 tuple(a.class_order) if a.class_order else None, a.test_fraction,
                 a.cv_folds, a.random_state, a.n_jobs, a.permutation_repeats, a.dpi)
    ClassificationAnalysis(cfg).run()


if __name__ == "__main__":
    main()
