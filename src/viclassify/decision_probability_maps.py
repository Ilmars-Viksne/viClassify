"""Plot two-predictor slices of a fitted classifier's class probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


@dataclass(frozen=True)
class ProbabilityMapConfig:
    """Visual settings for probability maps."""

    grid_resolution: int = 200
    padding_fraction: float = 0.05
    contour_levels: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)
    multiclass_columns: int = 3
    maximum_classes_per_figure: int = 6
    probability_colormap: str = "viridis"
    show_decision_overview: bool = True
    show_probability_contours: bool = True
    show_other_class_points: bool = True
    show_fixed_median_annotation: bool = True
    show_colorbar: bool = True
    show_legends: bool = True
    dpi: int = 160
    file_format: str = "png"


class DecisionProbabilityMapper:
    """Create decision maps and, when available, probability maps.

    Any fitted classifier with ``predict`` is supported. Classifiers that
    also expose ``predict_proba`` get probability surfaces as well. All
    non-plotted predictors are fixed at their training medians, and grid
    points pass through the estimator (including pipeline preprocessing).
    """

    def __init__(
        self,
        estimator: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        config: ProbabilityMapConfig | None = None,
    ) -> None:
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if X_train.empty or X_train.columns.has_duplicates:
            raise ValueError("X_train must have rows and unique column names.")
        if not all(pd.api.types.is_numeric_dtype(X_train[c]) for c in X_train):
            raise ValueError("All predictors must be numeric.")
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have equal lengths.")
        self.estimator = estimator
        self.X_train = X_train.copy()
        self.y_train = pd.Series(y_train, index=X_train.index).copy()
        self.config = config or ProbabilityMapConfig()
        if self.config.grid_resolution < 2:
            raise ValueError("grid_resolution must be at least 2.")

    def plot_selected_pairs_multiclass(
        self,
        predictor_pairs: list[tuple[str, str]],
        X_points: pd.DataFrame,
        y_points: pd.Series,
        output_directory: str | Path,
        filename_prefix: str = "probability_map",
    ) -> list[Path]:
        """Save a decision overview and one probability panel per class/pair."""
        classes = list(getattr(self.estimator, "classes_", pd.unique(self.y_train)))
        has_probabilities = callable(getattr(self.estimator, "predict_proba", None))
        if not predictor_pairs:
            raise ValueError("At least one predictor pair is required.")
        if len(X_points) != len(y_points):
            raise ValueError("X_points and y_points must have equal lengths.")
        for first, second in predictor_pairs:
            if (
                first == second
                or first not in self.X_train
                or second not in self.X_train
            ):
                raise ValueError(f"Invalid predictor pair: {(first, second)!r}.")

        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        ext = self.config.file_format.lstrip(".")

        for pair_number, (first, second) in enumerate(predictor_pairs, start=1):
            x_values = self.X_train[first]
            y_values = self.X_train[second]
            x_pad = (
                x_values.max() - x_values.min()
            ) * self.config.padding_fraction or 1.0
            y_pad = (
                y_values.max() - y_values.min()
            ) * self.config.padding_fraction or 1.0
            gx, gy = np.meshgrid(
                np.linspace(
                    x_values.min() - x_pad,
                    x_values.max() + x_pad,
                    self.config.grid_resolution,
                ),
                np.linspace(
                    y_values.min() - y_pad,
                    y_values.max() + y_pad,
                    self.config.grid_resolution,
                ),
            )
            grid = pd.DataFrame(
                {
                    column: np.full(gx.size, self.X_train[column].median())
                    for column in self.X_train
                },
            )
            grid[first] = gx.ravel()
            grid[second] = gy.ravel()
            predicted = np.asarray(self.estimator.predict(grid)).reshape(gx.shape)
            probabilities = None
            if has_probabilities:
                probabilities = np.asarray(self.estimator.predict_proba(grid))

            if self.config.show_decision_overview:
                fig, ax = plt.subplots(figsize=(8, 6))
                class_codes = {label: i for i, label in enumerate(classes)}
                codes = np.vectorize(class_codes.__getitem__)(predicted)
                ax.contourf(
                    gx,
                    gy,
                    codes,
                    levels=np.arange(len(classes) + 1) - 0.5,
                    cmap=ListedColormap(plt.get_cmap("tab10").colors[: len(classes)]),
                    alpha=0.28,
                )
                for label in classes:
                    mask = np.asarray(y_points) == label
                    ax.scatter(
                        X_points.loc[mask, first],
                        X_points.loc[mask, second],
                        s=16,
                        label=str(label),
                        edgecolors="white",
                        linewidths=0.3,
                    )
                ax.set(
                    title=f"Decision regions: {first} vs {second}",
                    xlabel=first,
                    ylabel=second,
                )
                if self.config.show_legends:
                    ax.legend(title="Observed class")
                path = (
                    output
                    / f"{filename_prefix}_{pair_number}_{first}_vs_{second}_decision.{ext}"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                fig.tight_layout()
                fig.savefig(path, dpi=self.config.dpi)
                plt.close(fig)
                paths.append(path)

            if probabilities is None:
                continue

            class_count = min(len(classes), self.config.maximum_classes_per_figure)
            columns = min(self.config.multiclass_columns, class_count)
            rows = int(np.ceil(class_count / columns))
            fig, axes = plt.subplots(
                rows, columns, figsize=(6 * columns, 4.8 * rows), squeeze=False
            )
            for i, ax in enumerate(axes.flat):
                if i >= class_count:
                    ax.set_visible(False)
                    continue
                surface = probabilities[:, i].reshape(gx.shape)
                filled = ax.contourf(
                    gx,
                    gy,
                    surface,
                    levels=30,
                    cmap=self.config.probability_colormap,
                    vmin=0,
                    vmax=1,
                )
                if self.config.show_probability_contours:
                    lines = ax.contour(
                        gx,
                        gy,
                        surface,
                        levels=self.config.contour_levels,
                        colors="black",
                        linewidths=0.7,
                    )
                    ax.clabel(lines, inline=True, fontsize=8, fmt="%.2f")
                if self.config.show_other_class_points:
                    for label in classes:
                        mask = np.asarray(y_points) == label
                        ax.scatter(
                            X_points.loc[mask, first],
                            X_points.loc[mask, second],
                            s=12,
                            label=str(label),
                            edgecolors="white",
                            linewidths=0.25,
                        )
                ax.set(title=f"P({classes[i]})", xlabel=first, ylabel=second)
                if self.config.show_fixed_median_annotation:
                    ax.text(
                        0.02,
                        0.02,
                        "Other predictors fixed at training medians",
                        transform=ax.transAxes,
                        fontsize=7,
                        color="white",
                    )
                if self.config.show_colorbar:
                    fig.colorbar(filled, ax=ax, label="Predicted probability")
                if (
                    self.config.show_legends
                    and i == 0
                    and self.config.show_other_class_points
                ):
                    ax.legend(title="Observed class", fontsize="small")
            fig.suptitle(f"Class probabilities: {first} vs {second}")
            path = (
                output
                / f"{filename_prefix}_{pair_number}_{first}_vs_{second}_probabilities.{ext}"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(path, dpi=self.config.dpi)
            plt.close(fig)
            paths.append(path)
        return paths
