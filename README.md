# viClassify

**viClassify** is a script-based command-line application for reproducible binary and multiclass classification of numerical tabular data. It validates and audits a CSV dataset, creates an untouched chronological holdout, tunes four classifier families only on the earlier training partition, evaluates each selected configuration on the holdout, and then refits production pipelines on all available data.

The application is intended for scientific and engineering datasets in which CSV row order represents time, process sequence, acquisition order, or another meaningful chronology. It exports detailed CSV diagnostics, publication-ready plots, serialized pipelines, model metadata, and explicit Logistic Regression equations in both standardized and original predictor units.

## Table of contents

- [Key capabilities](#key-capabilities)
- [Validation design at a glance](#validation-design-at-a-glance)
- [Requirements](#requirements)
- [Installation](#installation)
- [Expected data format](#expected-data-format)
- [Quick start](#quick-start)
- [Command-line reference](#command-line-reference)
- [End-to-end workflow](#end-to-end-workflow)
- [Preprocessing](#preprocessing)
- [Model families and mathematical rationale](#model-families-and-mathematical-rationale)
- [Hyperparameter tuning and model selection](#hyperparameter-tuning-and-model-selection)
- [Evaluation metrics](#evaluation-metrics)
- [Interpretability outputs](#interpretability-outputs)
- [Output files](#output-files)
- [Loading an exported model](#loading-an-exported-model)
- [Using the Logistic Regression equations](#using-the-logistic-regression-equations)
- [Reproducibility](#reproducibility)
- [Developer guide](#developer-guide)
- [Limitations and scientific cautions](#limitations-and-scientific-cautions)
- [Troubleshooting](#troubleshooting)
- [Security](#security)
- [Roadmap](#roadmap)
- [License](#license)

## Key capabilities

- Binary and multiclass classification.
- User-selectable target and ordered numerical predictor columns.
- Explicit prevention of target leakage through CLI validation.
- Detection of duplicate column names and the reserved `Sequence` column.
- Logging of unused input columns, failed numeric conversions, infinities, missing values, duplicate rows, constant features, and near-constant features.
- Descriptive statistics for the complete dataset and separately by class.
- Pearson and Spearman correlation matrices, ranked correlation-pair tables, and heatmaps.
- Class-distribution, histogram, class-wise boxplot, and sequence diagnostic plots.
- A final chronological holdout created before any hyperparameter tuning.
- Median imputation fitted inside each machine-learning pipeline.
- Standardization for Logistic Regression only.
- Stratified shuffled K-fold cross-validation on the chronological training partition.
- Multi-metric `GridSearchCV` with a configurable selection metric.
- Logistic Regression, Random Forest, Extra Trees, and Histogram Gradient Boosting.
- Evaluation artifacts fitted on chronological training data only.
- Production artifacts refitted on all rows after holdout evaluation.
- Accuracy, balanced accuracy, macro precision, macro recall, macro F1, weighted F1, Matthews correlation coefficient, and log loss.
- Per-class classification reports, predicted class probabilities, confidence, prediction margin, and misclassification tables.
- Count and recall-normalized confusion matrices.
- Holdout permutation importance for every model family.
- Logistic Regression intercepts and coefficients in standardized and original predictor units.
- A model export manifest, generated-file manifest, run configuration, validation notes, and package versions.

## Validation design at a glance

viClassify deliberately separates **model evaluation** from **final production fitting**.

```text
Complete CSV in original row order
│
├── Earlier rows: chronological training partition
│   ├── Stratified shuffled K-fold GridSearchCV
│   ├── Hyperparameter selection
│   └── Evaluation model fitted on the complete training partition
│
└── Final rows: untouched chronological holdout
    └── One-time evaluation of each selected model configuration

After holdout evaluation:
complete CSV ──> production model refit for each family
```

The holdout boundary is

```text
split_index = floor(n_rows * (1 - test_fraction))
```

Rows before `split_index` form the training partition. The remaining final rows form the testing partition. Tuning, preprocessing fitting, and configuration selection use only the training partition. After evaluation, the selected hyperparameters are retained and each production pipeline is refitted on the complete dataset.

> **Important:** the cross-validation folds are stratified and shuffled, not chronological. This is suitable only when observations within the chronological training partition may reasonably be treated as exchangeable. It does not eliminate leakage caused by repeated subjects, specimens, batches, cycles, or temporally dependent neighboring observations.

## Requirements

- Python 3.10 or later, because the source uses modern union type syntax such as `Path | None`.
- NumPy
- pandas
- Matplotlib
- SciPy
- scikit-learn
- joblib

Install the runtime dependencies:

```bash
python -m pip install numpy pandas matplotlib scipy scikit-learn joblib
```

For a reproducible project, pin tested versions in a `requirements.txt`, `requirements.lock`, or `pyproject.toml`. viClassify records the versions used for each run in `run_metadata.json`, but it does not currently enforce package-version constraints.

## Installation

The current project is a standalone Python script rather than an installed package.

Suggested repository layout:

```text
viClassify/
├── viclassify_cli.py
├── README.md
├── data_OF.csv
└── data_OBGM.csv
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib scipy scikit-learn joblib
python .\viclassify_cli.py --help
```

### Linux or macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib scipy scikit-learn joblib
python viclassify_cli.py --help
```

## Expected data format

The input must be a comma-separated CSV file containing:

1. One target column.
2. At least two distinct, nonempty target labels.
3. At least one selected predictor column.
4. Predictor values that are numerical or can be converted to numerical values.

The default column configuration is:

```text
X1, X2, X3, X4, X5, X6, X7, Y
```

Example:

```csv
X1,X2,X3,X4,X5,X6,X7,Y
1.30,4.20,2.10,8.00,0.50,3.20,7.10,O
1.10,3.90,2.80,7.50,0.30,3.70,6.80,F
1.50,4.60,3.00,8.20,0.70,3.10,7.60,O
```

### Input handling rules

- Leading and trailing whitespace is removed from CSV column names.
- Leading spaces after delimiters are ignored by `pandas.read_csv(..., skipinitialspace=True)`.
- The target and feature names supplied through the CLI are stripped of surrounding whitespace.
- Predictor conversion uses `pandas.to_numeric(..., errors="coerce")`.
- Non-convertible predictor entries become missing values and are reported in `numeric_conversion_problems.csv`.
- Positive and negative infinities are counted in `infinite_values.csv`, then replaced with missing values.
- Missing predictor values are retained in the audit dataset and median-imputed inside each fitted pipeline.
- Missing or empty target labels are rejected. Target labels are never imputed.
- A selected feature containing no usable numerical value is rejected.
- Duplicate input column names are rejected.
- An existing input column named `Sequence` is rejected because viClassify reserves that name.
- The target may not also be listed as a feature.
- Extra columns are allowed, recorded in `unused_columns.csv`, and excluded from modeling.
- Exact duplicate records are reported using the selected predictors and target. They are not removed automatically.
- Constant and near-constant features are reported but not removed automatically.

## Quick start

### Binary classification

Bash:

```bash
python viclassify_cli.py \
    --data data_OF.csv \
    --output of_results \
    --target Y \
    --features X1 X2 X3 X4 X5 X6 X7 \
    --class-order O F
```

Windows PowerShell:

```powershell
python .\viclassify_cli.py `
    --data .\data_OF.csv `
    --output .\of_results `
    --target Y `
    --features X1 X2 X3 X4 X5 X6 X7 `
    --class-order O F
```

### Multiclass classification

```bash
python viclassify_cli.py \
    --data data_OBGM.csv \
    --output obgm_results \
    --target Y \
    --features X1 X2 X3 X4 X5 X6 X7 \
    --class-order O B G M
```

### Change the selection metric

```bash
python viclassify_cli.py \
    --data data_OBGM.csv \
    --selection-metric balanced_accuracy
```

`--class-order` controls reporting and plotting order. It does not define the estimator's internal probability-column order. Exported artifacts therefore store both `reporting_class_order` and `estimator_class_order`.

## Command-line reference

| Argument | Required | Default | Validation and purpose |
|---|---:|---|---|
| `--data` | Yes | None | Input CSV path. The file must exist when analysis starts. |
| `--output` | No | `viclassify_results` | Output directory, created with parent directories when needed. Existing files with matching names are overwritten. |
| `--target` | No | `Y` | Nonempty target-column name. It must not also occur in `--features`. |
| `--features` | No | `X1 X2 X3 X4 X5 X6 X7` | One or more ordered, nonempty, distinct predictor-column names. |
| `--class-order` | No | Detected and sorted | Optional distinct, nonempty reporting labels. Unobserved requested labels produce a warning and are omitted. Observed labels not listed are appended in sorted order. |
| `--test-fraction` | No | `0.20` | Fraction of final rows reserved for chronological holdout evaluation. Must be strictly between 0 and 1. |
| `--cv-folds` | No | `5` | Requested maximum stratified fold count. Must be at least 2. Automatically limited by the rarest class count in chronological training data. |
| `--random-state` | No | `42` | Seed supplied to shuffled CV, supported estimators, and permutation importance. |
| `--n-jobs` | No | `-1` | Parallel workers. Must be `-1` or a positive integer. `-1` requests all processors supported by joblib/scikit-learn. |
| `--permutation-repeats` | No | `20` | Number of holdout permutations per feature and model. Must be positive. |
| `--dpi` | No | `150` | Plot resolution. Must be at least 50. |
| `--near-constant-threshold` | No | `0.99` | A feature is flagged when its most common nonmissing value has at least this fraction. Must be in `(0, 1]`. |
| `--selection-metric` | No | `macro_f1` | Metric used by `GridSearchCV` for refitting and for selecting the best overall production family. Choices: `accuracy`, `balanced_accuracy`, `macro_precision`, `macro_recall`, `macro_f1`, `weighted_f1`. |

## End-to-end workflow

`ClassificationAnalysis.run()` executes these stages in order:

1. **Load and validate data**
   - Check the input path, headers, requested columns, dataset size, reserved names, and target labels.
   - Add a one-based `Sequence` column that preserves original row order.
   - Coerce selected predictors to numerical values and replace infinities with missing values.

2. **Audit and describe**
   - Export missingness, duplicates, class frequencies, overall summaries, class-wise summaries, and feature-quality indicators.

3. **Analyze correlations**
   - Export Pearson and Spearman matrices, ranked variable pairs, and heatmaps.

4. **Create exploratory plots**
   - Plot class balance, feature histograms, and class-wise boxplots.

5. **Create the chronological holdout**
   - Reserve the final fraction of rows.
   - Require every observed class in the training partition.
   - Warn if a class is absent from the testing partition or if the holdout has fewer than 10 observations.

6. **Build model families**
   - Construct preprocessing and estimator pipelines and define search grids.

7. **Tune on chronological training data only**
   - Use shuffled stratified K-fold CV and multi-metric scoring.
   - Refit the configuration maximizing the chosen selection metric.
   - Export complete search results and fold scores for the selected configuration.

8. **Evaluate the untouched holdout**
   - Fit each selected configuration on all chronological training rows.
   - Export an evaluation pipeline for each family.
   - Predict the holdout once and export metrics, reports, probabilities, errors, confusion matrices, confidence plots, and holdout permutation importance.

9. **Export evaluation Logistic Regression parameters**
   - Export imputation medians, scaling parameters, standardized coefficients, original-unit coefficients, equations, and a coefficient plot.

10. **Refit production models on all data**
    - Fit every selected family configuration on all rows.
    - Export every production pipeline and the best overall production pipeline.
    - Export production Logistic Regression parameters.

11. **Record metadata and generated files**
    - Save configuration, methodological notes, package versions, and an exact file manifest.

## Preprocessing

### Median imputation

For feature `j`, missing values are replaced with the median learned from the data supplied to the pipeline fit:

```text
x_tilde_ij = x_ij                         if x_ij is observed
x_tilde_ij = median_training(x_j)         if x_ij is missing
```

Because `SimpleImputer` is inside the pipeline, each CV training fold learns its own median. This prevents validation-fold information from entering preprocessing during CV. The evaluation model learns medians from the chronological training partition; the production model learns them from all available rows.

### Standardization for Logistic Regression

For each imputed feature:

```text
z_ij = (x_tilde_ij - mu_j) / s_j
```

where `mu_j` and `s_j` are the training mean and scale learned by `StandardScaler`. Standardization makes the regularization penalty comparable across predictors measured in different units and improves numerical conditioning.

Tree-based models receive median-imputed values but are not standardized because tree split ordering is invariant to monotonic affine rescaling of a single feature.

## Model families and mathematical rationale

### Logistic Regression

Logistic Regression models class probabilities through linear scores. It provides a comparatively interpretable baseline and works well when classes can be separated by approximately linear decision boundaries in standardized feature space.

#### Binary case

For estimator classes `C0` and `C1`, the score is

```text
eta = beta_0 + sum_j(beta_j * z_j)
```

and the sigmoid probability is

```text
P(C1 | z) = 1 / (1 + exp(-eta))
P(C0 | z) = 1 - P(C1 | z)
```

The log-odds are linear:

```text
log(P(C1 | z) / P(C0 | z)) = eta
```

#### Multiclass case

For class `k`:

```text
eta_k = beta_0k + sum_j(beta_kj * z_j)
P(Y = k | z) = exp(eta_k) / sum_l(exp(eta_l))
```

A numerically stable manual softmax subtracts `max_l(eta_l)` from every score before exponentiation.

#### Objective and regularization

The `lbfgs` solver minimizes multinomial or binary negative log-likelihood with L2 regularization. Conceptually:

```text
J(beta) = -sum_i sum_k y_ik * log(p_ik) + lambda * ||beta||_2^2
```

The scikit-learn parameter `C` is inverse regularization strength. Smaller `C` means stronger shrinkage; larger `C` allows larger coefficients. viClassify tunes:

```text
C = [0.1, 1.0, 10.0]
```

Implementation settings:

- Median imputation.
- Standard scaling.
- `class_weight="balanced"`.
- `solver="lbfgs"`.
- `max_iter=10000`.

Balanced class weights are inversely proportional to observed class frequencies. They increase the contribution of rarer classes to the loss, which can improve minority-class sensitivity but changes the fitted intercepts and probability estimates relative to unweighted maximum likelihood.

### Random Forest

Random Forest builds an ensemble of decision trees from bootstrap samples. At each split, it evaluates only a random subset of predictors. For classification, predicted class probabilities are obtained by averaging tree probabilities, followed by selection of the largest probability.

A split is chosen to reduce node impurity. With the default Gini criterion, impurity is

```text
G(t) = 1 - sum_k(p_k(t)^2)
```

and a candidate split is valued by its weighted impurity reduction:

```text
Delta G = G(parent)
          - (n_left / n_parent) * G(left)
          - (n_right / n_parent) * G(right)
```

Randomization and aggregation reduce the variance of a single deep tree. Random Forest can model nonlinear effects and high-order interactions without explicitly specifying them.

viClassify uses `class_weight="balanced_subsample"`, recalculating class weights for each bootstrap sample, and tunes:

```text
n_estimators      = [300, 600]
min_samples_leaf  = [1, 2, 5]
max_features      = ["sqrt", None]
```

Larger forests usually stabilize predictions but increase computation and model size. Larger leaves regularize the model by smoothing local partitions. `max_features="sqrt"` increases diversity among trees; `None` allows every predictor to be considered at each split.

### Extra Trees

Extra Trees, or Extremely Randomized Trees, also averages many decision trees, but it introduces stronger split randomization than Random Forest. The additional randomization can reduce variance and training cost at the expense of potentially greater bias.

Like Random Forest, it naturally represents nonlinear boundaries and feature interactions. It is especially useful as a complementary ensemble because its randomization mechanism can produce different error patterns from bootstrap-based forests.

viClassify uses `class_weight="balanced"` and tunes the same structural grid:

```text
n_estimators      = [300, 600]
min_samples_leaf  = [1, 2, 5]
max_features      = ["sqrt", None]
```

### Histogram Gradient Boosting

Histogram Gradient Boosting constructs an additive model sequentially:

```text
F_m(x) = F_(m-1)(x) + learning_rate * h_m(x)
```

Each new tree `h_m` is fitted to improve the current model with respect to the classification loss. Continuous values are discretized into histogram bins, allowing efficient split evaluation on larger tabular datasets.

Boosting primarily reduces bias by correcting errors from the existing ensemble. The learning rate controls each tree's contribution, while the number of iterations controls ensemble length. These values should be interpreted jointly: a smaller learning rate often requires more iterations.

viClassify tunes:

```text
learning_rate       = [0.05, 0.10]
max_iter            = [200, 400]
max_leaf_nodes      = [15, 31]
l2_regularization   = [0.0, 1.0]
```

`max_leaf_nodes` controls tree complexity. L2 regularization penalizes large leaf values and can reduce overfitting. The implementation does not set class weights for Histogram Gradient Boosting, so class imbalance is addressed through the macro-oriented selection metric rather than weighted fitting.

## Hyperparameter tuning and model selection

### Effective fold count

The requested CV fold count is automatically limited by the rarest class in the chronological training partition:

```text
effective_folds = min(requested_cv_folds, minimum_training_class_count)
```

At least two training observations are required in every class. The splitter is:

```python
StratifiedKFold(
    n_splits=effective_folds,
    shuffle=True,
    random_state=random_state,
)
```

Stratification approximately preserves class proportions in each fold. Shuffling makes the fold assignment reproducible through `random_state`.

### Multi-metric grid search

Every hyperparameter candidate is assessed using:

- Accuracy
- Balanced accuracy
- Macro precision
- Macro recall
- Macro F1
- Weighted F1

`GridSearchCV` refits the candidate maximizing `--selection-metric`. The same CV selection score determines the best overall production family. Holdout performance is reported for external comparison but is not used to choose the production family, preventing direct holdout-driven model selection.

The exported `selected_configuration_training_cv_*.csv` files re-evaluate the already selected configuration on the same CV design. These are useful fold-level diagnostics, but they are **not nested-CV generalization estimates** because selection and reporting reuse the same training-partition validation design.

## Evaluation metrics

Let `N` be the number of holdout observations, `K` the number of classes, and `TP_k`, `FP_k`, and `FN_k` the one-versus-rest counts for class `k`.

### Accuracy

```text
accuracy = number of correct predictions / N
```

Accuracy is easy to interpret but may conceal poor minority-class performance.

### Balanced accuracy

```text
balanced_accuracy = (1 / K) * sum_k recall_k
```

It gives every class equal influence regardless of frequency.

### Precision, recall, and F1

```text
precision_k = TP_k / (TP_k + FP_k)
recall_k    = TP_k / (TP_k + FN_k)
F1_k        = 2 * precision_k * recall_k
              / (precision_k + recall_k)
```

Undefined precision or recall contributions are set to zero. Macro averages assign equal weight to classes:

```text
macro_F1 = (1 / K) * sum_k F1_k
```

Weighted F1 weights each class by its holdout support:

```text
weighted_F1 = sum_k(n_k * F1_k) / N
```

Macro F1 is the default selection metric because it exposes weak performance on small classes more clearly than accuracy or weighted F1.

### Matthews correlation coefficient

MCC measures agreement between observed and predicted labels while taking the complete confusion matrix into account. It is useful for imbalanced binary and multiclass problems. Values approach `1` for perfect prediction, `0` for agreement near chance, and `-1` for systematic disagreement in the binary limiting case.

### Log loss

For predicted probability `p_i,y_i` assigned to the true class:

```text
log_loss = -(1 / N) * sum_i log(p_i,y_i)
```

Log loss rewards calibrated, confident correct predictions and strongly penalizes confident errors. If the holdout does not contain all estimator classes, viClassify still supplies the estimator class order to `log_loss`. If scikit-learn rejects the calculation, the exported value is `NaN`.

### CV-to-holdout gap

`model_comparison.csv` includes:

```text
cv_holdout_<selection_metric>_gap
    = mean training-partition CV score - holdout score
```

A large positive gap can indicate distribution shift, dependence, overfitting to the training and tuning design, or an unusually difficult chronological segment. It is a diagnostic, not a formal hypothesis test.

## Interpretability outputs

### Logistic Regression coefficients

viClassify exports both standardized and original-unit equations for evaluation and production Logistic Regression models.

Standardized coefficients apply to:

```text
z_j = (imputed_x_j - scaler_mean_j) / scaler_scale_j
```

The original-unit transformation is:

```text
beta_j_original = beta_j / scaler_scale_j

beta_0_original = beta_0
                  - sum_j(beta_j * scaler_mean_j / scaler_scale_j)
```

Therefore:

```text
eta = beta_0_original + sum_j(beta_j_original * imputed_x_j)
```

Missing values must still be replaced with the exported imputation medians before using the original-unit equation.

Coefficient signs describe conditional changes in a class score while the other included predictors are held fixed. They do not establish causality. Strong correlation among predictors can make individual coefficients unstable even when predictive accuracy remains acceptable.

### Permutation importance

For each fitted evaluation model and feature `j`, viClassify repeatedly shuffles that feature in the chronological holdout and measures the change in macro F1:

```text
importance_j = baseline_holdout_macro_F1
               - permuted_holdout_macro_F1_j
```

A larger positive value indicates greater dependence of holdout performance on that feature. Near-zero values indicate little marginal contribution under the fitted model and observed holdout. Negative values mean the permuted data performed better in that sample and can arise from noise, correlated features, small holdouts, distribution shift, or model instability.

Because permutation importance is computed after evaluating the same holdout, it is an interpretation of that holdout rather than an additional independent validation result.

### Prediction confidence and margin

For each holdout prediction:

```text
Prediction_Confidence = largest predicted class probability
Prediction_Margin     = largest probability - second-largest probability
```

These quantities indicate separation among model outputs, not guaranteed empirical correctness. viClassify does not currently perform probability calibration or calibration-curve analysis.

## Output files

The exact set of files depends on the four fixed model families. `generated_files.txt` is the authoritative manifest for a completed run.

### Data audit and exploratory analysis

```text
unused_columns.csv
numeric_conversion_problems.csv
infinite_values.csv
missing_values.csv
duplicate_rows.csv
class_distribution.csv
class_distribution.png
numerical_summary.csv
numerical_summary_by_class.csv
feature_audit.csv
pearson_correlation_matrix.csv
pearson_correlation_pairs.csv
pearson_correlation_matrix.png
spearman_correlation_matrix.csv
spearman_correlation_pairs.csv
spearman_correlation_matrix.png
feature_histograms.png
feature_boxplots_by_class.png
```

- `feature_audit.csv` includes nonmissing count, missing count, unique values, variance, most-common-value fraction, constant flag, and near-constant flag.
- Correlation matrices use pairwise available observations according to pandas behavior. Correlation-pair files rank unique feature pairs by absolute correlation.

### Chronological split diagnostics

```text
chronological_split_distribution.csv
chronological_split_assignments.csv
chronological_split_distribution.png
predictors_and_classes_by_sequence.png
```

`chronological_split_assignments.csv` maps each generated `Sequence` value and target label to `training` or `testing`.

### Hyperparameter search outputs

For each model slug:

```text
tuning_cv_results_<model>.csv
selected_configuration_training_cv_<model>.csv
hyperparameter_search_<model>.png
```

Model slugs are:

```text
logistic_regression
random_forest
extra_trees
histogram_gradient_boosting
```

`tuning_cv_results_*.csv` contains the complete `GridSearchCV.cv_results_` table. The selected-configuration files contain fold-level scores and explicitly state that they are tuning-CV scores rather than nested-CV estimates.

### Evaluation artifacts and holdout outputs

For each model slug:

```text
evaluation_<model>.joblib
holdout_report_<model>.csv
holdout_predictions_<model>.csv
holdout_misclassifications_<model>.csv
holdout_confusion_matrix_<model>.csv
holdout_confusion_matrix_normalized_<model>.csv
holdout_confusion_matrix_<model>.png
holdout_prediction_confidence_<model>.png
holdout_errors_by_sequence_<model>.png
holdout_permutation_importance_<model>.csv
holdout_permutation_importance_<model>.png
```

Combined outputs:

```text
chronological_holdout_metrics.csv
model_comparison.csv
model_comparison.png
```

Evaluation artifacts are fitted only on the chronological training partition. Use them to reproduce holdout predictions and inspect the model at evaluation time. Do not confuse them with full-data production models.

### Evaluation Logistic Regression equations

```text
evaluation_logistic_preprocessing_parameters.csv
evaluation_logistic_coefficients_and_intercepts.csv
evaluation_logistic_coefficients_original_units.csv
evaluation_logistic_equation_usage.json
evaluation_logistic_coefficients.png
```

### Production model artifacts

```text
production_logistic_regression.joblib
production_random_forest.joblib
production_extra_trees.joblib
production_histogram_gradient_boosting.joblib
best_overall_model.joblib
model_export_manifest.csv
```

Every production model is refitted on all available rows using the hyperparameters selected from chronological training CV. `best_overall_model.joblib` duplicates the production pipeline from the family with the highest mean CV selection score.

### Production Logistic Regression equations

```text
production_logistic_preprocessing_parameters.csv
production_logistic_coefficients_and_intercepts.csv
production_logistic_coefficients_original_units.csv
production_logistic_equation_usage.json
production_logistic_coefficients.png
```

### Metadata

```text
run_metadata.json
generated_files.txt
```

`run_metadata.json` records configuration, observed reporting class order, validation and interpretation notes, and versions of Python, NumPy, pandas, SciPy, and scikit-learn.

## Loading an exported model

Each `.joblib` artifact is a dictionary. Current keys include:

```text
model
model_type
model_role
features
reporting_class_order
estimator_class_order
selection_metric
best_parameters
```

Example:

```python
import joblib
import pandas as pd

artifact = joblib.load(
    "of_results/production_extra_trees.joblib"
)

pipeline = artifact["model"]
features = artifact["features"]
estimator_classes = artifact["estimator_class_order"]

new_data = pd.DataFrame(
    [
        {
            "X1": 1.2,
            "X2": 3.4,
            "X3": 2.1,
            "X4": 5.6,
            "X5": 7.8,
            "X6": 4.3,
            "X7": 6.5,
        }
    ]
)

X_new = new_data[features]
prediction = pipeline.predict(X_new)
probabilities = pipeline.predict_proba(X_new)

print("Model role:", artifact["model_role"])
print("Predicted class:", prediction[0])

for label, probability in zip(estimator_classes, probabilities[0]):
    print(f"P({label}) = {probability:.6f}")
```

Always map `predict_proba` columns with `estimator_class_order`, not `reporting_class_order`.

### Evaluation versus production artifacts

- Use `evaluation_<model>.joblib` to reproduce predictions from the untouched chronological holdout analysis.
- Use `production_<model>.joblib` for subsequent predictions after evaluation because it was fitted on all available data.
- Use `best_overall_model.joblib` when the CV-selected overall family is appropriate for deployment.

The production model has no untouched internal test set remaining after full-data refitting. Its expected generalization should be judged from the earlier evaluation artifacts and reports, subject to distribution stability.

## Using the Logistic Regression equations

### Inspect fitted arrays

```python
import joblib

artifact = joblib.load(
    "of_results/production_logistic_regression.joblib"
)

pipeline = artifact["model"]
classifier = pipeline.named_steps["classifier"]

print("Estimator class order:", classifier.classes_)
print("Intercepts:", classifier.intercept_)
print("Coefficients:", classifier.coef_)
```

For binary classification, scikit-learn stores one coefficient row describing `classes_[1]` relative to `classes_[0]`. For multiclass classification, it stores one score equation per estimator class.

### Recommended practice

Use the serialized pipeline for operational prediction. It preserves:

- exact feature order;
- median imputation;
- standardization where required;
- selected estimator settings;
- estimator class order; and
- probability-column order.

Manual equations are most useful for documentation, verification, scientific interpretation, and implementation in environments where loading a Python pipeline is not possible.

## Reproducibility

The default random seed is `42`:

```bash
python viclassify_cli.py \
    --data data_OF.csv \
    --random-state 123
```

The seed controls shuffled fold assignment, supported estimator randomization, and permutation importance. Reproducibility also depends on:

- unchanged input row order and values;
- identical requested feature order;
- compatible Python and package versions;
- consistent parallel numerical libraries and platform behavior; and
- unchanged source code and search grids.

For traceable scientific runs, archive together:

```text
viclassify_cli.py
input CSV or its cryptographic hash
run_metadata.json
generated_files.txt
model_export_manifest.csv
complete output directory
requirements or environment lock file
```

The current metadata does not record a source-code hash, input-file hash, timestamp, operating system, or joblib version. These are reasonable future additions for stricter provenance.

## Developer guide

### Architecture

The source contains two primary abstractions:

- `Config`: an immutable dataclass containing validated run settings.
- `ClassificationAnalysis`: an orchestration class containing data state, model definitions, tuning results, fitted evaluation and production pipelines, exports, and plotting methods.

The CLI entry flow is:

```text
parse_args()
    └── Config(...)
        └── ClassificationAnalysis(cfg).run()
```

### Main extension points

#### Add a model family

Update all of the following consistently:

1. `ClassificationAnalysis.MODEL_NAMES`.
2. `self.models` in `build_models()`.
3. `self.grids` in `build_models()`.
4. Ensure the estimator implements `predict()` and `predict_proba()`.
5. Confirm compatibility with the shared scoring metrics and permutation importance.
6. Confirm serialization with joblib.
7. Update this README and tests.

The current runtime loops over `self.models`; therefore, most tuning, evaluation, plotting, and export filenames are generated automatically once a compatible pipeline and grid are registered.

#### Add a selection metric

1. Add it to `ClassificationAnalysis.SCORING`.
2. Add the exact key to the `--selection-metric` choices.
3. Confirm higher values always represent better performance, because results are sorted in descending order.
4. Confirm the metric is meaningful for every supported class configuration.
5. Update documentation and tests.

#### Change preprocessing

Preprocessing is part of each scikit-learn `Pipeline`. Keep learned transformations inside the pipeline to prevent validation leakage. If categorical variables are added, use explicit column groups and compatible transformers in `ColumnTransformer`, and update feature-name handling for coefficient exports and interpretability.

#### Change output files

All tracked output paths should be created by `self.path(name)`. This registers the filename for `generated_files.txt`. Direct writes that bypass `self.path()` will not appear in the manifest.

### Suggested development checks

The repository does not currently include an automated test suite. Recommended tests include:

- CLI rejection of invalid fractions, fold counts, worker counts, DPI, duplicate features, target leakage, and invalid thresholds.
- Rejection of missing files, empty datasets, duplicate columns, reserved `Sequence`, missing target labels, one-class targets, and all-missing features.
- Numeric coercion and infinity replacement.
- Preservation of chronological split boundaries.
- Absence of holdout rows from grid search.
- Automatic fold-count reduction.
- Exact artifact keys and estimator probability order.
- Correct original-unit Logistic Regression conversion.
- Expected output manifest for binary and multiclass fixtures.
- Reproducibility with a fixed seed.
- Behavior when the holdout lacks one or more classes.
- Headless plotting under the configured Matplotlib `Agg` backend.

A practical test stack would use `pytest`, `tmp_path`, small deterministic CSV fixtures, and `subprocess.run()` for end-to-end CLI tests.

### Code-quality tooling

Possible developer tooling, not currently enforced by the script:

```bash
python -m pip install pytest ruff mypy
ruff check viclassify_cli.py
ruff format --check viclassify_cli.py
pytest
```

Type checking may require additional annotations or third-party stubs before strict `mypy` settings pass.

### Packaging direction

A future package layout could be:

```text
viClassify/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── viclassify/
│       ├── __init__.py
│       ├── cli.py
│       └── analysis.py
└── tests/
```

Suggested names:

```text
Distribution/package: viclassify
Console command:       viclassify
```

A `pyproject.toml` console entry point would allow:

```bash
viclassify --data data_OF.csv
```

## Limitations and scientific cautions

- **Chronology must be meaningful.** The final-row holdout evaluates a future-like segment only when file order represents a defensible sequence.
- **CV inside training is shuffled.** It does not reproduce forward-chaining or blocked temporal validation.
- **No grouped validation is implemented.** Related records from the same subject, specimen, batch, cycle, or run can appear in both training and validation folds.
- **No nested CV is implemented.** Training-partition CV scores are used for hyperparameter selection and are not unbiased nested-CV estimates.
- **The holdout is reused across model families.** It remains untouched during tuning, but inspecting four families on one holdout can still influence human model choice. The scripted best overall artifact is selected by training CV, not holdout score.
- **No calibration analysis is performed.** Predicted probabilities and confidence values should not automatically be interpreted as calibrated frequencies.
- **No threshold optimization is performed.** Class predictions use each estimator's default decision rule.
- **No sample or class weighting is used for Histogram Gradient Boosting.** Severe imbalance may require a different design.
- **Duplicates are reported but retained.** Repeated observations can inflate apparent performance if they are scientifically dependent.
- **Constant and near-constant predictors are retained.** They may add cost without predictive information.
- **All predictors must be numerical.** Categorical encoding is not implemented.
- **Correlation is not causation.** Pearson captures linear association; Spearman captures monotonic rank association. Neither establishes a causal relation.
- **Multicollinearity affects coefficient interpretation.** Correlated predictors can produce unstable conditional Logistic Regression coefficients.
- **Permutation importance is model- and holdout-dependent.** Correlated predictors can divide or mask importance.
- **Small holdouts are unstable.** The script warns below 10 observations, but scientific adequacy generally depends on class-specific counts and intended precision, not only total size.
- **Production models are refitted on all data.** They no longer possess an internal untouched test set.
- **Serialized models are environment-sensitive.** Reuse them with compatible scikit-learn and Python versions.

## Troubleshooting

### Input file not found

Supply the correct path:

```bash
python viclassify_cli.py --data ./data/data_OF.csv
```

### Required columns are missing

Inspect the CSV header and match `--target` and `--features` exactly. Column-name surrounding whitespace is removed, but spelling and case still matter.

### Input already contains `Sequence`

Rename or remove the input `Sequence` column. viClassify creates this reserved column to track original row position.

### A selected feature has no usable numerical values

Inspect `numeric_conversion_problems.csv` if it was written before termination. Correct the source values or remove that feature from `--features`.

### Target contains missing or empty labels

Correct or remove the affected records. viClassify does not impute target labels.

### Chronological training partition lacks a class

The classifier cannot learn a class absent from training. Reduce `--test-fraction`, obtain earlier examples of the class, or redesign validation. Do not randomly reorder genuinely chronological data merely to suppress this error.

### Chronological testing partition lacks a class

The run continues with a warning. Per-class metrics for the absent class are not empirically evaluated in that holdout, and some aggregate metrics require careful interpretation.

### Cross-validation cannot run

Every class in the chronological training partition must contain at least two observations. The requested fold count is automatically reduced to the rarest training-class count, but never below two.

### Execution is slow

The default grids evaluate:

- 3 Logistic Regression configurations;
- 12 Random Forest configurations;
- 12 Extra Trees configurations; and
- 16 Histogram Gradient Boosting configurations.

Each configuration is fitted once per effective CV fold, followed by additional selected-configuration CV, holdout model fitting, permutation importance, and full-data production refitting. Use available processors:

```bash
python viclassify_cli.py \
    --data data_OF.csv \
    --n-jobs -1
```

Reducing grids in `build_models()` lowers computation but narrows the model search. Lowering `--permutation-repeats` reduces interpretation cost but increases importance uncertainty.

### Plots do not appear on screen

This is expected. Matplotlib uses the noninteractive `Agg` backend and writes PNG files to the output directory.

### Existing results disappear

The application reuses deterministic filenames and overwrites files with matching names in the selected output directory. Use a new output directory for each run when preserving prior results.

## Security

- Load `.joblib` files only from trusted sources. Deserialization can execute malicious code.
- Treat input CSV files and generated predictions according to applicable data-governance requirements.
- Review output directories before sharing them because prediction tables reproduce selected feature values and target labels.
- Avoid exposing sensitive filesystem paths from `run_metadata.json`; it records resolved input and output paths.

## Roadmap

Potential future extensions include:

- grouped and stratified-group cross-validation;
- blocked, rolling, or expanding-window temporal validation;
- nested cross-validation;
- probability calibration and calibration plots;
- decision-threshold optimization;
- cost-sensitive Histogram Gradient Boosting;
- categorical predictors;
- YAML or TOML configuration;
- input and source hashing for stronger provenance;
- automated tests and continuous integration;
- `pyproject.toml` packaging and a `viclassify` console entry point;
- additional estimators and user-configurable search grids.

## Project status

viClassify currently provides a complete standalone CLI workflow for numerical tabular binary and multiclass classification. The code distinguishes chronological-training evaluation models from full-data production refits and exports the information needed to prevent those roles from being confused.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
