# viClassify

**viClassify** is a command-line application for reproducible binary and multiclass classification of tabular data. It validates a CSV dataset, audits data quality, calculates descriptive statistics and correlation matrices, tunes four classifier families, evaluates them on a chronological holdout set, and exports fitted preprocessing and prediction pipelines.

The current implementation supports:

- Logistic Regression
- Random Forest
- Extra Trees
- Histogram Gradient Boosting

Model-family hyperparameters are selected using stratified cross-validation and macro F1. The best fitted pipeline from every model family is exported, together with the best overall model.

## Features

- Binary and multiclass classification
- Configurable target and predictor columns
- Median imputation for missing predictor values
- Standardization for Logistic Regression
- Stratified shuffled cross-validation
- Hyperparameter tuning with `GridSearchCV`
- Chronological holdout evaluation
- Macro F1 model selection
- Accuracy, balanced accuracy, precision, recall, F1, MCC, and log-loss metrics
- Per-class classification reports
- Predicted probabilities for every class
- Confusion-matrix plots
- Pearson and Spearman correlation matrices
- Logistic Regression coefficients and intercepts
- Binary sigmoid and multiclass softmax usage documentation
- Export of the best model from each supported family
- Reproducible configuration and environment metadata

## Expected data format

The input must be a CSV file containing:

1. One target column.
2. Two or more distinct target classes.
3. One or more numerical predictor columns.

The default configuration expects:

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

Predictor values that cannot be converted to numbers are treated as missing and subsequently median-imputed inside each model pipeline. Target labels must not be missing or empty.

## Requirements

- Python 3.10 or later
- NumPy
- pandas
- Matplotlib
- SciPy
- scikit-learn
- joblib

Install the required packages:

```bash
python -m pip install numpy pandas matplotlib scipy scikit-learn joblib
```

A virtual environment is recommended:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib scipy scikit-learn joblib
```

### Linux or macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib scipy scikit-learn joblib
```

## Installation

Clone or download the project and place the CLI script in the repository root:

```text
viClassify/
├── obgm_classification_cli.py
├── README.md
└── data_OF.csv
```

Display the CLI help:

```bash
python obgm_classification_cli.py --help
```

## Basic usage

### Binary classification

Example for target classes `O` and `F`:

```bash
python obgm_classification_cli.py \
    --data data_OF.csv \
    --output of_results \
    --target Y \
    --features X1 X2 X3 X4 X5 X6 X7 \
    --class-order O F
```

Windows PowerShell equivalent:

```powershell
python .\obgm_classification_cli.py `
    --data .\data_OF.csv `
    --output .\of_results `
    --target Y `
    --features X1 X2 X3 X4 X5 X6 X7 `
    --class-order O F
```

### Multiclass classification

Example for classes `O`, `B`, `G`, and `M`:

```bash
python obgm_classification_cli.py \
    --data data_OBGM.csv \
    --output obgm_results \
    --target Y \
    --features X1 X2 X3 X4 X5 X6 X7 \
    --class-order O B G M
```

The `--class-order` option controls reporting and plotting order. If it is omitted, viClassify discovers and sorts the observed class labels.

## Command-line arguments

| Argument | Required | Default | Description |
|---|---:|---|---|
| `--data` | Yes | None | Path to the input CSV file. |
| `--output` | No | `classification_results` | Directory for generated results. |
| `--target` | No | `Y` | Target-column name. |
| `--features` | No | `X1 X2 X3 X4 X5 X6 X7` | Space-separated predictor-column names. |
| `--class-order` | No | Detected automatically | Preferred class order in reports and plots. |
| `--test-fraction` | No | `0.20` | Final fraction of rows used for chronological holdout evaluation. |
| `--cv-folds` | No | `5` | Maximum number of stratified cross-validation folds. |
| `--random-state` | No | `42` | Seed used for reproducible splitting and model fitting. |
| `--n-jobs` | No | `-1` | Parallel jobs. `-1` uses all available processors. |
| `--permutation-repeats` | No | `20` | Permutation-importance repetitions. |
| `--dpi` | No | `150` | Resolution of generated plots. |

The effective number of cross-validation folds is automatically limited by the number of observations in the rarest class. At least two observations are required in every class for stratified cross-validation.

## Analysis workflow

viClassify performs the following steps:

1. Loads the CSV file.
2. Validates the requested target and feature columns.
3. Preserves original row order in a generated `Sequence` column.
4. Converts predictors to numerical values.
5. Audits missing values and exact duplicate rows.
6. Generates descriptive statistics.
7. Calculates Pearson and Spearman correlation matrices.
8. Builds preprocessing and classifier pipelines.
9. Tunes each model family with stratified cross-validation.
10. Refits and exports the best configuration from every model family.
11. Selects and exports the best overall model.
12. Evaluates tuned configurations on the final chronological segment.
13. Exports Logistic Regression coefficients and intercepts.
14. Calculates full-data permutation importance for the best overall model.
15. Writes run configuration and package-version metadata.

## Model families and tuning grids

### Logistic Regression

- Balanced class weights
- `lbfgs` solver
- Median imputation
- Standard scaling
- Tuned inverse regularization strength: `C = 0.1, 1.0, 10.0`

### Random Forest

- Balanced subsample weights
- Median imputation
- Tuned estimators: `300, 600`
- Tuned minimum leaf size: `1, 2, 5`
- Tuned maximum features: `sqrt`, all features

### Extra Trees

- Balanced class weights
- Median imputation
- Tuned estimators: `300, 600`
- Tuned minimum leaf size: `1, 2, 5`
- Tuned maximum features: `sqrt`, all features

### Histogram Gradient Boosting

- Median imputation
- Tuned learning rate: `0.05, 0.10`
- Tuned iterations: `200, 400`
- Tuned maximum leaf nodes: `15, 31`
- Tuned L2 regularization: `0.0, 1.0`

## Logistic Regression intercept β0

viClassify fits Logistic Regression with an intercept and exports it with the feature coefficients to:

```text
logistic_coefficients_and_intercepts.csv
```

It also writes equation instructions to:

```text
logistic_equation_usage.json
```

### Binary model

For internal classes `C0` and `C1`, scikit-learn stores one coefficient vector. viClassify labels it as `C1 relative to C0`.

The standardized linear score is:

```text
eta = beta_0 + beta_1*z_1 + ... + beta_p*z_p
```

The probabilities are:

```text
P(C1) = 1 / (1 + exp(-eta))
P(C0) = 1 - P(C1)
```

Here, `beta_0` is the fitted intercept and each `z_j` is the value produced by the fitted preprocessing pipeline after median imputation and standardization.

### Multiclass model

For class `k`, the linear score is:

```text
eta_k = beta_0_k + beta_k1*z_1 + ... + beta_kp*z_p
```

The softmax probability is:

```text
P(class k) = exp(eta_k) / sum_l(exp(eta_l))
```

For numerical stability, subtract the largest score before exponentiation when calculating softmax manually.

### Recommended use

Use the exported pipeline for prediction instead of manually reconstructing the equation. The pipeline applies exactly the same feature order, missing-value imputation, scaling, and probability transformation used during training.

## Correlation matrices

The requested Pearson outputs are:

```text
pearson_correlation_matrix.csv
pearson_correlation_matrix.png
```

Pearson correlation summarizes linear association between numerical predictors.

The program also exports Spearman rank correlation:

```text
spearman_correlation_matrix.csv
spearman_correlation_matrix.png
```

Correlation does not establish causation. Strongly correlated predictors may also make individual Logistic Regression coefficients less stable or harder to interpret conditionally.

## Output files

A typical output directory contains:

```text
classification_results/
├── class_distribution.csv
├── numerical_summary.csv
├── missing_values.csv
├── duplicate_rows.csv
├── pearson_correlation_matrix.csv
├── pearson_correlation_matrix.png
├── spearman_correlation_matrix.csv
├── spearman_correlation_matrix.png
├── cv_results_logistic_regression.csv
├── cv_results_random_forest.csv
├── cv_results_extra_trees.csv
├── cv_results_histogram_gradient_boosting.csv
├── model_comparison.csv
├── best_logistic_regression.joblib
├── best_random_forest.joblib
├── best_extra_trees.joblib
├── best_histogram_gradient_boosting.joblib
├── best_overall_model.joblib
├── chronological_holdout_metrics.csv
├── holdout_report_logistic_regression.csv
├── holdout_report_random_forest.csv
├── holdout_report_extra_trees.csv
├── holdout_report_histogram_gradient_boosting.csv
├── holdout_predictions_logistic_regression.csv
├── holdout_predictions_random_forest.csv
├── holdout_predictions_extra_trees.csv
├── holdout_predictions_histogram_gradient_boosting.csv
├── holdout_confusion_matrix_logistic_regression.png
├── holdout_confusion_matrix_random_forest.png
├── holdout_confusion_matrix_extra_trees.png
├── holdout_confusion_matrix_histogram_gradient_boosting.png
├── logistic_coefficients_and_intercepts.csv
├── logistic_equation_usage.json
├── permutation_importance_full_data.csv
└── run_metadata.json
```

## Loading an exported model

Each model artifact is a dictionary containing the fitted pipeline and its metadata.

```python
import joblib
import pandas as pd

artifact = joblib.load(
    "of_results/best_extra_trees.joblib"
)

model = artifact["model"]
features = artifact["features"]
classes = artifact["classes"]

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

prediction = model.predict(new_data[features])
probabilities = model.predict_proba(new_data[features])

print("Predicted class:", prediction[0])

for label, probability in zip(classes, probabilities[0]):
    print(f"P({label}) = {probability:.6f}")
```

## Inspecting β0 and the Logistic Regression coefficients

```python
import joblib

artifact = joblib.load(
    "of_results/best_logistic_regression.joblib"
)

pipeline = artifact["model"]
classifier = pipeline.named_steps["classifier"]

print("Classes:", classifier.classes_)
print("Intercept β0:", classifier.intercept_)
print("Coefficients:", classifier.coef_)
```

For a binary model, `classifier.intercept_` has one value and `classifier.coef_` has one row. For a multiclass model, the fitted classifier provides a score equation for each class.

## Evaluation outputs

The chronological holdout metrics include:

- Accuracy
- Balanced accuracy
- Macro precision
- Macro recall
- Macro F1
- Weighted F1
- Matthews correlation coefficient
- Log loss

Macro F1 gives every class equal weight and is therefore the model-selection metric used by viClassify.

The chronological holdout consists of the final fraction of observations in file order. This is useful when row order represents time, process sequence, or acquisition order. If rows belong to repeated physical cycles, subjects, specimens, batches, or acquisition runs, provide and use a genuine group identifier in a future grouped-validation implementation. Random stratified folds can otherwise place related observations in both training and validation partitions.

## Interpretation cautions

- Cross-validation performance can be optimistic when adjacent or related records are split across folds.
- The chronological holdout is meaningful only when CSV row order has scientific or operational meaning.
- Logistic coefficients describe conditional associations, not causal effects.
- The Logistic Regression coefficients operate on standardized predictors.
- Class weighting changes the fitted decision boundary and intercept.
- Pearson correlation measures linear association and may miss nonlinear dependence.
- Full-data permutation importance is descriptive because it is calculated on data used to fit the exported full-data model.
- A `.joblib` file should be loaded only when it comes from a trusted source.
- Reuse exported models with a compatible Python and scikit-learn environment. Package versions are recorded in `run_metadata.json`.

## Reproducibility

The default random seed is `42`. Change it with:

```bash
python obgm_classification_cli.py \
    --data data_OF.csv \
    --random-state 123
```

The generated `run_metadata.json` records:

- Input and output paths
- Feature and target names
- Class order
- Validation settings
- Random seed
- Python version
- NumPy version
- pandas version
- SciPy version
- scikit-learn version

## Troubleshooting

### Input file not found

Check the path supplied to `--data`:

```bash
python obgm_classification_cli.py --data ./data/data_OF.csv
```

### Required columns are missing

Display the CSV headers and ensure they match `--target` and `--features` exactly. Leading and trailing whitespace in column names is removed automatically.

### Target contains missing labels

Correct or remove records with missing or empty target values. viClassify does not impute class labels.

### Chronological training partition lacks a class

Reduce `--test-fraction`, reorganize the validation design, or inspect whether a class appears only near the end of the file. A classifier cannot learn a class that is absent from its training partition.

### Cross-validation cannot run

Every class must have at least two observations. The requested fold count is automatically reduced to the size of the rarest class.

### Execution is slow

The tree-model grids evaluate multiple combinations. Use all processors with:

```bash
python obgm_classification_cli.py \
    --data data_OF.csv \
    --n-jobs -1
```

Reducing the tuning grids in `build_models()` lowers computation at the cost of a narrower search.

## Suggested repository name

```text
viClassify
```

Suggested package and executable naming for a future packaged release:

```text
Package: viclassify
CLI command: viclassify
```

## License

No license has been selected in this repository template. Add a `LICENSE` file before distributing or accepting external contributions, and update this section with the selected license.

## Project status

viClassify currently provides a complete script-based CLI workflow. Possible future extensions include grouped cross-validation, nested cross-validation, calibration analysis, threshold optimization, YAML configuration, additional estimators, and packaging with a `pyproject.toml` console entry point.
