# Beginner Project 21: AutoML Comparison Notebook

**Time:** 4-6 hours  
**Difficulty:** Intermediate Beginner  
**Focus:** Model benchmarking, hyperparameter tuning, train/validation/test workflows, cross-validation, and performance metric tracking

---

## Why This Project?

Machine learning in the real world is not about building one perfect model—it's about **systematically comparing models and hyperparameter choices** to find what works best for your data.

This project teaches you a professional ML workflow where you can:

- generate or load tabular datasets with train/validation/test splits,
- build wrappers for multiple model types (e.g., logistic regression, decision trees, random forests, SVM),
- run hyperparameter grids (grid search) or random search to find good settings,
- track and compare model performance across train, validation, and test sets,
- visualize metric trends and confusion matrices to understand model behavior,
- save benchmarking results and comparison reports for reproducibility and analysis.

---

## More Projects

You can access this project and more in this separate repository:

[student-interview-prep](https://github.com/ShamShamsw/student-interview-prep.git)

---

## What You Will Build

You will build a command-line AutoML benchmark system that:

1. Generates or loads a tabular dataset (e.g., binary classification) with train/val/test splits.
2. Defines a modular model interface (fit, predict, score) for multiple algorithms.
3. Implements model classes for at least 3 algorithms (e.g., LogisticRegression, DecisionTree, RandomForest).
4. Builds a hyperparameter grid or random search that trains models with different settings.
5. Evaluates each trained model on train, validation, and test sets with metrics (accuracy, precision, recall, F1, AUC).
6. Tracks the winning model and its hyperparameters.
7. Generates a comparison report showing all models ranked by test performance.
8. Saves a JSON summary of the benchmarking session including all configurations, results, and the best model metadata.
9. Visualizes metric trends across epochs/folds and confusion matrices for the best model.

---

## Requirements

- Python 3.11+
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

Install with:

```bash
pip install -r requirements.txt
```

---

## Example Session

```text
======================================================================
   AUTOML COMPARISON NOTEBOOK - TABULAR BENCHMARKING SUITE
======================================================================

Configuration:
   Dataset: Synthetic binary classification (2000 samples, 20 features)
   Train/validation/test split: 60% / 20% / 20%
   Random seed: 42
   Search type: Grid search
   CV folds: 5

Models to compare:
   1. LogisticRegression
   2. DecisionTreeClassifier
   3. RandomForestClassifier

Hyperparameter grids:
   LogisticRegression:
      - C: [0.001, 0.01, 0.1, 1.0, 10.0]
      - penalty: ['l2']
   DecisionTreeClassifier:
      - max_depth: [3, 5, 7, 10]
      - min_samples_split: [2, 5]
   RandomForestClassifier:
      - n_estimators: [50, 100, 200]
      - max_depth: [5, 10, None]

Benchmark Progress:
   [████████░░] 80% - Training model 16/20...
   
Dataset profile:
   Name: Synthetic binary classification
   Samples: 2000
   Features: 20
   Target classes: [0, 1]
   Class distribution: {0: 48.5%, 1: 51.5%}
   Train samples: 1200
   Validation samples: 400
   Test samples: 400

Benchmark Results:
   ╔═══════════════════════════════════════════════════════════════╗
   ║  Model                      │  Val Acc  │  Test Acc │ F1-Score  ║
   ╠═══════════════════════════════════════════════════════════════╣
   ║  1. RandomForest (Best)     │  0.882   │  0.879   │  0.877    ║
   ║     Hyperparams: n_est=200, depth=10                          ║
   ║  2. LogisticRegression      │  0.861   │  0.858   │  0.854    ║
   ║     Hyperparams: C=1.0, penalty=l2                            ║
   ║  3. DecisionTree            │  0.795   │  0.791   │  0.785    ║
   ║     Hyperparams: depth=7, min_split=2                         ║
   ╚═══════════════════════════════════════════════════════════════╝

Best Model Summary:
   Algorithm: RandomForestClassifier
   Best hyperparameters: {'n_estimators': 200, 'max_depth': 10}
   
   Training metrics:
      Accuracy: 0.898
      Precision: 0.896
      Recall: 0.901
      F1-Score: 0.898
      AUC-ROC: 0.950
   
   Validation metrics:
      Accuracy: 0.882
      Precision: 0.879
      Recall: 0.885
      F1-Score: 0.877
      AUC-ROC: 0.928
   
   Test metrics (final):
      Accuracy: 0.879
      Precision: 0.876
      Recall: 0.882
      F1-Score: 0.877
      AUC-ROC: 0.925

Artifacts saved:
   - Benchmarking report: data/runs/benchmark_report.json
   - Metrics comparison plot: data/runs/model_comparison.png
   - Confusion matrix (best model): data/runs/confusion_matrix.png
   - Learning curves: data/runs/learning_curves.png

Session completed in 42.3 seconds.
```

---

## Build Order

Follow this order for clean architecture:

1. `storage.py` — Load/save datasets and results JSON
2. `models.py` — Define model wrappers and configuration
3. `operations.py` — Implement grid search, training, evaluation
4. `display.py` — Format tables, plots, and reports
5. `main.py` — Orchestrate the full benchmark workflow

---

## Step-by-Step Instructions

### Step 1: Implement `storage.py`

Create functions to:
- Load or generate a synthetic tabular dataset (use `sklearn.datasets.make_classification`)
- Split into train/val/test (60/20/20 or configurable)
- Save and load results as JSON
- Ensure `data/` directory exists

**Key functions:**
- `load_or_generate_dataset()` → returns X, y, feature_names
- `train_val_test_split()` → returns X_train, X_val, X_test, y_train, y_val, y_test
- `save_benchmark_results()` → saves JSON with all model configs and metrics
- `load_benchmark_results()` → loads previously saved results

### Step 2: Implement `models.py`

Define model classes and configuration:
- Base model interface (abstract or minimal)
- Wrapper classes for LogisticRegression, DecisionTree, RandomForest
- Each wrapper implements: `fit()`, `predict()`, `predict_proba()`, `score()`
- Configuration dataclass/dict for each model with hyperparameter grid

**Key structures:**
- `create_model_config(model_name: str, hyperparams: dict) → dict`
- `fit_model(model_name: str, X_train, y_train, hyperparams) → model`
- `evaluate_model(model, X, y) → metrics_dict` (accuracy, precision, recall, F1, AUC)

### Step 3: Implement `operations.py`

Build the core benchmark workflow:
- Grid search over model types and hyperparameters
- For each configuration: fit on train, evaluate on train/val/test
- Track best model and all results
- Generate comparison rankings

**Key functions:**
- `run_grid_search(X_train, X_val, X_test, y_train, y_val, y_test) → results`
- `run_core_flow() → summary` (orchestrates full pipeline)

### Step 4: Implement `display.py`

Format output for readability:
- Header with configuration summary
- Progress indicator (simple text or progress bar)
- Comparison table (model name, metrics)
- Confusion matrix ASCII art or saved plot
- Final summary with best model details

**Key functions:**
- `format_header() → str`
- `format_config_summary(config) → str`
- `format_results_table(results) → str`
- `format_metrics_report(best_model_metrics) → str`

### Step 5: Implement `main.py`

Orchestrate the workflow:
1. Load or generate dataset
2. Print configuration and dataset profile
3. Run benchmark (grid search)
4. Print best model and comparison table
5. Save results to JSON
6. Generate and save plots

---

## Requirements.txt

Update these dependencies:

```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

---

## Done Criteria

- [ ] The project runs from `main.py` without crashes.
- [ ] At least 3 different model types are benchmarked.
- [ ] Grid search tries at least 12 different hyperparameter configurations.
- [ ] Train/validation/test metrics are tracked separately.
- [ ] A JSON summary file is saved with all results.
- [ ] A comparison table shows all models ranked by test performance.
- [ ] The best model's metrics are displayed clearly.
- [ ] Code is organized by concern (models, operations, display, storage).
- [ ] Every function has a docstring.

---

## Stretch Goals

1. Implement random search instead of (or in addition to) grid search.
2. Add cross-validation (k-fold CV) for more robust validation metrics.
3. Visualize learning curves (train/val loss vs. epoch) for models that support it.
4. Plot confusion matrices for the best model.
5. Export a detailed PDF report with all metrics and visualizations.
6. Allow users to load their own CSV dataset instead of synthetic.
7. Add feature importance plots for tree-based models.
8. Implement early stopping if val performance plateaus.
