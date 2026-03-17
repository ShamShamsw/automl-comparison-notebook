"""
display.py — Terminal output formatting and matplotlib/seaborn visualisations.
"""

from __future__ import annotations

import os
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe in all environments
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from storage import DATA_DIR


# ---------------------------------------------------------------------------
# Text formatting helpers
# ---------------------------------------------------------------------------


def format_header() -> str:
    """Return the application banner string.

    Returns:
        Multi-line string with the AUTOML banner.
    """
    line = "=" * 70
    title = "   AUTOML COMPARISON NOTEBOOK - TABULAR BENCHMARKING SUITE"
    return f"\n{line}\n{title}\n{line}\n"


def format_config_summary(config: dict[str, Any]) -> str:
    """Format configuration and hyperparameter grid information for display.

    Args:
        config: Dictionary with keys 'n_samples', 'n_features', 'val_size',
            'test_size', 'random_state', and 'search_type'.

    Returns:
        Human-readable configuration block.
    """
    from models import HYPERPARAMETER_GRIDS

    lines = ["Configuration:"]
    lines.append(
        f"   Dataset: Synthetic binary classification "
        f"({config['n_samples']} samples, {config['n_features']} features)"
    )
    train_pct = int((1 - config["val_size"] - config["test_size"]) * 100)
    val_pct = int(config["val_size"] * 100)
    test_pct = int(config["test_size"] * 100)
    lines.append(
        f"   Train/validation/test split: {train_pct}% / {val_pct}% / {test_pct}%"
    )
    lines.append(f"   Random seed: {config['random_state']}")
    lines.append(f"   Search type: {config['search_type']}")
    lines.append("")
    lines.append("Models to compare:")
    for i, name in enumerate(HYPERPARAMETER_GRIDS, start=1):
        lines.append(f"   {i}. {name}")
    lines.append("")
    lines.append("Hyperparameter grids:")
    for name, grid in HYPERPARAMETER_GRIDS.items():
        lines.append(f"   {name}:")
        for param, values in grid.items():
            lines.append(f"      - {param}: {values}")
    return "\n".join(lines)


def format_dataset_profile(profile: dict[str, Any]) -> str:
    """Format the dataset profile block.

    Args:
        profile: Dictionary with dataset statistics.

    Returns:
        Human-readable dataset profile block.
    """
    dist_str = ", ".join(
        f"{k}: {v}" for k, v in profile["class_distribution"].items()
    )
    lines = [
        "\nDataset profile:",
        f"   Name: {profile['name']}",
        f"   Samples: {profile['n_samples']}",
        f"   Features: {profile['n_features']}",
        f"   Target classes: {profile['target_classes']}",
        f"   Class distribution: {{{dist_str}}}",
        f"   Train samples: {profile['n_train']}",
        f"   Validation samples: {profile['n_val']}",
        f"   Test samples: {profile['n_test']}",
    ]
    return "\n".join(lines)


def format_results_table(ranked_results: list[dict[str, Any]], best_result: dict[str, Any] | None = None) -> str:
    """Format the benchmark results comparison table.

    Args:
        ranked_results: List of result dicts sorted by test accuracy (best first).
        best_result: Optional result dict for the model chosen as overall best
            (by validation accuracy).  When provided the "(Best)" suffix is
            applied to the matching row; otherwise it defaults to rank 1.

    Returns:
        Multi-line string containing the formatted comparison table.
    """
    border = "=" * 65
    header = f"{'Model':<32} {'Val Acc':>8} {'Test Acc':>9} {'F1-Score':>9}"
    lines = [
        "\nBenchmark Results:",
        f"   ╔{border}╗",
        f"   ║  {header}  ║",
        f"   ╠{border}╣",
    ]

    # Determine which config is the "best" model for labelling purposes.
    best_config = best_result["config"] if best_result else ranked_results[0]["config"]

    for rank, res in enumerate(ranked_results, start=1):
        name = res["config"]["model_name"]
        hp = res["config"]["hyperparams"]
        val_acc = res["val_metrics"]["accuracy"]
        test_acc = res["test_metrics"]["accuracy"]
        f1 = res["test_metrics"]["f1"]
        is_best = (
            res["config"]["model_name"] == best_config["model_name"]
            and res["config"]["hyperparams"] == best_config["hyperparams"]
        )
        suffix = " (Best)" if is_best else ""
        short_name = _shorten_name(name) + suffix
        lines.append(
            f"   ║  {rank}. {short_name:<28} {val_acc:>8.3f} {test_acc:>9.3f} {f1:>9.3f}  ║"
        )
        hp_str = _format_hyperparams(hp)
        lines.append(f"   ║     Hyperparams: {hp_str:<47}║")
    lines.append(f"   ╚{border}╝")
    return "\n".join(lines)


def format_metrics_report(result: dict[str, Any]) -> str:
    """Format the best model summary block.

    Args:
        result: A single result dictionary (the best model entry).

    Returns:
        Human-readable best model metrics report.
    """
    name = result["config"]["model_name"]
    hp = result["config"]["hyperparams"]
    lines = [
        "\nBest Model Summary:",
        f"   Algorithm: {name}",
        f"   Best hyperparameters: {hp}",
    ]
    for split_label, key in [("Training", "train_metrics"), ("Validation", "val_metrics"), ("Test (final)", "test_metrics")]:
        m = result[key]
        lines.append(f"\n   {split_label} metrics:")
        lines.append(f"      Accuracy:  {m['accuracy']:.3f}")
        lines.append(f"      Precision: {m['precision']:.3f}")
        lines.append(f"      Recall:    {m['recall']:.3f}")
        lines.append(f"      F1-Score:  {m['f1']:.3f}")
        lines.append(f"      AUC-ROC:   {m['auc']:.3f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _shorten_name(name: str) -> str:
    """Return an abbreviated model name for compact table display."""
    mapping = {
        "LogisticRegression": "LogisticRegression",
        "DecisionTreeClassifier": "DecisionTree",
        "RandomForestClassifier": "RandomForest",
    }
    return mapping.get(name, name)


def _format_hyperparams(hp: dict) -> str:
    """Return a compact one-line hyperparameter string."""
    parts = []
    abbrev = {
        "n_estimators": "n_est",
        "max_depth": "depth",
        "min_samples_split": "min_split",
        "penalty": "penalty",
        "C": "C",
    }
    for k, v in hp.items():
        short_key = abbrev.get(k, k)
        parts.append(f"{short_key}={v}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def save_model_comparison_plot(
    ranked_results: list[dict[str, Any]],
    output_path: str | None = None,
) -> str:
    """Save a bar chart comparing test accuracy across all model configurations.

    Args:
        ranked_results: List of result dicts sorted by test accuracy (best first).
        output_path: Optional file path for the PNG.  Defaults to
            ``data/runs/model_comparison.png``.

    Returns:
        Absolute path to the saved PNG file.
    """
    if output_path is None:
        output_path = os.path.join(DATA_DIR, "model_comparison.png")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    labels = []
    val_accs = []
    test_accs = []
    for r in ranked_results:
        short = _shorten_name(r["config"]["model_name"])
        hp_str = _format_hyperparams(r["config"]["hyperparams"])
        labels.append(f"{short}\n{hp_str}")
        val_accs.append(r["val_metrics"]["accuracy"])
        test_accs.append(r["test_metrics"]["accuracy"])

    x = np.arange(len(labels))
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.6), 6))
    ax.bar(x - width / 2, val_accs, width, label="Val Accuracy", color="#4C72B0")
    ax.bar(x + width / 2, test_accs, width, label="Test Accuracy", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Comparison — Val vs Test Accuracy")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    return output_path


def save_confusion_matrix_plot(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    output_path: str | None = None,
) -> str:
    """Save a confusion matrix heatmap for the best model on the test set.

    Args:
        model: A fitted sklearn estimator.
        X_test: Test feature matrix.
        y_test: True test labels.
        model_name: Display name used in the plot title.
        output_path: Optional file path for the PNG.  Defaults to
            ``data/runs/confusion_matrix.png``.

    Returns:
        Absolute path to the saved PNG file.
    """
    if output_path is None:
        output_path = os.path.join(DATA_DIR, "confusion_matrix.png")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=True)
    ax.set_title(f"Confusion Matrix — {_shorten_name(model_name)} (Test set)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    return output_path


def save_learning_curves_plot(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str,
    output_path: str | None = None,
    cv: int = 5,
    random_state: int = 42,
) -> str:
    """Save learning curves (train/val score vs training set size) for a model.

    Args:
        model: A fitted sklearn estimator (used as the estimator template).
        X_train: Training feature matrix.
        y_train: Training targets.
        model_name: Display name used in the plot title.
        output_path: Optional file path for the PNG.  Defaults to
            ``data/runs/learning_curves.png``.
        cv: Number of cross-validation folds.
        random_state: Seed for reproducibility (passed to ShuffleSplit).

    Returns:
        Absolute path to the saved PNG file.
    """
    from sklearn.model_selection import learning_curve, ShuffleSplit

    if output_path is None:
        output_path = os.path.join(DATA_DIR, "learning_curves.png")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cv_splitter = ShuffleSplit(n_splits=cv, test_size=0.2, random_state=random_state)
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X_train,
        y_train,
        cv=cv_splitter,
        scoring="accuracy",
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_mean, "o-", color="#4C72B0", label="Train score")
    ax.fill_between(
        train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#4C72B0"
    )
    ax.plot(train_sizes, val_mean, "o-", color="#DD8452", label="CV val score")
    ax.fill_between(
        train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="#DD8452"
    )
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Learning Curves — {_shorten_name(model_name)}")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    return output_path
