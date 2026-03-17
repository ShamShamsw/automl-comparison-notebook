"""
models.py — Model wrappers and evaluation utilities for the AutoML benchmark.

Supported algorithms:
  - LogisticRegression
  - DecisionTreeClassifier
  - RandomForestClassifier
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------

HYPERPARAMETER_GRIDS: dict[str, dict[str, list]] = {
    "LogisticRegression": {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0],
    },
    "DecisionTreeClassifier": {
        "max_depth": [3, 5, 7, 10],
        "min_samples_split": [2, 5],
    },
    "RandomForestClassifier": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, None],
    },
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def create_model_config(model_name: str, hyperparams: dict[str, Any]) -> dict[str, Any]:
    """Return a configuration dictionary describing a model and its hyperparameters.

    Args:
        model_name: One of 'LogisticRegression', 'DecisionTreeClassifier',
            or 'RandomForestClassifier'.
        hyperparams: Dictionary of hyperparameter name → value.

    Returns:
        Configuration dictionary with keys 'model_name' and 'hyperparams'.
    """
    return {"model_name": model_name, "hyperparams": hyperparams}


def fit_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    hyperparams: dict[str, Any],
    random_state: int = 42,
) -> Any:
    """Instantiate and fit a model with the given hyperparameters.

    Args:
        model_name: Algorithm identifier string.
        X_train: Training feature matrix.
        y_train: Training target array.
        hyperparams: Hyperparameter dictionary passed to the sklearn estimator.
        random_state: Seed for reproducibility (applied where the estimator
            accepts it).

    Returns:
        A fitted sklearn estimator.

    Raises:
        ValueError: If *model_name* is not recognised.
    """
    constructors: dict[str, type] = {
        "LogisticRegression": LogisticRegression,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "RandomForestClassifier": RandomForestClassifier,
    }
    if model_name not in constructors:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(constructors)}"
        )

    cls = constructors[model_name]
    # Inject random_state only for estimators that support it.
    init_params = dict(hyperparams)
    if model_name in ("LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier"):
        init_params.setdefault("random_state", random_state)
    if model_name == "LogisticRegression":
        init_params.setdefault("max_iter", 1000)

    model = cls(**init_params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: Any, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Compute classification metrics for a fitted model on a dataset split.

    Args:
        model: A fitted sklearn estimator with ``predict`` and
            ``predict_proba`` methods.
        X: Feature matrix.
        y: True target labels.

    Returns:
        Dictionary with keys 'accuracy', 'precision', 'recall', 'f1', 'auc'.
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y, y_proba)),
    }
