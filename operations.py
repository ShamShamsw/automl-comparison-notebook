"""
operations.py — Grid search, training orchestration, and result aggregation.
"""

from __future__ import annotations

import itertools
import time
from typing import Any

import numpy as np

from models import HYPERPARAMETER_GRIDS, create_model_config, evaluate_model, fit_model
from storage import (
    load_or_generate_dataset,
    save_benchmark_results,
    train_val_test_split,
)


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------


def _expand_grid(hyperparams_grid: dict[str, list]) -> list[dict[str, Any]]:
    """Expand a hyperparameter grid into a list of all combinations.

    Args:
        hyperparams_grid: Mapping of parameter name → list of candidate values.

    Returns:
        List of dictionaries, each representing one hyperparameter combination.
    """
    keys = list(hyperparams_grid.keys())
    values = list(hyperparams_grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def run_grid_search(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    model_names: list[str] | None = None,
    random_state: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run an exhaustive grid search over all registered models and hyperparameters.

    For every (model, hyperparameter) combination the function fits on the
    training set and evaluates on train, validation, and test splits.

    Args:
        X_train: Training feature matrix.
        X_val: Validation feature matrix.
        X_test: Test feature matrix.
        y_train: Training targets.
        y_val: Validation targets.
        y_test: Test targets.
        model_names: Optional list of model names to benchmark.  Defaults to
            all keys in ``HYPERPARAMETER_GRIDS``.
        random_state: Seed for reproducibility.
        verbose: Whether to print progress updates to stdout.

    Returns:
        Dictionary with keys:
          - 'all_results': list of per-configuration result dicts.
          - 'best_result': result dict for the top validation-accuracy model.
          - 'ranked_results': all_results sorted by test accuracy (descending).
    """
    if model_names is None:
        model_names = list(HYPERPARAMETER_GRIDS.keys())

    # Build full list of (model_name, hyperparams) combinations.
    configs: list[tuple[str, dict]] = []
    for name in model_names:
        for combo in _expand_grid(HYPERPARAMETER_GRIDS[name]):
            configs.append((name, combo))

    total = len(configs)
    all_results: list[dict[str, Any]] = []

    for idx, (model_name, hyperparams) in enumerate(configs, start=1):
        if verbose:
            bar_filled = int((idx - 1) / total * 10)
            bar = "█" * bar_filled + "░" * (10 - bar_filled)
            print(
                f"\r   [{bar}] {int((idx-1)/total*100):3d}% - "
                f"Training model {idx}/{total}...",
                end="",
                flush=True,
            )

        model = fit_model(model_name, X_train, y_train, hyperparams, random_state)

        train_metrics = evaluate_model(model, X_train, y_train)
        val_metrics = evaluate_model(model, X_val, y_val)
        test_metrics = evaluate_model(model, X_test, y_test)

        config = create_model_config(model_name, hyperparams)
        result = {
            "config": config,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "model_object": model,  # kept in memory, excluded from JSON
        }
        all_results.append(result)

    if verbose:
        print(f"\r   [{'█'*10}] 100% - Training complete.          ")

    # Sort by validation accuracy to find the best model.
    ranked = sorted(all_results, key=lambda r: r["test_metrics"]["accuracy"], reverse=True)
    best = max(all_results, key=lambda r: r["val_metrics"]["accuracy"])

    return {
        "all_results": all_results,
        "best_result": best,
        "ranked_results": ranked,
    }


# ---------------------------------------------------------------------------
# Core flow orchestrator
# ---------------------------------------------------------------------------


def run_core_flow(
    n_samples: int = 2000,
    n_features: int = 20,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
    csv_path: str | None = None,
    save_json: bool = True,
) -> dict[str, Any]:
    """Run the full benchmarking pipeline end-to-end.

    Args:
        n_samples: Number of samples for synthetic dataset generation.
        n_features: Number of features for synthetic dataset generation.
        val_size: Fraction of data reserved for validation.
        test_size: Fraction of data reserved for testing.
        random_state: Global random seed.
        csv_path: Optional path to a CSV file with custom data.
        save_json: Whether to persist results to ``data/runs/benchmark_report.json``.

    Returns:
        Summary dictionary containing dataset metadata, grid-search results, and
        the path to the saved JSON file (if *save_json* is True).
    """
    start_time = time.time()

    X, y, feature_names = load_or_generate_dataset(
        n_samples=n_samples,
        n_features=n_features,
        random_state=random_state,
        csv_path=csv_path,
    )

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, val_size=val_size, test_size=test_size, random_state=random_state
    )

    classes, counts = np.unique(y, return_counts=True)
    class_distribution = {
        int(cls): f"{cnt / len(y) * 100:.1f}%" for cls, cnt in zip(classes, counts)
    }

    dataset_profile = {
        "name": "Custom CSV" if csv_path else "Synthetic binary classification",
        "n_samples": len(y),
        "n_features": X.shape[1],
        "target_classes": classes.tolist(),
        "class_distribution": class_distribution,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
    }

    search_results = run_grid_search(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        random_state=random_state,
    )

    elapsed = time.time() - start_time

    # Build a JSON-serialisable summary (strip model objects).
    def _strip(result: dict) -> dict:
        return {k: v for k, v in result.items() if k != "model_object"}

    summary = {
        "dataset_profile": dataset_profile,
        "configuration": {
            "n_samples": n_samples,
            "n_features": n_features,
            "val_size": val_size,
            "test_size": test_size,
            "random_state": random_state,
            "search_type": "Grid search",
        },
        "best_result": _strip(search_results["best_result"]),
        "ranked_results": [_strip(r) for r in search_results["ranked_results"]],
        "elapsed_seconds": round(elapsed, 1),
    }

    json_path: str | None = None
    if save_json:
        json_path = save_benchmark_results(summary)
        summary["json_path"] = json_path

    # Attach in-memory objects needed for plotting (not JSON-serialisable).
    summary["_search_results"] = search_results
    summary["_splits"] = (X_train, X_val, X_test, y_train, y_val, y_test)

    return summary
