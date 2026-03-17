"""
storage.py — Dataset loading/generation, train/val/test splits, and JSON persistence.
"""

import json
import os

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "runs")


def _ensure_data_dir() -> None:
    """Create the data/runs directory if it does not already exist."""
    os.makedirs(DATA_DIR, exist_ok=True)


def load_or_generate_dataset(
    n_samples: int = 2000,
    n_features: int = 20,
    n_informative: int = 10,
    n_redundant: int = 5,
    random_state: int = 42,
    csv_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a CSV dataset or generate a synthetic binary classification dataset.

    Args:
        n_samples: Number of samples to generate (ignored when csv_path is given).
        n_features: Total number of features (ignored when csv_path is given).
        n_informative: Number of informative features.
        n_redundant: Number of redundant features.
        random_state: Random seed for reproducibility.
        csv_path: Optional path to a CSV file.  The last column is treated as
            the target; all other columns are features.

    Returns:
        Tuple of (X, y, feature_names) where X is a 2-D float array,
        y is a 1-D integer array, and feature_names is a list of strings.
    """
    if csv_path is not None:
        import pandas as pd

        df = pd.read_csv(csv_path)
        feature_names = list(df.columns[:-1])
        X = df.iloc[:, :-1].to_numpy(dtype=float)
        y = df.iloc[:, -1].to_numpy(dtype=int)
        return X, y, feature_names

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=2,
        random_state=random_state,
    )
    feature_names = [f"feature_{i}" for i in range(n_features)]
    return X, y, feature_names


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into train, validation, and test subsets.

    Args:
        X: Feature matrix.
        y: Target array.
        val_size: Fraction of the full dataset to use for validation.
        test_size: Fraction of the full dataset to use for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    relative_test = test_size / (1.0 - val_size)
    X_temp, X_val, y_temp, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_state
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_benchmark_results(results: dict, filename: str = "benchmark_report.json") -> str:
    """Persist the benchmark results dictionary to a JSON file.

    Args:
        results: Dictionary containing all model configurations and metrics.
        filename: Base filename (placed inside DATA_DIR).

    Returns:
        Absolute path to the saved file.
    """
    _ensure_data_dir()
    path = os.path.join(DATA_DIR, filename)

    # Convert numpy scalar types to plain Python types for JSON serialisation.
    def _to_serialisable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=_to_serialisable)
    return path


def load_benchmark_results(filename: str = "benchmark_report.json") -> dict:
    """Load previously saved benchmark results from a JSON file.

    Args:
        filename: Base filename (resolved relative to DATA_DIR).

    Returns:
        Dictionary with all stored benchmark data.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
