"""
storage.py - Persistence layer for Project 21: AutoML Comparison Notebook
==========================================================================

Handles:
    - Dataset loading/generation and train/val/test splitting
    - Saving and loading benchmark results as JSON
    - File I/O and directory management
"""

from pathlib import Path
import json
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent / 'data'
RUNS_DIR = DATA_DIR / 'runs'


def ensure_data_dir() -> None:
    """Create local data and runs directories if they do not exist.
    
    Returns:
        None
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def load_or_generate_dataset(
    n_samples: int = 2000,
    n_features: int = 20,
    n_informative: int = 15,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Load or generate a synthetic tabular binary classification dataset.
    
    Parameters:
        n_samples (int): Total number of samples to generate.
        n_features (int): Total number of features.
        n_informative (int): Number of informative features.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        Tuple[np.ndarray, np.ndarray, list]: (X, y, feature_names)
            - X: Feature matrix of shape (n_samples, n_features)
            - y: Target vector of shape (n_samples,)
            - feature_names: List of feature names
    
    TODO: Implement dataset generation using sklearn.datasets.make_classification
    """
    pass


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_size: float = 0.6,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split dataset into train/validation/test sets.
    
    Parameters:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        train_size (float): Proportion for training (0.0-1.0).
        val_size (float): Proportion for validation (0.0-1.0).
        random_state (int): Random seed.
    
    Returns:
        Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    
    Design note:
        Using sklearn's train_test_split twice (first to isolate test set,
        then to split remaining into train/val) ensures clean, stratified splits.
        Alternative considered: manual numpy splitting — rejected because
        it loses sklearn's stratification benefits.
    
    TODO: Implement three-way split using sklearn.model_selection.train_test_split
    """
    pass


def save_benchmark_results(filename: str, results: dict) -> None:
    """Save benchmark results to JSON in the runs directory.
    
    Parameters:
        filename (str): Output filename (e.g., 'benchmark_results.json').
        results (dict): Benchmark results dictionary including:
            - config: Benchmark configuration
            - dataset_profile: Dataset metadata
            - model_results: List of model results with metrics
            - best_model: Metadata for best performing model
            - timestamp: ISO-8601 timestamp
    
    Returns:
        None
    
    TODO: Implement JSON serialization with pretty printing
    """
    pass


def load_benchmark_results(filename: str) -> dict:
    """Load previously saved benchmark results from JSON.
    
    Parameters:
        filename (str): Input filename.
    
    Returns:
        dict: Benchmark results, or empty dict if file not found.
    
    TODO: Implement JSON deserialization with error handling
    """
    pass


def load_json(filename: str):
    """Load JSON data from the local data directory.
    
    Parameters:
        filename (str): Filename in data directory.
    
    Returns:
        Loaded JSON data, or empty list if file not found.
    
    TODO: Implement generic JSON loading
    """
    ensure_data_dir()
    path = DATA_DIR / filename
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding='utf-8'))


def save_json(filename: str, data) -> None:
    """Save JSON data to the local data directory.
    
    Parameters:
        filename (str): Filename in data directory.
        data: Data to serialize (must be JSON-serializable).
    
    Returns:
        None
    
    TODO: Implement generic JSON saving
    """
    ensure_data_dir()
    path = DATA_DIR / filename
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')
