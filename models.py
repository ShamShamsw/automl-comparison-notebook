"""
models.py - Data models for Project 21: AutoML Comparison Notebook
==================================================================

Defines:
    - Model wrapper classes (LogisticRegression, DecisionTree, RandomForest)
    - Hyperparameter grid configurations for grid search
    - Configuration dataclasses for the benchmark session
"""

from datetime import datetime
from typing import Dict, List, Any


def _utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp.
    
    Returns:
        str: Timestamp string with trailing Z.
    """
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def create_project_config(
    n_models: int = 3,
    cv_folds: int = 5,
    random_state: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> Dict[str, Any]:
    """Create the default runtime configuration for AutoML benchmarking.
    
    Parameters:
        n_models (int): Number of models to compare.
        cv_folds (int): Number of cross-validation folds.
        random_state (int): Random seed for reproducibility.
        test_size (float): Proportion of data for test set.
        val_size (float): Proportion of data for validation set.
    
    Returns:
        dict: Configuration dictionary with all benchmark settings.
    
    TODO: Implement configuration creation
    """
    pass


def create_model_instance(model_name: str, hyperparams: Dict[str, Any]):
    """Create a scikit-learn model instance with given hyperparameters.
    
    Parameters:
        model_name (str): Name of model ('LogisticRegression', 'DecisionTree', 'RandomForest').
        hyperparams (dict): Hyperparameter dictionary for the model.
    
    Returns:
        Model instance (sklearn classifier).
    
    Design note:
        This factory function abstracts model instantiation, making it easy to
        add new models or change default parameters without touching the
        search/evaluation code. Alternative: hardcode model creation in
        search loop — rejected because it's less maintainable.
    
    TODO: Implement model factory supporting at least 3 algorithms
    """
    pass


def create_hyperparameter_grids() -> Dict[str, List[Dict[str, Any]]]:
    """Define hyperparameter grids for grid search over all models.
    
    Returns:
        dict: Mapping of model_name to list of hyperparameter dicts.
            Example: {
                'LogisticRegression': [
                    {'C': 0.001, 'penalty': 'l2'},
                    {'C': 0.01, 'penalty': 'l2'},
                    ...
                ],
                'DecisionTree': [...],
                'RandomForest': [...]
            }
    
    Why grids instead of random search:
        Grids are deterministic and easier to understand/debug for beginners.
        Random search is slightly better in practice but adds complexity.
    
    TODO: Implement grids with at least 4 options per model
    """
    pass


def create_model_result_record(
    model_name: str,
    hyperparams: Dict[str, Any],
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
) -> Dict[str, Any]:
    """Create a record for one trained model's results.
    
    Parameters:
        model_name (str): Name of the model.
        hyperparams (dict): Hyperparameters used for training.
        train_metrics (dict): Metrics on training set.
        val_metrics (dict): Metrics on validation set.
        test_metrics (dict): Metrics on test set.
    
    Returns:
        dict: Complete result record containing all metadata and metrics.
    
    TODO: Implement record creation
    """
    pass


def create_benchmark_summary(
    config: Dict[str, Any],
    dataset_profile: Dict[str, Any],
    model_results: List[Dict[str, Any]],
    best_model_idx: int,
    execution_time_seconds: float,
) -> Dict[str, Any]:
    """Create a complete summary of the benchmark session.
    
    Parameters:
        config (dict): Benchmark configuration.
        dataset_profile (dict): Dataset metadata.
        model_results (list): Results for all trained models.
        best_model_idx (int): Index of best model in model_results.
        execution_time_seconds (float): Total runtime.
    
    Returns:
        dict: Comprehensive benchmark summary for serialization/reporting.
    
    TODO: Implement summary creation
    """
    pass
