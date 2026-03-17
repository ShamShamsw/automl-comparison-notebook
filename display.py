"""
display.py - Presentation helpers for Project 21: AutoML Comparison Notebook
============================================================================

Formats:
    - Benchmark session headers and configuration
    - Results tables and comparisons
    - Performance reports
    - Progress indicators
"""

from typing import Dict, List, Any


def format_header() -> str:
    """Return a formatted header for the benchmark session.
    
    Returns:
        str: ASCII header with title and separator lines.
    
    Design note:
        A consistent header makes output professional and helps users
        understand what program is running. Could also be used for logging.
    
    TODO: Implement header with decorative separators
    """
    pass


def format_startup_guide(config: Dict[str, Any], profile: Dict[str, Any]) -> str:
    """Format the startup/configuration summary.
    
    Parameters:
        config (dict): Benchmark configuration (from models.create_project_config).
        profile (dict): Dataset profile (from operations.load_dataset_profile).
    
    Returns:
        str: Formatted configuration and dataset information ready to print.
    
    TODO: Implement configuration display with indentation
    """
    pass


def format_results_table(results: List[Dict[str, Any]]) -> str:
    """Format a comparison table of all benchmarked models.
    
    Parameters:
        results (list): List of model result dicts, sorted by performance.
    
    Returns:
        str: ASCII table with columns:
            - Rank
            - Model Name
            - Best Hyperparameters
            - Val Accuracy
            - Test Accuracy
            - F1-Score
    
    Design note:
        ASCII tables are readable in terminals without external dependencies.
        Alternative: pandas DataFrame display — rejected for this project
        to keep dependencies minimal.
    
    TODO: Implement table formatting with aligned columns
    """
    pass


def format_best_model_report(best_result: Dict[str, Any]) -> str:
    """Format a detailed report for the best performing model.
    
    Parameters:
        best_result (dict): Result dict for the best model.
    
    Returns:
        str: Formatted report showing:
            - Model name and hyperparameters
            - Train/val/test metrics
            - Why this model is best
    
    TODO: Implement detailed best model report
    """
    pass


def format_run_report(summary: Dict[str, Any]) -> str:
    """Format the final session summary report.
    
    Parameters:
        summary (dict): Benchmark summary (from models.create_benchmark_summary).
    
    Returns:
        str: Complete session report including:
            - Results table
            - Best model details
            - Artifacts saved
            - Execution time
    
    TODO: Implement final report assembly
    """
    pass


def format_progress(current: int, total: int) -> str:
    """Format a simple progress indicator.
    
    Parameters:
        current (int): Current step.
        total (int): Total steps.
    
    Returns:
        str: Progress display (e.g., "[████████░░] 80% - Training...").
    
    TODO: Implement progress bar
    """
    pass
