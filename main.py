"""
main.py — Entry point for the AutoML Comparison Notebook benchmark suite.

Usage:
    python main.py [--samples N] [--features N] [--seed N] [--csv PATH]
"""

from __future__ import annotations

import argparse

from display import (
    format_config_summary,
    format_dataset_profile,
    format_header,
    format_metrics_report,
    format_results_table,
    save_confusion_matrix_plot,
    save_learning_curves_plot,
    save_model_comparison_plot,
)
from operations import run_core_flow
from storage import DATA_DIR


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace object with parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="AutoML Comparison Notebook — tabular benchmarking suite."
    )
    parser.add_argument("--samples", type=int, default=2000, help="Number of samples (synthetic).")
    parser.add_argument("--features", type=int, default=20, help="Number of features (synthetic).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--csv", type=str, default=None, help="Path to a custom CSV dataset.")
    return parser.parse_args()


def main() -> None:
    """Run the complete AutoML benchmark workflow and print results.

    Workflow:
        1. Parse CLI arguments.
        2. Print banner and configuration summary.
        3. Run grid search via run_core_flow().
        4. Print dataset profile and benchmark results.
        5. Save comparison plot, confusion matrix, and learning curves.
        6. Print artifact paths and elapsed time.
    """
    args = _parse_args()

    print(format_header())

    config = {
        "n_samples": args.samples,
        "n_features": args.features,
        "val_size": 0.2,
        "test_size": 0.2,
        "random_state": args.seed,
        "search_type": "Grid search",
    }
    print(format_config_summary(config))

    print("\nBenchmark Progress:")
    summary = run_core_flow(
        n_samples=args.samples,
        n_features=args.features,
        random_state=args.seed,
        csv_path=args.csv,
        save_json=True,
    )

    print(format_dataset_profile(summary["dataset_profile"]))
    print(format_results_table(summary["ranked_results"], best_result=summary["best_result"]))
    print(format_metrics_report(summary["best_result"]))

    # --- Retrieve in-memory objects for visualisation ---
    search_results = summary["_search_results"]
    X_train, X_val, X_test, y_train, y_val, y_test = summary["_splits"]
    best_model_obj = search_results["best_result"]["model_object"]
    best_model_name = summary["best_result"]["config"]["model_name"]

    comparison_path = save_model_comparison_plot(summary["ranked_results"])
    confusion_path = save_confusion_matrix_plot(
        best_model_obj, X_test, y_test, best_model_name
    )
    learning_path = save_learning_curves_plot(
        best_model_obj, X_train, y_train, best_model_name, random_state=args.seed
    )

    json_path = summary.get("json_path", "data/runs/benchmark_report.json")

    print("\nArtifacts saved:")
    print(f"   - Benchmarking report:     {json_path}")
    print(f"   - Metrics comparison plot: {comparison_path}")
    print(f"   - Confusion matrix:        {confusion_path}")
    print(f"   - Learning curves:         {learning_path}")
    print(f"\nSession completed in {summary['elapsed_seconds']} seconds.")


if __name__ == "__main__":
    main()
