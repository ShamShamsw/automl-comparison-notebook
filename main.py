"""
main.py - Entry point for Project 21: AutoML Comparison Notebook
=================================================================

Thin-controller module that only orchestrates calls to:
    - operations.py for workflow execution
    - display.py for presentation formatting
    - storage.py for result persistence
    - models.py for configuration and data models
"""

import time
from display import format_header, format_startup_guide, format_run_report
from models import create_project_config
from operations import load_dataset_profile, run_core_flow
from storage import ensure_data_dir, save_benchmark_results


def main() -> None:
    """Run one complete AutoML benchmarking session.
    
    Workflow:
        1. Print header and configuration
        2. Load/generate dataset and print profile
        3. Run benchmark (orchestrated by operations.run_core_flow)
        4. Print results and best model report
        5. Save results to JSON
    
    Returns:
        None
    
    TODO: Implement main orchestration
    """
    # Ensure data directories exist
    ensure_data_dir()
    
    # Create configuration
    config = create_project_config()
    
    # Load dataset profile
    profile = load_dataset_profile()
    
    # Print header and setup info
    print(format_header())
    print(format_startup_guide(config, profile))
    
    # Run benchmark
    start_time = time.time()
    run_summary = run_core_flow()
    elapsed = time.time() - start_time
    run_summary['execution_time_seconds'] = elapsed
    
    # Print final report
    print(format_run_report(run_summary))
    
    # Save results to JSON
    save_benchmark_results('benchmark_results.json', run_summary)
    print(f"\n✓ Results saved to data/runs/benchmark_results.json")


if __name__ == "__main__":
    # Guard keeps imports side-effect free for tests and reuse.
    main()
