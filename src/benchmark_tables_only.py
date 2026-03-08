"""
Generate tables from existing benchmark results (CSV) without retraining models.
Usage: python -m src.benchmark_tables_only <results_dir>
"""

import sys
from pathlib import Path

import pandas as pd

from src.benchmark.tables.summary_by_attribute import generate_summary_tables_by_attribute
from src.benchmark.runner import _generate_initial_performance_table


def regenerate_tables_from_results(results_dir: str):
    """Load existing benchmark results and regenerate all tables."""
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Load summary with confidence intervals
    summary_ci_path = results_dir / "benchmark_summary_ci.csv"
    if not summary_ci_path.exists():
        print(f"ERROR: Summary file not found: {summary_ci_path}")
        print("Available files:")
        for f in results_dir.glob("*.csv"):
            print(f"  - {f.name}")
        sys.exit(1)
    
    print(f"Loading results from {summary_ci_path}")
    summary_ci = pd.read_csv(summary_ci_path)
    
    # Generate attribute-specific summary tables (SEX, RAC1P, etc)
    print("Generating summary tables by attribute...")
    generate_summary_tables_by_attribute(summary_ci, results_dir)
    
    # Generate initial performance table for paper
    print("Generating initial performance table...")
    task = "income"  # Default, can be inferred from summary_ci
    if "task" in summary_ci.columns:
        task = summary_ci["task"].iloc[0]
    
    # Filter for no-retrain if maintenance column exists
    initial_subset = summary_ci.copy()
    if "maintenance" in initial_subset.columns:
        initial_subset = initial_subset[initial_subset["maintenance"] == "no-retrain"]
    
    initial_table_path = _generate_initial_performance_table(
        initial_subset,
        task,
        results_dir,
    )
    
    print("\n✓ Tables regenerated successfully")
    print(f"  Output directory: {results_dir}")
    if initial_table_path:
        print(f"  Initial performance table: {initial_table_path}")
    
    # List generated tables
    tex_files = list(results_dir.glob("*.tex"))
    if tex_files:
        print("\n  Generated LaTeX tables:")
        for tex_file in tex_files:
            print(f"    - {tex_file.name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.benchmark_tables_only <results_dir>")
        print("\nExample:")
        print("  python -m src.benchmark_tables_only results/folktables/benchmark_temporal")
        sys.exit(1)
    
    regenerate_tables_from_results(sys.argv[1])
