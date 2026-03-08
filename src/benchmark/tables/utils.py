"""Common utilities for table generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def format_ci(row: pd.Series | None, metric: str) -> str:
    """Format confidence interval as 'mean [ci_lower--ci_upper]'.
    
    Args:
        row: DataFrame row containing metric values, or None if missing
        metric: Metric name (e.g., 'auc', 'dp_gap')
    
    Returns:
        Formatted string like "0.855 [0.854--0.856]" or "-" if data missing
    """
    if row is None:
        return "-"

    mean_col = f"{metric}_mean"
    lower_col = f"{metric}_ci_lower"
    upper_col = f"{metric}_ci_upper"

    if mean_col not in row or pd.isna(row[mean_col]):
        return "-"

    mean = float(row[mean_col])
    lower = float(row[lower_col]) if lower_col in row and not pd.isna(row[lower_col]) else mean
    upper = float(row[upper_col]) if upper_col in row and not pd.isna(row[upper_col]) else mean
    return f"{mean:.3f} [{lower:.3f}--{upper:.3f}]"


def resolve_latest_output_dir(base_dir: Path) -> Path:
    """Resolve the latest benchmark output directory.
    
    Resolution order:
    1. Check run_history.csv for latest output_dir
    2. Check if base_dir contains benchmark_summary_ci.csv directly
    3. Scan subdirectories for benchmark_summary_ci.csv, use most recent
    
    Args:
        base_dir: Base directory containing benchmark runs
    
    Returns:
        Path to directory containing benchmark_summary_ci.csv
    
    Raises:
        FileNotFoundError: If no valid run directory found
    """
    history_path = base_dir / "run_history.csv"
    if history_path.exists():
        history_df = pd.read_csv(history_path)
        if not history_df.empty and "output_dir" in history_df.columns:
            latest_output_dir = Path(str(history_df.iloc[-1]["output_dir"]))
            if latest_output_dir.exists():
                return latest_output_dir

    if (base_dir / "benchmark_summary_ci.csv").exists():
        return base_dir

    run_dirs = [
        p
        for p in base_dir.iterdir()
        if p.is_dir() and (p / "benchmark_summary_ci.csv").exists()
    ]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories with benchmark_summary_ci.csv found in {base_dir}")

    run_dirs.sort(key=lambda p: p.stat().st_mtime)
    return run_dirs[-1]


def pick_row(
    df: pd.DataFrame,
    method: str,
    maintenance: str | None = None,
    sensitive_attribute: str | None = None,
) -> pd.Series | None:
    """Select a row from benchmark results matching specified criteria.
    
    Args:
        df: DataFrame with benchmark results
        method: Method name (e.g., 'baseline', 'reweighing')
        maintenance: Maintenance strategy (e.g., 'no-retrain'), or None to skip filter
        sensitive_attribute: Sensitive attribute (e.g., 'SEX', 'RAC1P'), or None to skip filter
    
    Returns:
        First matching row as Series, or None if no match
    """
    subset = df[df["method"] == method].copy() if "method" in df.columns else df.copy()

    if maintenance is not None and "maintenance" in subset.columns:
        subset = subset[subset["maintenance"] == maintenance]

    if sensitive_attribute is not None and "sensitive_attribute" in subset.columns:
        subset = subset[subset["sensitive_attribute"] == sensitive_attribute]

    if subset.empty:
        return None
    return subset.iloc[0]
