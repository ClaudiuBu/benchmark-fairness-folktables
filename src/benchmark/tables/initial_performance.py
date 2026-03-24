"""Generate initial model performance tables for both tasks."""

import sys
from pathlib import Path

import pandas as pd

from src.benchmark.metrics import METRIC_NAMES, METRIC_LABELS


def _format_ci(row: pd.Series, metric: str) -> str:
    """Format metric_mean [metric_ci_lower - metric_ci_upper]."""
    mean_col = f"{metric}_mean"
    lower_col = f"{metric}_ci_lower"
    upper_col = f"{metric}_ci_upper"
    
    if mean_col not in row or pd.isna(row[mean_col]):
        return "-"
    
    mean = float(row[mean_col])
    lower = float(row[lower_col]) if lower_col in row and not pd.isna(row[lower_col]) else mean
    upper = float(row[upper_col]) if upper_col in row and not pd.isna(row[upper_col]) else mean
    
    return f"{mean:.3f} [{lower:.3f}--{upper:.3f}]"


def generate_initial_performance_tables(results_path: str, output_dir: Path) -> tuple:
    """Generate initial performance tables for income and employment.
    
    Returns tuple of (income_table_path, employment_table_path).
    """
    
    df = pd.read_csv(results_path)
    
    # Filter for temporal mode (should have 'maintenance' column if present)
    if "maintenance" in df.columns:
        # Use only no-retrain (initial models)
        df = df[df["maintenance"] == "no-retrain"].copy()
    
    rows_income = []
    rows_employment = []
    
    metrics = [metric for metric in METRIC_NAMES if f"{metric}_mean" in df.columns]
    if not metrics:
        return (output_dir / "initial_performance_income.tex", output_dir / "initial_performance_employment.tex")
    
    has_task_column = "task" in df.columns
    
    for metric in metrics:
        label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
        row_income = {"Metric": label}
        row_employment = {"Metric": label}
        
        for method in ["baseline", "reweighing", "equalized_odds", "fairness_constraint"]:
            if has_task_column:
                income_data = df[(df["method"] == method) & (df["task"] == "income")]
                employment_data = df[(df["method"] == method) & (df["task"] == "employment")]
            else:
                income_data = df[df["method"] == method]
                employment_data = df[df["method"] == method]
            
            col_name = method.replace("_", " ").title()
            
            row_income[col_name] = _format_ci(income_data.iloc[0], metric) if not income_data.empty else "-"
            row_employment[col_name] = _format_ci(employment_data.iloc[0], metric) if not employment_data.empty else "-"
        
        rows_income.append(row_income)
        rows_employment.append(row_employment)
    
    # Create DataFrames
    table_income = pd.DataFrame(rows_income)
    table_employment = pd.DataFrame(rows_employment)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as LaTeX
    income_path = output_dir / "initial_performance_income.tex"
    employment_path = output_dir / "initial_performance_employment.tex"
    
    latex_income = table_income.to_latex(index=False, escape=False)
    latex_employment = table_employment.to_latex(index=False, escape=False)
    
    income_path.write_text(latex_income)
    employment_path.write_text(latex_employment)
    
    return (income_path, employment_path)


def generate_from_command_line(results_path: str, output_dir_str: str):
    """Command-line interface."""
    output_dir = Path(output_dir_str)
    income_path, employment_path = generate_initial_performance_tables(results_path, output_dir)
    
    print(f"✓ Saved income table to {income_path}")
    print(f"✓ Saved employment table to {employment_path}")
    
    # Also print as text for quick inspection
    print("\n--- Income Table Preview ---")
    print(pd.read_csv(results_path).head(10))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m src.benchmark.tables.initial_performance <results.csv> <output_dir>")
        raise SystemExit(1)
    
    generate_from_command_line(sys.argv[1], sys.argv[2])
