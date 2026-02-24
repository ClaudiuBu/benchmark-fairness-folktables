"""Generate summary tables grouped by sensitive attribute and task."""

from pathlib import Path
import pandas as pd


def _format_ci(row: pd.Series, metric: str) -> str:
    """Format metric with confidence interval."""
    mean_col = f"{metric}_mean"
    lower_col = f"{metric}_ci_lower"
    upper_col = f"{metric}_ci_upper"
    
    if mean_col not in row or pd.isna(row[mean_col]):
        return "-"
    
    mean = float(row[mean_col])
    lower = float(row[lower_col]) if lower_col in row and not pd.isna(row[lower_col]) else mean
    upper = float(row[upper_col]) if upper_col in row and not pd.isna(row[upper_col]) else mean
    
    return f"{mean:.3f} [{lower:.3f}--{upper:.3f}]"


def _get_attribute_description(attr: str) -> str:
    """Get human-readable description of sensitive attribute."""
    descriptions = {
        "SEX": "Gender (Male=1, Female=0)",
        "RAC1P": "Race (White=1, Non-White=0)",
    }
    return descriptions.get(attr, attr)


def _get_task_description(task: str) -> str:
    """Get human-readable description of task."""
    descriptions = {
        "income": "Income Prediction (>\\$50K)",
        "employment": "Employment Status Prediction",
    }
    return descriptions.get(task, task)


def _build_latex_table_string(table_df: pd.DataFrame, num_cols: int) -> str:
    """Convert DataFrame to LaTeX table string.
    
    Args:
        table_df: DataFrame with table data
        num_cols: Number of columns (used for validation)
        
    Returns:
        LaTeX tabular content as string
    """
    col_spec = "l" * num_cols  # Left-align all columns
    # Use pandas to_latex with escape=False to avoid double-escaping
    # (escaping is already handled in the data)
    latex = table_df.to_latex(index=False, escape=False, column_format=col_spec)
    return latex



def _wrap_table_with_metadata(table_latex: str, task: str, attribute: str, output_filename: str, has_maintenance: bool = False) -> str:
    """Wrap table with caption, label, and professional formatting.
    
    Args:
        table_latex: Raw LaTeX table content
        task: Task name (e.g., 'income', 'employment')
        attribute: Sensitive attribute (e.g., 'SEX', 'RAC1P')
        output_filename: Output filename for label generation
        has_maintenance: Whether table includes maintenance strategies (retrain vs no-retrain)
    """
    task_desc = _get_task_description(task)
    attr_desc = _get_attribute_description(attribute)
    
    # Create label from filename (without .tex)
    label = f"tab:{output_filename.replace('.tex', '')}"
    
    # Build caption - mention maintenance strategies if present
    if has_maintenance:
        caption = f"Fairness metrics by method and maintenance strategy. Task: {task_desc}. Sensitive attribute: {attr_desc}. " \
                  f"Methods shown with 'No-Retrain' (model not updated on new years) and 'Retrain' (model updated annually). " \
                  f"Values shown as mean [95\\% CI lower--upper] across 20 random seeds."
    else:
        caption = f"Fairness metrics by method. Task: {task_desc}. Sensitive attribute: {attr_desc}. " \
                  f"Values shown as mean [95\\% CI lower--upper] across 20 random seeds."
    
    # Build complete table environment
    table_env = f"""\\begin{{table}}[h]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\small
{table_latex}
\\end{{table}}"""
    
    return table_env


def generate_summary_tables_by_attribute(summary_ci: pd.DataFrame, output_dir: Path):
    """Generate summary tables for each (task, sensitive_attribute) combination.
    
    Args:
        summary_ci: DataFrame with results grouped by method and maintenance (if temporal)
        output_dir: Directory to save LaTeX tables
        
    Generates:
        - summary_{task}_{attribute}.tex for each task+attribute pair
        Example: summary_income_sex.tex, summary_employment_rac1p.tex
    """
    
    if "sensitive_attribute" not in summary_ci.columns:
        print("⚠ Warning: 'sensitive_attribute' column not found. Skipping attribute-specific tables.")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = ["eo_gap", "dp_gap", "accuracy", "auc"]
    metric_labels = {
        "eo_gap": "EO Gap",
        "dp_gap": "DP Gap",
        "accuracy": "Accuracy",
        "auc": "AUC"
    }
    
    # Check if task column exists
    has_task = "task" in summary_ci.columns
    
    if has_task:
        tasks = sorted(summary_ci["task"].unique())
    else:
        print("⚠ Warning: 'task' column not found. No tables generated.")
        return
    
    attributes = sorted(summary_ci["sensitive_attribute"].unique())
    
    for task in tasks:
        # Filter data by task
        task_data = summary_ci[summary_ci["task"] == task]
        
        for attr in attributes:
            attr_data = task_data[task_data["sensitive_attribute"] == attr]
            
            if attr_data.empty:
                continue
            
            rows = []
            has_maintenance = "maintenance" in attr_data.columns
            
            for metric in metrics:
                row = {"Metric": metric_labels.get(metric, metric)}
                
                # Get all methods for this task+attribute combination
                for method in sorted(attr_data["method"].unique()):
                    method_data = attr_data[attr_data["method"] == method]
                    
                    if has_maintenance and not method_data.empty:
                        # For temporal mode: show both maintenance strategies (no-retrain and retrain)
                        for maint in sorted(method_data["maintenance"].unique()):
                            maint_data = method_data[method_data["maintenance"] == maint]
                            if not maint_data.empty:
                                val = _format_ci(maint_data.iloc[0], metric)
                                # Column name: "Method (Strategy)" e.g. "Baseline (No-Retrain)"
                                maint_label = maint.replace("-", " ").title()
                                col_name = f"{method.replace('_', ' ').title()} ({maint_label})"
                                row[col_name] = val
                    else:
                        # Static mode: just show method
                        if not method_data.empty:
                            val = _format_ci(method_data.iloc[0], metric)
                        else:
                            val = "-"
                        
                        col_name = method.replace("_", " ").title()
                        row[col_name] = val
                
                rows.append(row)
            
            table_df = pd.DataFrame(rows)
            
            # Generate LaTeX table manually to avoid pandas escaping issues
            num_cols = len(table_df.columns)
            latex_table = _build_latex_table_string(table_df, num_cols)
            
            # Save with explicit task name (not "general")
            attr_name = attr.lower()
            output_filename = f"summary_{task}_{attr_name}.tex"
            table_path = output_dir / output_filename
            
            # Wrap with metadata (pass has_maintenance flag for caption)
            wrapped_latex = _wrap_table_with_metadata(latex_table, task, attr, output_filename, has_maintenance=has_maintenance)
            table_path.write_text(wrapped_latex)
            
            print(f"✓ Saved {task.upper()} - {attr} fairness table to {table_path}")
