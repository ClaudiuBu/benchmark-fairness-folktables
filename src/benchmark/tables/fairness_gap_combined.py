"""Generate a combined fairness-gap LaTeX table for income and employment.

Usage:
  python -m src.benchmark.tables.fairness_gap_combined <config.yaml>
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

from src.benchmark.metrics import METRIC_LABELS


def _format_ci(row: pd.Series | None, metric: str) -> str:
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


def _resolve_latest_output_dir(base_dir: Path) -> Path:
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
        path
        for path in base_dir.iterdir()
        if path.is_dir() and (path / "benchmark_summary_ci.csv").exists()
    ]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories with benchmark_summary_ci.csv found in {base_dir}")

    run_dirs.sort(key=lambda path: path.stat().st_mtime)
    return run_dirs[-1]


def _pick_row(
    df: pd.DataFrame,
    method: str,
    maintenance: str | None,
    sensitive_attribute: str,
) -> pd.Series | None:
    subset = df[df["method"] == method].copy()

    if maintenance is not None and "maintenance" in subset.columns:
        subset = subset[subset["maintenance"] == maintenance]

    if "sensitive_attribute" in subset.columns:
        subset = subset[subset["sensitive_attribute"] == sensitive_attribute]

    if subset.empty:
        return None
    return subset.iloc[0]


def _load_summary_ci(base_dir: str) -> pd.DataFrame:
    run_dir = _resolve_latest_output_dir(Path(base_dir))
    summary_ci_path = run_dir / "benchmark_summary_ci.csv"
    if not summary_ci_path.exists():
        raise FileNotFoundError(f"Missing file: {summary_ci_path}")
    return pd.read_csv(summary_ci_path)


def generate_fairness_gap_combined_table(config_path: str) -> Path:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    inputs = config["inputs"]
    selection = config.get("selection", {})
    output_cfg = config["output"]

    income_df = _load_summary_ci(inputs["income_base_dir"])
    employment_df = _load_summary_ci(inputs["employment_base_dir"])

    method = selection.get("method", "baseline")
    maintenance = selection.get("maintenance", "no-retrain")
    race_attr = selection.get("race_attribute", "RAC1P")
    sex_attr = selection.get("sex_attribute", "SEX")
    metrics = selection.get(
        "metrics",
        ["auc", "brier_score", "oe_gap", "accuracy", "sensitivity", "f1_score", "dp_gap", "eo_gap"],
    )
    drop_fully_missing_metrics = bool(selection.get("drop_fully_missing_metrics", True))

    income_race_row = _pick_row(income_df, method=method, maintenance=maintenance, sensitive_attribute=race_attr)
    employment_race_row = _pick_row(employment_df, method=method, maintenance=maintenance, sensitive_attribute=race_attr)
    income_sex_row = _pick_row(income_df, method=method, maintenance=maintenance, sensitive_attribute=sex_attr)
    employment_sex_row = _pick_row(employment_df, method=method, maintenance=maintenance, sensitive_attribute=sex_attr)

    table_rows: list[tuple[str, str, str, str, str]] = []
    for metric in metrics:
        row = (
            METRIC_LABELS.get(metric, metric.replace("_", " ").title()),
            _format_ci(income_race_row, metric),
            _format_ci(employment_race_row, metric),
            _format_ci(income_sex_row, metric),
            _format_ci(employment_sex_row, metric),
        )
        if drop_fully_missing_metrics and all(cell == "-" for cell in row[1:]):
            continue
        table_rows.append(row)

    race_label = selection.get("race_label", "Race-based gaps (White vs Non-White)")
    sex_label = selection.get("sex_label", "Sex-based gaps (Male vs Female)")
    income_label = selection.get("income_label", "Income")
    employment_label = selection.get("employment_label", "Employment")

    latex_lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        f" & \\multicolumn{{2}}{{c}}{{{race_label}}} & \\multicolumn{{2}}{{c}}{{{sex_label}}} \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
        f"Metric & {income_label} & {employment_label} & {income_label} & {employment_label} \\\\",
        "\\midrule",
    ]

    for row in table_rows:
        latex_lines.append(f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} \\\\")

    latex_lines.extend(["\\bottomrule", "\\end{tabular}", ""])

    output_path = Path(output_cfg["path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(latex_lines))
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.benchmark.tables.fairness_gap_combined <config.yaml>")
        raise SystemExit(1)

    path = generate_fairness_gap_combined_table(sys.argv[1])
    print(f"Saved combined fairness-gap table to {path}")