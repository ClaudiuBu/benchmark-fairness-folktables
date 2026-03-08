"""Generate a combined initial-performance LaTeX table for income and employment.

Usage:
  python -m src.benchmark.tables.initial_performance_combined <config.yaml>
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

from src.benchmark.metrics import METRIC_LABELS


def _format_ci(row: pd.Series, metric: str) -> str:
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
        p
        for p in base_dir.iterdir()
        if p.is_dir() and (p / "benchmark_summary_ci.csv").exists()
    ]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories with benchmark_summary_ci.csv found in {base_dir}")

    run_dirs.sort(key=lambda p: p.stat().st_mtime)
    return run_dirs[-1]


def _pick_row(df: pd.DataFrame, method: str, maintenance: str | None, sensitive_attribute: str | None) -> pd.Series | None:
    subset = df[df["method"] == method].copy() if "method" in df.columns else df.copy()

    if maintenance is not None and "maintenance" in subset.columns:
        subset = subset[subset["maintenance"] == maintenance]

    if sensitive_attribute is not None and "sensitive_attribute" in subset.columns:
        subset = subset[subset["sensitive_attribute"] == sensitive_attribute]

    if subset.empty:
        return None
    return subset.iloc[0]


def _build_task_values(
    summary_ci_path: Path,
    method: str,
    maintenance: str | None,
    sex_attr: str,
    race_attr: str,
    performance_metrics: list[str],
) -> dict[str, str]:
    df = pd.read_csv(summary_ci_path)

    values: dict[str, str] = {}

    # Performance metrics (AUC, accuracy, etc.) don't vary by sensitive attribute
    # They are duplicated in CSV rows, so we can pick any row (prefer first available)
    perf_row = _pick_row(df, method=method, maintenance=maintenance, sensitive_attribute=sex_attr)
    if perf_row is None:
        perf_row = _pick_row(df, method=method, maintenance=maintenance, sensitive_attribute=None)

    for metric in performance_metrics:
        if perf_row is None:
            values[metric] = "-"
        else:
            values[metric] = _format_ci(perf_row, metric)

    # Fairness metrics (DP/EO gap) DO vary by sensitive attribute, pick specific rows
    sex_row = _pick_row(df, method=method, maintenance=maintenance, sensitive_attribute=sex_attr)
    race_row = _pick_row(df, method=method, maintenance=maintenance, sensitive_attribute=race_attr)

    values["dp_gap_sex"] = _format_ci(sex_row, "dp_gap") if sex_row is not None else "-"
    values["eo_gap_sex"] = _format_ci(sex_row, "eo_gap") if sex_row is not None else "-"
    values["dp_gap_race"] = _format_ci(race_row, "dp_gap") if race_row is not None else "-"
    values["eo_gap_race"] = _format_ci(race_row, "eo_gap") if race_row is not None else "-"

    return values


def generate_combined_initial_performance_table(config_path: str) -> Path:
    with open(config_path, "r") as file:
        cfg = yaml.safe_load(file)

    inputs = cfg["inputs"]
    selection = cfg.get("selection", {})
    output_cfg = cfg["output"]

    income_base_dir = Path(inputs["income_base_dir"])
    employment_base_dir = Path(inputs["employment_base_dir"])

    method = selection.get("method", "baseline")
    maintenance = selection.get("maintenance", "no-retrain")
    sensitive_attrs = selection.get("sensitive_attributes", {})
    sex_attr = sensitive_attrs.get("sex", "SEX")
    race_attr = sensitive_attrs.get("race", "RAC1P")
    performance_metrics = selection.get(
        "performance_metrics",
        ["auc", "brier_score", "accuracy", "sensitivity", "f1_score", "oe_gap"],
    )

    income_run_dir = _resolve_latest_output_dir(income_base_dir)
    employment_run_dir = _resolve_latest_output_dir(employment_base_dir)

    income_summary_ci = income_run_dir / "benchmark_summary_ci.csv"
    employment_summary_ci = employment_run_dir / "benchmark_summary_ci.csv"

    if not income_summary_ci.exists():
        raise FileNotFoundError(f"Missing file: {income_summary_ci}")
    if not employment_summary_ci.exists():
        raise FileNotFoundError(f"Missing file: {employment_summary_ci}")

    income_values = _build_task_values(
        income_summary_ci,
        method=method,
        maintenance=maintenance,
        sex_attr=sex_attr,
        race_attr=race_attr,
        performance_metrics=performance_metrics,
    )
    employment_values = _build_task_values(
        employment_summary_ci,
        method=method,
        maintenance=maintenance,
        sex_attr=sex_attr,
        race_attr=race_attr,
        performance_metrics=performance_metrics,
    )

    output_path = Path(output_cfg["path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    latex_lines = [
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Metric & Income ($>$\\$50K) & Employment \\\\",
        "\\midrule",
    ]

    for metric in performance_metrics:
        metric_label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
        latex_lines.append(
            f"{metric_label} & {income_values[metric]} & {employment_values[metric]} \\\\"
        )

    latex_lines.extend(
        [
            "\\midrule",
            "\\multicolumn{3}{l}{\\textit{Fairness Metrics (Sex):}} \\\\",
            f"DP Gap & {income_values['dp_gap_sex']} & {employment_values['dp_gap_sex']} \\\\",
            f"EO Gap & {income_values['eo_gap_sex']} & {employment_values['eo_gap_sex']} \\\\",
            "\\midrule",
            "\\multicolumn{3}{l}{\\textit{Fairness Metrics (Race):}} \\\\",
            f"DP Gap & {income_values['dp_gap_race']} & {employment_values['dp_gap_race']} \\\\",
            f"EO Gap & {income_values['eo_gap_race']} & {employment_values['eo_gap_race']} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "",
        ]
    )

    output_path.write_text("\n".join(latex_lines))
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.benchmark.tables.initial_performance_combined <config.yaml>")
        raise SystemExit(1)

    path = generate_combined_initial_performance_table(sys.argv[1])
    print(f"Saved combined initial performance table to {path}")