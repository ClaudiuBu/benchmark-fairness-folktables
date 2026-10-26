"""Generate a combined initial-performance LaTeX table for income and employment.

Usage:
  python -m src.benchmark.tables.initial_performance_combined <config.yaml>
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml
from scipy import stats

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


def _summarize_initial_rows(run_dir: Path) -> pd.DataFrame:
    """Summarize explicit 2014 baseline rows from benchmark_results_by_year.csv."""
    results_by_year_path = run_dir / "benchmark_results_by_year.csv"
    if not results_by_year_path.exists():
        raise FileNotFoundError(f"Missing file: {results_by_year_path}")

    df = pd.read_csv(results_by_year_path)
    if "phase" in df.columns:
        df = df[df["phase"] == "baseline"].copy()
    else:
        df = df[df["year"] == df["year"].min()].copy()

    if "maintenance" in df.columns:
        df = df[df["maintenance"] == "no-retrain"].copy()
    if "sensitive_attribute_value" in df.columns:
        df = df[df["sensitive_attribute_value"].astype(str) == "ALL"].copy()

    group_cols = [
        col for col in ["method", "maintenance", "task", "sensitive_attribute"] if col in df.columns
    ]
    numeric_cols = [
        col for col in df.select_dtypes(include="number").columns if col not in {"seed", "year"}
    ]

    rows = []
    for group_vals, group_df in df.groupby(group_cols, sort=True):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        row = {col: group_vals[idx] for idx, col in enumerate(group_cols)}
        for metric in numeric_cols:
            values = pd.to_numeric(group_df[metric], errors="coerce").dropna().values
            if len(values) == 0:
                continue
            mean = float(values.mean())
            if len(values) < 2:
                ci_range = 0.0
            else:
                stderr = stats.sem(values, nan_policy="omit")
                ci_range = 0.0 if pd.isna(stderr) else float(stderr * stats.t.ppf(0.975, len(values) - 1))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_lower"] = mean - ci_range
            row[f"{metric}_ci_upper"] = mean + ci_range
        rows.append(row)

    return pd.DataFrame(rows)


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
    summary_ci: pd.DataFrame,
    method: str,
    maintenance: str | None,
    sex_attr: str,
    race_attr: str,
    performance_metrics: list[str],
    race_summary_ci: pd.DataFrame | None = None,
) -> dict[str, str]:
    df = summary_ci

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
    # Race gaps come from the race-configured run when one is given (one run per attribute)
    race_df = race_summary_ci if race_summary_ci is not None else df
    race_row = _pick_row(race_df, method=method, maintenance=maintenance, sensitive_attribute=race_attr)

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

    income_summary_ci = _summarize_initial_rows(income_run_dir)
    employment_summary_ci = _summarize_initial_rows(employment_run_dir)

    def _race_summary(key: str) -> pd.DataFrame | None:
        race_dir = inputs.get(key)
        return _summarize_initial_rows(_resolve_latest_output_dir(Path(race_dir))) if race_dir else None

    income_values = _build_task_values(
        income_summary_ci,
        method=method,
        maintenance=maintenance,
        sex_attr=sex_attr,
        race_attr=race_attr,
        performance_metrics=performance_metrics,
        race_summary_ci=_race_summary("income_race_dir"),
    )
    employment_values = _build_task_values(
        employment_summary_ci,
        method=method,
        maintenance=maintenance,
        sex_attr=sex_attr,
        race_attr=race_attr,
        performance_metrics=performance_metrics,
        race_summary_ci=_race_summary("employment_race_dir"),
    )

    output_path = Path(output_cfg["path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    latex_lines = [
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcc}",
        "\\toprule",
        "Metric & Income ($>$\\$50K) & Employment \\\\",
        "\\midrule",
    ]

    for metric in performance_metrics:
        metric_label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
        if metric == "oe_gap":
            metric_label += " (Sex)"
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
            "\\end{tabular*}",
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
