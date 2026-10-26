"""Generate final temporal delta summary tables from benchmark outputs.

The table uses explicit 2014 baseline rows and aggregates simulated
within-year folds into calendar-year estimates before computing the final
temporal delta. This keeps yearly evidence separate from fold-level plots.

Usage:
  python -m src.benchmark.tables.final_delta_summary <config.yaml>
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.benchmark.metrics import METRIC_LABELS


def _resolve_latest_output_dir(base_dir: Path) -> Path:
    history_path = base_dir / "run_history.csv"
    if history_path.exists():
        history_df = pd.read_csv(history_path)
        if not history_df.empty and "output_dir" in history_df.columns:
            latest_output_dir = Path(str(history_df.iloc[-1]["output_dir"]))
            if latest_output_dir.exists():
                return latest_output_dir

    if (base_dir / "benchmark_results_by_year.csv").exists():
        return base_dir

    run_dirs = [
        path
        for path in base_dir.iterdir()
        if path.is_dir() and (path / "benchmark_results_by_year.csv").exists()
    ]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories with benchmark_results_by_year.csv found in {base_dir}")

    run_dirs.sort(key=lambda path: path.stat().st_mtime)
    return run_dirs[-1]


def _mean_ci(values: pd.Series) -> tuple[float, float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return (np.nan, np.nan, np.nan)
    mean = float(clean.mean())
    if len(clean) < 2:
        return (mean, mean, mean)
    stderr = float(clean.std(ddof=1) / math.sqrt(len(clean)))
    # Normal approximation is sufficient here because this is a compact
    # diagnostic table; detailed CIs are available in the benchmark outputs.
    ci = 1.96 * stderr
    return (mean, mean - ci, mean + ci)


def _format_value(value: float) -> str:
    if pd.isna(value):
        return "--"
    return f"{value:.3f}"


def _format_delta(value: float) -> str:
    if pd.isna(value):
        return "--"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.3f}"


def _format_relative(value: float) -> str:
    if pd.isna(value):
        return "--"
    sign = "+" if value > 0 else ""
    return f"{sign}{100.0 * value:.1f}\\%"


def _parse_formatted_float(value: str) -> float:
    if value in {"", "--"} or pd.isna(value):
        return np.nan
    return float(str(value).replace("\\%", "").replace("+", ""))


def _linear_slope(df: pd.DataFrame, metric: str) -> float:
    """Estimate yearly linear slope after aggregating seeds within calendar year."""
    if metric not in df.columns:
        return np.nan
    yearly = (
        df.groupby("calendar_year", sort=True)[metric]
        .mean()
        .reset_index()
        .dropna()
    )
    if len(yearly) < 2:
        return np.nan
    return float(np.polyfit(yearly["calendar_year"], yearly[metric], deg=1)[0])


def _load_yearly_summary(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "benchmark_results_by_year.csv")
    if "sensitive_attribute_value" in df.columns:
        df = df[df["sensitive_attribute_value"].astype(str) == "ALL"].copy()

    df["calendar_year"] = np.floor(df["year"].astype(float)).astype(int)
    group_cols = [
        "phase",
        "calendar_year",
        "seed",
        "method",
        "task",
        "maintenance",
        "sensitive_attribute",
    ]
    numeric_cols = [
        col
        for col in df.select_dtypes(include="number").columns
        if col not in {"year", "calendar_year", "seed"}
    ]
    return df.groupby(group_cols, dropna=False, sort=True)[numeric_cols].mean().reset_index()


def _rows_for_task(
    run_dir: Path,
    task_label: str,
    methods: list[str],
    maintenances: list[str],
    attributes: list[str],
    gap_metrics: list[str],
    utility_metrics: list[str],
) -> list[dict[str, str]]:
    yearly = _load_yearly_summary(run_dir)
    temporal = yearly[yearly["phase"] == "temporal"].copy()
    if temporal.empty:
        return []
    final_year = int(temporal["calendar_year"].max())

    rows = []
    for method in methods:
        for maintenance in maintenances:
            for attribute in attributes:
                baseline = yearly[
                    (yearly["phase"] == "baseline")
                    & (yearly["method"] == method)
                    & (yearly["maintenance"] == "no-retrain")
                    & (yearly["sensitive_attribute"] == attribute)
                ]
                final = yearly[
                    (yearly["phase"] == "temporal")
                    & (yearly["calendar_year"] == final_year)
                    & (yearly["method"] == method)
                    & (yearly["maintenance"] == maintenance)
                    & (yearly["sensitive_attribute"] == attribute)
                ]
                if baseline.empty or final.empty:
                    continue

                for gap_metric in gap_metrics:
                    initial_mean, _, _ = _mean_ci(baseline[gap_metric])
                    final_mean, _, _ = _mean_ci(final[gap_metric])
                    absolute_delta = final_mean - initial_mean
                    slope = _linear_slope(
                        temporal[
                            (temporal["method"] == method)
                            & (temporal["maintenance"] == maintenance)
                            & (temporal["sensitive_attribute"] == attribute)
                        ],
                        gap_metric,
                    )
                    relative_delta = (
                        np.nan
                        if pd.isna(initial_mean) or abs(initial_mean) < 1e-6
                        else absolute_delta / initial_mean
                    )

                    utility_values = {}
                    for metric in utility_metrics:
                        utility_values[metric] = _mean_ci(final[metric])[0] if metric in final.columns else np.nan

                    rows.append(
                        {
                            "task": task_label,
                            "attribute": attribute,
                            "method": method,
                            "maintenance": maintenance,
                            "metric": gap_metric,
                            "initial": _format_value(initial_mean),
                            "final": _format_value(final_mean),
                            "delta": _format_delta(absolute_delta),
                            "relative_delta": _format_relative(relative_delta),
                            "slope": _format_delta(slope),
                            **{metric: _format_value(value) for metric, value in utility_values.items()},
                        }
                    )
    return rows


def _method_label(method: str) -> str:
    return {
        "baseline": "Baseline",
        "reweighing": "Reweighing",
        "demographic_parity": "Dem. Parity",
        "equalized_odds": "Eq. Odds",
        "fairness_constraint": "Fair. Constraint",
    }.get(method, method.replace("_", " ").title())


def _attribute_label(attribute: str) -> str:
    return {"SEX": "Sex", "RAC1P": "Race"}.get(attribute, attribute)


def _maintenance_label(maintenance: str) -> str:
    return {
        "no-retrain": "No-retrain",
        "cumulative-retrain": "Cumulative",
        "last-year-retrain": "Last-year",
        "threshold-recalibration": "Threshold recal.",
        "cumulative-retrain-fixed-n": "Cumulative fixed-N",
        "retrain": "Retrain",
    }.get(maintenance, maintenance)


def _pipeline_label(method: str, maintenance: str) -> str:
    return f"{_method_label(method)} / {_maintenance_label(maintenance)}"


def _summary_label(kind: str) -> str:
    return {
        "baseline_no_retrain": "Frozen baseline",
        "best_final_gap": "Lowest final gap",
        "largest_reduction": "Largest reduction",
    }.get(kind, kind.replace("_", " ").title())


def _select_compact_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Select a paper-sized diagnostic table from the exhaustive delta rows."""
    if not rows:
        return []

    df = pd.DataFrame(rows)
    df["_final_num"] = df["final"].map(_parse_formatted_float)
    df["_delta_num"] = df["delta"].map(_parse_formatted_float)

    selected = []
    group_cols = ["task", "attribute", "metric"]
    for _, group in df.groupby(group_cols, sort=True):
        baseline = group[
            (group["method"] == "baseline") & (group["maintenance"] == "no-retrain")
        ]
        if not baseline.empty:
            row = baseline.iloc[0].copy()
            row["summary"] = "baseline_no_retrain"
            selected.append(row)

        best_final = group.sort_values(["_final_num", "_delta_num"], na_position="last").head(1)
        if not best_final.empty:
            row = best_final.iloc[0].copy()
            row["summary"] = "best_final_gap"
            selected.append(row)

        largest_reduction = group.sort_values(["_delta_num", "_final_num"], na_position="last").head(1)
        if not largest_reduction.empty:
            row = largest_reduction.iloc[0].copy()
            row["summary"] = "largest_reduction"
            selected.append(row)

    compact = pd.DataFrame(selected)
    compact = compact.drop_duplicates(
        subset=["task", "attribute", "metric", "method", "maintenance", "summary"],
        keep="first",
    )
    return compact.drop(columns=[col for col in compact.columns if col.startswith("_")]).to_dict("records")


def _write_full_latex(rows: list[dict[str, str]], output_path: Path) -> None:
    latex_lines = [
        "\\begingroup",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\begin{longtable}{llcllrrrrrrrr}",
        "\\caption{Exhaustive final temporal delta summary. Initial values refer to the 2014 hold-out baseline, while final values refer to the final temporal evaluation year after aggregating simulated within-year folds into yearly estimates. PPR denotes positive prediction rate.}",
        "\\label{tab:final_delta_summary_full}\\\\",
        "\\toprule",
        "Task & Attr. & Method & Maint. & Gap & Initial & Final & $\\Delta$ & Rel. $\\Delta$ & Slope & Final Acc. & Final F1 & Final PPR \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        "Task & Attr. & Method & Maint. & Gap & Initial & Final & $\\Delta$ & Rel. $\\Delta$ & Slope & Final Acc. & Final F1 & Final PPR \\\\",
        "\\midrule",
        "\\endhead",
    ]

    for row in rows:
        metric_label = METRIC_LABELS.get(row["metric"], row["metric"].replace("_", " ").title())
        latex_lines.append(
            f"{row['task']} & {_attribute_label(row['attribute'])} & {_method_label(row['method'])} & "
            f"{_maintenance_label(row['maintenance'])} & {metric_label} & {row['initial']} & "
            f"{row['final']} & {row['delta']} & {row['relative_delta']} & {row['slope']} & "
            f"{row.get('accuracy', '--')} & {row.get('f1_score', '--')} & {row.get('positive_rate', '--')} \\\\"
        )

    latex_lines.extend(["\\bottomrule", "\\end{longtable}", "\\endgroup", ""])
    output_path.write_text("\n".join(latex_lines))


def _write_compact_latex(rows: list[dict[str, str]], output_path: Path) -> None:
    compact_rows = _select_compact_rows(rows)
    latex_lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\caption{Compact final temporal delta summary. Initial values refer to the 2014 hold-out baseline; final values refer to the final evaluation year after aggregating simulated within-year folds into calendar-year estimates. The table shows the frozen baseline, the lowest final gap, and the largest reduction for each task--attribute--gap combination. The exhaustive task $\\times$ attribute $\\times$ method $\\times$ maintenance table is exported with the generated results.}",
        "\\label{tab:final_delta_summary}",
        "\\begin{tabular}{lll l p{0.25\\textwidth} rrr l}",
        "\\toprule",
        "Task & Attr. & Gap & Row & Pipeline / maintenance & Initial & Final & $\\Delta$ & Acc/F1/PPR \\\\",
        "\\midrule",
    ]

    for row in compact_rows:
        metric_label = METRIC_LABELS.get(row["metric"], row["metric"].replace("_", " ").title())
        utility = f"{row.get('accuracy', '--')}/{row.get('f1_score', '--')}/{row.get('positive_rate', '--')}"
        latex_lines.append(
            f"{row['task']} & {_attribute_label(row['attribute'])} & {metric_label} & "
            f"{_summary_label(row['summary'])} & {_pipeline_label(row['method'], row['maintenance'])} & "
            f"{row['initial']} & {row['final']} & {row['delta']} & {utility} \\\\"
        )

    latex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    output_path.write_text("\n".join(latex_lines))

    compact_csv_path = output_path.with_name(output_path.stem + "_compact.csv")
    pd.DataFrame(compact_rows).to_csv(compact_csv_path, index=False)


def generate_final_delta_summary(config_path: str) -> Path:
    with open(config_path, "r") as file:
        cfg = yaml.safe_load(file)

    inputs = cfg["inputs"]
    selection = cfg.get("selection", {})
    output_cfg = cfg["output"]

    methods = selection.get(
        "methods",
        ["baseline", "reweighing", "demographic_parity", "equalized_odds", "fairness_constraint"],
    )
    maintenances = selection.get("maintenance", ["no-retrain", "retrain"])
    attributes = selection.get("attributes", ["RAC1P", "SEX"])
    gap_metrics = selection.get("gap_metrics", ["dp_gap", "eo_gap"])
    utility_metrics = selection.get("utility_metrics", ["accuracy", "f1_score", "positive_rate"])

    rows = []
    for task_key, task_cfg in inputs.items():
        attribute_dirs = task_cfg.get("attribute_dirs", {})
        for attribute in attributes:
            base_dir = attribute_dirs.get(attribute, task_cfg["base_dir"])
            run_dir = _resolve_latest_output_dir(Path(base_dir))
            rows.extend(
                _rows_for_task(
                    run_dir=run_dir,
                    task_label=task_cfg.get("label", task_key.title()),
                    methods=methods,
                    maintenances=maintenances,
                    attributes=[attribute],
                    gap_metrics=gap_metrics,
                    utility_metrics=utility_metrics,
                )
            )

    output_path = Path(output_cfg["path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _write_compact_latex(rows, output_path)

    full_latex_path = output_path.with_name(output_path.stem + "_full.tex")
    _write_full_latex(rows, full_latex_path)

    csv_path = output_path.with_suffix(".csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.benchmark.tables.final_delta_summary <config.yaml>")
        raise SystemExit(1)

    path = generate_final_delta_summary(sys.argv[1])
    print(f"Saved final delta summary table to {path}")
