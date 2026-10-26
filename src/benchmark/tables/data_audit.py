"""Generate data-audit tables for Folktables cohorts.

The audit is intentionally independent from model training. It reports the
actual task cohorts by year so suspicious demographic counts, such as RAC1P
race categories, can be checked before interpreting fairness results.

Usage:
  python -m src.benchmark.tables.data_audit configs/data_audit.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml
from folktables import ACSDataSource, ACSIncome, ACSEmployment

from src.benchmark.tables.cohort_table import RACE_LABELS


SEX_LABELS = {1: "Male", 2: "Female"}


def _task_class(task: str):
    if task == "income":
        return ACSIncome
    if task == "employment":
        return ACSEmployment
    raise ValueError(f"Unsupported task: {task!r}")


def _load_task_frame(task: str, state: str, year: int) -> pd.DataFrame:
    data_source = ACSDataSource(survey_year=str(year), horizon="1-Year", survey="person")
    acs_data = data_source.get_data(states=[state], download=True)
    X_df, y, _ = _task_class(task).df_to_pandas(acs_data)
    df = X_df.copy()
    df["__y__"] = y.values.astype(int)
    df["task"] = task
    df["state"] = state
    df["year"] = int(year)
    return df


def _distribution_rows(df: pd.DataFrame, column: str, labels: dict[int, str]) -> list[dict]:
    rows = []
    if column not in df.columns:
        return rows

    counts = df[column].map(labels).fillna("Unknown").value_counts(dropna=False)
    for label, count in counts.sort_index().items():
        rows.append(
            {
                "task": df["task"].iloc[0],
                "state": df["state"].iloc[0],
                "year": int(df["year"].iloc[0]),
                "variable": column,
                "group": label,
                "n": int(count),
                "pct": float(100.0 * count / len(df)),
            }
        )
    return rows


def _summary_row(df: pd.DataFrame) -> dict:
    row = {
        "task": df["task"].iloc[0],
        "state": df["state"].iloc[0],
        "year": int(df["year"].iloc[0]),
        "n": int(len(df)),
        "outcome_positive_n": int(df["__y__"].sum()),
        "outcome_positive_pct": float(100.0 * df["__y__"].mean()),
        "has_pwgtp": bool("PWGTP" in df.columns),
    }
    if "SEX" in df.columns:
        row["female_n"] = int((df["SEX"] == 2).sum())
        row["female_pct"] = float(100.0 * (df["SEX"] == 2).mean())
        row["male_n"] = int((df["SEX"] == 1).sum())
        row["male_pct"] = float(100.0 * (df["SEX"] == 1).mean())
    if "RAC1P" in df.columns:
        race = df["RAC1P"].map(RACE_LABELS).fillna("Unknown")
        for label in sorted(set(RACE_LABELS.values()) | {"Unknown"}):
            key = label.lower().replace("/", "_").replace(" ", "_").replace("-", "_")
            row[f"race_{key}_n"] = int((race == label).sum())
            row[f"race_{key}_pct"] = float(100.0 * (race == label).mean())
    return row


def generate_data_audit(config_path: str) -> tuple[Path, Path]:
    with open(config_path, "r") as file:
        cfg = yaml.safe_load(file)

    output_dir = Path(cfg.get("output_dir", "results/folktables/tables"))
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    distribution_rows = []
    for task_cfg in cfg["tasks"]:
        task = task_cfg["task"]
        states = task_cfg.get("states", ["CA"])
        years = task_cfg["years"]
        for state in states:
            for year in years:
                df = _load_task_frame(task, state, int(year))
                summary_rows.append(_summary_row(df))
                distribution_rows.extend(_distribution_rows(df, "SEX", SEX_LABELS))
                distribution_rows.extend(_distribution_rows(df, "RAC1P", RACE_LABELS))

    summary_path = output_dir / "data_audit_summary.csv"
    distributions_path = output_dir / "data_audit_distributions.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(distribution_rows).to_csv(distributions_path, index=False)
    print(f"Saved data audit summary to {summary_path}")
    print(f"Saved data audit distributions to {distributions_path}")
    return summary_path, distributions_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m src.benchmark.tables.data_audit <config.yaml>")
    generate_data_audit(sys.argv[1])
