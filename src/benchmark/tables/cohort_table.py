"""Generate cohort characteristics table for training vs temporal period."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from folktables import ACSDataSource, ACSIncome, ACSEmployment


RACE_LABELS = {
    1: "White",
    2: "Black",
    3: "American Indian/Alaska Native",
    4: "Asian",
    5: "Native Hawaiian/Pacific Islander",
    6: "Other",
    7: "Two or more races",
    8: "Other",
    9: "Two or more races",
}


def _education_bucket(schl: pd.Series) -> pd.Series:
    # ACS SCHL codes: 1-15 (<HS), 16-17 (HS/GED), 18-20 (some college/assoc), 21-24 (bachelor+)
    bins = pd.Series(index=schl.index, dtype=object)
    bins.loc[schl.between(1, 15, inclusive="both")] = "Less than HS"
    bins.loc[schl.between(16, 17, inclusive="both")] = "HS or GED"
    bins.loc[schl.between(18, 20, inclusive="both")] = "Some college"
    bins.loc[schl.between(21, 24, inclusive="both")] = "Bachelor+"
    bins = bins.fillna("Unknown")
    return bins


def _load_period_dataset(task: str, states, years, max_samples: int, random_state: int) -> pd.DataFrame:
    task_cls = ACSIncome if task == "income" else ACSEmployment
    all_rows = []

    for year in years:
        data_source = ACSDataSource(
            survey_year=str(year), horizon="1-Year", survey="person"
        )
        for state in states:
            acs_data = data_source.get_data(states=[state], download=True)
            X_df, y, _ = task_cls.df_to_pandas(acs_data)
            X_df = X_df.copy()
            X_df["__y__"] = y.values.astype(int)
            X_df["__year__"] = int(year)
            all_rows.append(X_df)

    df = pd.concat(all_rows, axis=0).reset_index(drop=True)
    if max_samples is not None and max_samples < len(df):
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(df), size=max_samples, replace=False)
        df = df.iloc[idx].reset_index(drop=True)

    return df


def _format_percent(value: float) -> str:
    return f"{value:.1f}\\%"


def _format_median_iqr(series: pd.Series) -> str:
    median = float(series.median())
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    return f"{median:.1f} [{q1:.1f}, {q3:.1f}]"


def _summarize(df: pd.DataFrame) -> dict:
    summary = {}
    summary["N"] = f"{len(df):,}".replace(",", " ")
    summary["Positive %"] = _format_percent(100.0 * float(df["__y__"].mean()))

    if "SEX" in df.columns:
        summary["Female %"] = _format_percent(100.0 * float((df["SEX"] == 2).mean()))
        summary["Male %"] = _format_percent(100.0 * float((df["SEX"] == 1).mean()))

    if "RAC1P" in df.columns:
        race_counts = df["RAC1P"].value_counts(dropna=False)
        for code, label in RACE_LABELS.items():
            if code in race_counts.index:
                summary[f"Race: {label} %"] = _format_percent(100.0 * float(race_counts[code] / len(df)))

    if "AGEP" in df.columns:
        summary["Age median [IQR]"] = _format_median_iqr(df["AGEP"].dropna())

    if "SCHL" in df.columns:
        edu = _education_bucket(df["SCHL"].dropna())
        edu_counts = edu.value_counts(dropna=False)
        for label in ["Less than HS", "HS or GED", "Some college", "Bachelor+"]:
            if label in edu_counts.index:
                summary[f"Education: {label} %"] = _format_percent(100.0 * float(edu_counts[label] / len(edu)))

    return summary


def generate_cohort_table(config_path: str) -> Path:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    task = data_cfg["task"]
    states = data_cfg["states"]
    max_samples = data_cfg.get("max_samples")
    sample_seed = int(config.get("sample_seed", 42))

    if data_cfg.get("mode", "static") == "temporal":
        train_years = data_cfg["train_years"]
        temporal_years = data_cfg["val_years"] + data_cfg["test_years"]
    else:
        train_years = data_cfg["years"]
        temporal_years = data_cfg["years"]

    train_df = _load_period_dataset(task, states, train_years, max_samples, sample_seed)
    temporal_df = _load_period_dataset(task, states, temporal_years, max_samples, sample_seed + 1)

    train_summary = _summarize(train_df)
    temporal_summary = _summarize(temporal_df)

    rows = []
    for key in train_summary.keys():
        rows.append({
            "Characteristic": key,
            f"Training ({min(train_years)}-{max(train_years)})": train_summary.get(key, "-"),
            f"Temporal ({min(temporal_years)}-{max(temporal_years)})": temporal_summary.get(key, "-"),
        })

    table_df = pd.DataFrame(rows)

    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cohort_table.tex"

    latex = table_df.to_latex(index=False, escape=False)
    output_path.write_text(latex)

    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.benchmark.tables.cohort_table <config.yaml>")
        raise SystemExit(1)

    path = generate_cohort_table(sys.argv[1])
    print(f"Saved cohort table to {path}")
