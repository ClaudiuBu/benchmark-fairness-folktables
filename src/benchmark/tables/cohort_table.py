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
    4: "American Indian/Alaska Native",
    5: "American Indian/Alaska Native",
    6: "Asian",
    7: "Native Hawaiian/Pacific Islander",
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


def _format_period_label(years: list[int]) -> str:
    year_min = min(years)
    year_max = max(years)
    if year_min == year_max:
        return f"{year_min}"
    return f"{year_min}--{year_max}"


def _summarize(df: pd.DataFrame, task: str) -> dict:
    """Return dict with both counts and percentages for each characteristic."""
    summary = {}
    n_total = len(df)
    
    # Total N
    summary["Persons (observations)"] = (n_total, 100.0)

    if "SEX" in df.columns:
        summary["_sex_heading"] = ("sex_heading", None)
        n_female = int((df["SEX"] == 2).sum())
        pct_female = 100.0 * float((df["SEX"] == 2).mean())
        summary["Female"] = (n_female, pct_female)
        
        n_male = int((df["SEX"] == 1).sum())
        pct_male = 100.0 * float((df["SEX"] == 1).mean())
        summary["Male"] = (n_male, pct_male)

    if "RAC1P" in df.columns:
        summary["_race_heading"] = ("race_heading", None)
        race_labels = df["RAC1P"].map(RACE_LABELS).fillna("Unknown")
        race_counts = race_labels.value_counts(dropna=False)
        race_display_order = [
            "White",
            "Black",
            "American Indian/Alaska Native",
            "Asian",
            "Native Hawaiian/Pacific Islander",
            "Other",
            "Two or more races",
            "Unknown",
        ]
        for label in race_display_order:
            if label in race_counts.index:
                n_race = int(race_counts[label])
                pct_race = 100.0 * float(race_counts[label] / len(df))
                summary[label] = (n_race, pct_race)

    if "AGEP" in df.columns:
        # Age stays as median [IQR], no counts for this
        summary["Age median [IQR]"] = ("age_special", _format_median_iqr(df["AGEP"].dropna()))

    if "SCHL" in df.columns:
        summary["_education_heading"] = ("education_heading", None)
        edu = _education_bucket(df["SCHL"].dropna())
        edu_counts = edu.value_counts(dropna=False)
        for label in ["Less than HS", "HS or GED", "Some college", "Bachelor+", "Unknown"]:
            if label in edu_counts.index:
                n_edu = int(edu_counts[label])
                pct_edu = 100.0 * float(edu_counts[label] / len(edu))
                summary[label] = (n_edu, pct_edu)

    # Outcome (at the end)
    summary["_outcome_heading"] = ("outcome_heading", None)
    n_positive = int((df["__y__"] == 1).sum())
    pct_positive = 100.0 * float(df["__y__"].mean())
    
    if task == "income":
        summary["Income $>$ \\$50K"] = (n_positive, pct_positive)
    elif task == "employment":
        summary["Employed"] = (n_positive, pct_positive)
    else:
        summary["Positive"] = (n_positive, pct_positive)

    return summary


def generate_cohort_table(config_path: str) -> Path:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    task = data_cfg["task"]
    states = data_cfg["states"]
    # Cohort tables should report real population counts by default.
    # Optional override: set data.cohort_max_samples in config.
    max_samples = data_cfg.get("cohort_max_samples")
    sample_seed = int(config.get("sample_seed", 42))

    if data_cfg.get("mode", "static") == "temporal":
        train_years = sorted(set(data_cfg["train_years"] + data_cfg.get("val_years", [])))
        development_years = set(train_years)
        temporal_years = sorted(
            year for year in data_cfg["test_years"] if year not in development_years
        )
        if not temporal_years:
            temporal_years = sorted(set(data_cfg["test_years"]))
    else:
        train_years = data_cfg["years"]
        temporal_years = data_cfg["years"]

    train_df = _load_period_dataset(task, states, train_years, max_samples, sample_seed)
    temporal_df = _load_period_dataset(task, states, temporal_years, max_samples, sample_seed + 1)

    train_summary = _summarize(train_df, task)
    temporal_summary = _summarize(temporal_df, task)

    train_period_label = _format_period_label(train_years)
    temporal_period_label = _format_period_label(temporal_years)

    rows = []
    for key in train_summary.keys():
        train_val = train_summary.get(key)
        temporal_val = temporal_summary.get(key)
        
        # Special handling for category headings
        if key == "_sex_heading":
            rows.append(
                {
                    "label": "\\addlinespace\n\\textit{Sex:}",
                    "train_n": "",
                    "train_pct": "",
                    "temporal_n": "",
                    "temporal_pct": "",
                }
            )
        elif key == "_race_heading":
            rows.append(
                {
                    "label": "\\addlinespace\n\\textit{Race:}",
                    "train_n": "",
                    "train_pct": "",
                    "temporal_n": "",
                    "temporal_pct": "",
                }
            )
        elif key == "_education_heading":
            rows.append(
                {
                    "label": "\\addlinespace\n\\textit{Education:}",
                    "train_n": "",
                    "train_pct": "",
                    "temporal_n": "",
                    "temporal_pct": "",
                }
            )
        elif key == "_outcome_heading":
            rows.append(
                {
                    "label": "\\addlinespace\n\\textit{Outcome:}",
                    "train_n": "",
                    "train_pct": "",
                    "temporal_n": "",
                    "temporal_pct": "",
                }
            )
        # Special handling for Age (median/IQR)
        elif key == "Age median [IQR]":
            rows.append(
                {
                    "label": key,
                    "train_n": "-",
                    "train_pct": train_val[1] if isinstance(train_val, tuple) else "-",
                    "temporal_n": "-",
                    "temporal_pct": temporal_val[1] if isinstance(temporal_val, tuple) else "-",
                }
            )
        else:
            # Normal case with counts and percentages
            train_n = f"{train_val[0]:,}".replace(",", " ") if isinstance(train_val, tuple) else "-"
            train_pct = _format_percent(train_val[1]) if isinstance(train_val, tuple) else "-"
            temporal_n = f"{temporal_val[0]:,}".replace(",", " ") if isinstance(temporal_val, tuple) else "-"
            temporal_pct = _format_percent(temporal_val[1]) if isinstance(temporal_val, tuple) else "-"

            rows.append(
                {
                    "label": key,
                    "train_n": train_n,
                    "train_pct": train_pct,
                    "temporal_n": temporal_n,
                    "temporal_pct": temporal_pct,
                }
            )

    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cohort_table.tex"

    latex_lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        f" & \\multicolumn{{2}}{{c}}{{Training period ({train_period_label})}} & \\multicolumn{{2}}{{c}}{{Temporal validation period ({temporal_period_label})}} \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
        " & N & \\% & N & \\% \\\\",
        "\\midrule",
    ]

    for row in rows:
        latex_lines.append(
            f"{row['label']} & {row['train_n']} & {row['train_pct']} & {row['temporal_n']} & {row['temporal_pct']} \\\\" 
        )

    latex_lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    latex = "\n".join(latex_lines)
    output_path.write_text(latex)

    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.benchmark.tables.cohort_table <config.yaml>")
        raise SystemExit(1)

    path = generate_cohort_table(sys.argv[1])
    print(f"Saved cohort table to {path}")
