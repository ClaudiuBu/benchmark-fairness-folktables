"""Data loading and splitting utilities for Folktables benchmarks."""

import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSIncome, ACSEmployment
from sklearn.model_selection import StratifiedKFold, train_test_split


def extract_sensitive_attribute(X_df, sensitive_attribute: str):
    if sensitive_attribute == "SEX":
        return (X_df["SEX"] == 1).astype(int)
    if sensitive_attribute == "RAC1P":
        return (X_df["RAC1P"] == 1).astype(int)
    return (X_df[sensitive_attribute] > X_df[sensitive_attribute].median()).astype(int)


def load_folktables(
    task: str,
    states,
    years,
    sensitive_attribute: str,
    max_samples: int = None,
    random_state: int = 42,
):
    task_cls = ACSIncome if task == "income" else ACSEmployment
    all_X = []
    all_y = []
    all_A = []

    for year in years:
        data_source = ACSDataSource(
            survey_year=str(year), horizon="1-Year", survey="person"
        )
        for state in states:
            acs_data = data_source.get_data(states=[state], download=True)
            X_df, y, _ = task_cls.df_to_pandas(acs_data)
            A = extract_sensitive_attribute(X_df, sensitive_attribute)
            X_df = X_df.drop(columns=[sensitive_attribute], errors="ignore")

            all_X.append(X_df)
            all_y.append(y.values)
            all_A.append(A.values)

    X_all = pd.concat(all_X, axis=0).reset_index(drop=True)
    y_all = np.concatenate(all_y).astype(int).ravel()
    A_all = np.concatenate(all_A).astype(int)

    if max_samples is not None and max_samples < len(y_all):
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(y_all), size=max_samples, replace=False)
        X_all = X_all.iloc[idx].reset_index(drop=True)
        y_all = y_all[idx]
        A_all = A_all[idx]

    return X_all, y_all, A_all


def load_folktables_by_year(
    task: str,
    states,
    years,
    sensitive_attribute: str,
    max_samples_per_year: int = None,
    random_state: int = 42,
):
    task_cls = ACSIncome if task == "income" else ACSEmployment
    datasets = {}
    base_columns = None

    for year in years:
        all_X = []
        all_y = []
        all_A = []

        data_source = ACSDataSource(
            survey_year=str(year), horizon="1-Year", survey="person"
        )
        for state in states:
            acs_data = data_source.get_data(states=[state], download=True)
            X_df, y, _ = task_cls.df_to_pandas(acs_data)
            A = extract_sensitive_attribute(X_df, sensitive_attribute)
            X_df = X_df.drop(columns=[sensitive_attribute], errors="ignore")

            all_X.append(X_df)
            all_y.append(y.values)
            all_A.append(A.values)

        X_year = pd.concat(all_X, axis=0).reset_index(drop=True)
        if base_columns is None:
            base_columns = X_year.columns
        else:
            X_year = X_year.reindex(columns=base_columns, fill_value=0)

        y_year = np.concatenate(all_y).astype(int).ravel()
        A_year = np.concatenate(all_A).astype(int)

        if max_samples_per_year is not None and max_samples_per_year < len(y_year):
            rng = np.random.RandomState(random_state + int(year))
            idx = rng.choice(len(y_year), size=max_samples_per_year, replace=False)
            X_year = X_year.iloc[idx].reset_index(drop=True)
            y_year = y_year[idx]
            A_year = A_year[idx]

        datasets[year] = (X_year, y_year, A_year)

    return datasets, base_columns


def _split_into_quarters(X_year, y_year, A_year, random_state: int):
    """Split a year into four stratified quarters using labels and sensitive attribute."""
    strata = (y_year.astype(int) * 2 + A_year.astype(int)).astype(int)
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
    quarters = {}

    for idx, (_, test_idx) in enumerate(skf.split(X_year, strata), start=1):
        X_q = X_year.iloc[test_idx].reset_index(drop=True)
        y_q = y_year[test_idx]
        A_q = A_year[test_idx]
        quarters[idx] = (X_q, y_q, A_q)

    return quarters


def load_folktables_by_period(
    task: str,
    states,
    years,
    sensitive_attribute: str,
    frequency: str = "year",
    max_samples_per_year: int = None,
    random_state: int = 42,
):
    """Load Folktables data by year or simulated quarterly periods."""
    if frequency == "year":
        return load_folktables_by_year(
            task=task,
            states=states,
            years=years,
            sensitive_attribute=sensitive_attribute,
            max_samples_per_year=max_samples_per_year,
            random_state=random_state,
        )

    if frequency != "quarter":
        raise ValueError(f"Unsupported frequency: {frequency}")

    task_cls = ACSIncome if task == "income" else ACSEmployment
    datasets = {}
    base_columns = None

    for year in years:
        all_X = []
        all_y = []
        all_A = []

        data_source = ACSDataSource(
            survey_year=str(year), horizon="1-Year", survey="person"
        )
        for state in states:
            acs_data = data_source.get_data(states=[state], download=True)
            X_df, y, _ = task_cls.df_to_pandas(acs_data)
            A = extract_sensitive_attribute(X_df, sensitive_attribute)
            X_df = X_df.drop(columns=[sensitive_attribute], errors="ignore")

            all_X.append(X_df)
            all_y.append(y.values)
            all_A.append(A.values)

        X_year = pd.concat(all_X, axis=0).reset_index(drop=True)
        if base_columns is None:
            base_columns = X_year.columns
        else:
            X_year = X_year.reindex(columns=base_columns, fill_value=0)

        y_year = np.concatenate(all_y).astype(int).ravel()
        A_year = np.concatenate(all_A).astype(int)

        if max_samples_per_year is not None and max_samples_per_year < len(y_year):
            rng = np.random.RandomState(random_state + int(year))
            idx = rng.choice(len(y_year), size=max_samples_per_year, replace=False)
            X_year = X_year.iloc[idx].reset_index(drop=True)
            y_year = y_year[idx]
            A_year = A_year[idx]

        quarters = _split_into_quarters(X_year, y_year, A_year, random_state + int(year))
        for quarter_idx, (X_q, y_q, A_q) in quarters.items():
            period_key = year + (quarter_idx - 1) / 4
            datasets[period_key] = (X_q, y_q, A_q)

    return datasets, base_columns


def stratified_split(X, y, A, seed: int, split):
    train_ratio, val_ratio, test_ratio = split
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    strata = (y.astype(int) * 2 + A.astype(int)).astype(int)
    X_train, X_temp, y_train, y_temp, A_train, A_temp = train_test_split(
        X,
        y,
        A,
        test_size=1 - train_ratio,
        random_state=seed,
        stratify=strata,
    )

    val_size = val_ratio / (val_ratio + test_ratio)
    strata_temp = (y_temp.astype(int) * 2 + A_temp.astype(int)).astype(int)
    X_val, X_test, y_val, y_test, A_val, A_test = train_test_split(
        X_temp,
        y_temp,
        A_temp,
        test_size=1 - val_size,
        random_state=seed,
        stratify=strata_temp,
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train.ravel(),
        y_val.ravel(),
        y_test.ravel(),
        A_train,
        A_val,
        A_test,
    )
