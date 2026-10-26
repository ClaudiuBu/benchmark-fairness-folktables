"""Data loading utilities for Folktables benchmarks."""

import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSIncome, ACSEmployment
from sklearn.model_selection import StratifiedKFold

from src.benchmark.data.attributes import extract_sensitive_attribute


def _get_task_class(task: str):
    """Get the appropriate task class (Income or Employment)."""
    if task == "income":
        return ACSIncome
    elif task == "employment":
        return ACSEmployment
    else:
        raise ValueError(f"Unknown task: {task}")


def _load_year_data(
    task: str,
    states: list,
    year: int,
    sensitive_attribute: str,
    keep_sensitive_cols: bool = False,
) -> tuple:
    """Load data from ACS API for a specific year and states.
    
    Returns:
        (X_df, y, A) where A is the sensitive attribute
    """
    task_cls = _get_task_class(task)
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
        
        if not keep_sensitive_cols:
            X_df = X_df.drop(columns=[sensitive_attribute], errors="ignore")

        all_X.append(X_df)
        all_y.append(y.values)
        all_A.append(A.values)

    X = pd.concat(all_X, axis=0).reset_index(drop=True)
    y = np.concatenate(all_y).astype(int).ravel()
    A = np.concatenate(all_A).astype(int)

    return X, y, A


def _sample_data(
    X: pd.DataFrame,
    y: np.ndarray,
    A: np.ndarray,
    max_samples: int,
    random_state: int,
) -> tuple:
    """Subsample data if max_samples is specified."""
    if max_samples is None or max_samples >= len(y):
        return X, y, A
    
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(y), size=max_samples, replace=False)
    return (
        X.iloc[idx].reset_index(drop=True),
        y[idx],
        A[idx],
    )


def load_folktables(
    task: str,
    states: list,
    years: list,
    sensitive_attribute: str,
    max_samples: int = None,
    random_state: int = 42,
    keep_sensitive_cols: bool = False,
) -> tuple:
    """Load Folktables data aggregated across years and states.
    
    Args:
        task: "income" or "employment"
        states: List of state codes
        years: List of years to load
        sensitive_attribute: Name of sensitive attribute column
        max_samples: Maximum samples to use (random sampling if exceeded)
        random_state: Random seed for sampling
        keep_sensitive_cols: Whether to keep sensitive attribute columns in X
        
    Returns:
        (X_df, y, A) where X_df is DataFrame, y and A are numpy arrays
    """
    all_X = []
    all_y = []
    all_A = []

    for year in years:
        X, y, A = _load_year_data(task, states, year, sensitive_attribute, keep_sensitive_cols)
        all_X.append(X)
        all_y.append(y)
        all_A.append(A)

    X_all = pd.concat(all_X, axis=0).reset_index(drop=True)
    y_all = np.concatenate(all_y).astype(int).ravel()
    A_all = np.concatenate(all_A).astype(int)

    return _sample_data(X_all, y_all, A_all, max_samples, random_state)


def load_folktables_by_year(
    task: str,
    states: list,
    years: list,
    sensitive_attribute: str,
    max_samples_per_year: int = None,
    random_state: int = 42,
    keep_sensitive_cols: bool = False,
) -> tuple:
    """Load Folktables data separately for each year.
    
    Returns:
        (datasets_dict, base_columns) where datasets_dict[year] = (X, y, A)
    """
    datasets = {}
    base_columns = None

    for year in years:
        X, y, A = _load_year_data(task, states, year, sensitive_attribute, keep_sensitive_cols=keep_sensitive_cols)
        
        if base_columns is None:
            base_columns = X.columns
        else:
            X = X.reindex(columns=base_columns, fill_value=0)

        X, y, A = _sample_data(X, y, A, max_samples_per_year, random_state + int(year))
        datasets[year] = (X, y, A)

    return datasets, base_columns


def _split_into_within_year_folds(
    X_year: pd.DataFrame,
    y_year: np.ndarray,
    A_year: np.ndarray,
    random_state: int,
) -> dict:
    """Split a year's data into four stratified within-year folds."""
    strata = (y_year.astype(int) * 2 + A_year.astype(int)).astype(int)
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
    folds = {}

    for idx, (_, test_idx) in enumerate(skf.split(X_year, strata), start=1):
        X_q = X_year.iloc[test_idx].reset_index(drop=True)
        y_q = y_year[test_idx]
        A_q = A_year[test_idx]
        folds[idx] = (X_q, y_q, A_q)

    return folds


def load_folktables_by_period(
    task: str,
    states: list,
    years: list,
    sensitive_attribute: str,
    frequency: str = "year",
    max_samples_per_year: int = None,
    random_state: int = 42,
    keep_sensitive_cols: bool = False,
) -> tuple:
    """Load Folktables data by year or simulated stratified within-year folds.
    
    Args:
        frequency: "year" (default) or "within_year_fold". The legacy value
            "quarter" is accepted as an alias, but these are not calendar
            quarters.
        
    Returns:
        (datasets_dict, base_columns) where datasets_dict[period_key] = (X, y, A)
    """
    if frequency == "year":
        return load_folktables_by_year(
            task=task,
            states=states,
            years=years,
            sensitive_attribute=sensitive_attribute,
            max_samples_per_year=max_samples_per_year,
            random_state=random_state,
            keep_sensitive_cols=keep_sensitive_cols,
        )

    if frequency == "quarter":
        frequency = "within_year_fold"

    if frequency != "within_year_fold":
        raise ValueError(f"Unsupported frequency: {frequency}")

    # Within-year split: load each year then subdivide into stratified folds.
    datasets = {}
    base_columns = None

    for year in years:
        X, y, A = _load_year_data(task, states, year, sensitive_attribute, keep_sensitive_cols=keep_sensitive_cols)
        
        if base_columns is None:
            base_columns = X.columns
        else:
            X = X.reindex(columns=base_columns, fill_value=0)

        X, y, A = _sample_data(X, y, A, max_samples_per_year, random_state + int(year))
        
        folds = _split_into_within_year_folds(X, y, A, random_state + int(year))
        for fold_idx, (X_q, y_q, A_q) in folds.items():
            period_key = year + (fold_idx - 1) / 4
            datasets[period_key] = (X_q, y_q, A_q)

    return datasets, base_columns
