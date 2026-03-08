"""Data loading and splitting utilities for Folktables benchmarks."""

import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSIncome, ACSEmployment
from sklearn.model_selection import StratifiedKFold, train_test_split


def extract_sensitive_attribute(X_df, sensitive_attribute: str):
    """Extract a binary sensitive attribute vector from the feature DataFrame.

    Encoding conventions (matching ACS codebook):
    - SEX:   1 = Male  → A=1,  2 = Female → A=0
    - RAC1P: 1 = White → A=1,  otherwise  → A=0
    - Other: above-median → A=1, at-or-below-median → A=0
    """
    if sensitive_attribute == "SEX":
        return (X_df["SEX"] == 1).astype(int)
    if sensitive_attribute == "RAC1P":
        return (X_df["RAC1P"] == 1).astype(int)
    return (X_df[sensitive_attribute] > X_df[sensitive_attribute].median()).astype(int)


def _create_strata(y, A):
    """Create combined stratification labels from class and sensitive attribute.

    Encodes the joint (y, A) pair as a single integer so that stratified
    splits preserve both class balance and sensitive-attribute proportions.
    """
    return (y.astype(int) * 2 + A.astype(int)).astype(int)


def load_folktables(
    task: str,
    states,
    years,
    sensitive_attribute: str,
    max_samples: int = None,
    random_state: int = 42,
    keep_sensitive_cols: bool = False,
):
    """Load and pool Folktables data across years and states.

    Args:
        task: 'income' or 'employment'.
        states: Iterable of state abbreviations (e.g. ['CA']).
        years: Iterable of survey years (e.g. [2014, 2015]).
        sensitive_attribute: Name of the column used as sensitive attribute.
        max_samples: If set, randomly subsample to this many rows.
        random_state: RNG seed for reproducible subsampling.
        keep_sensitive_cols: When True the sensitive-attribute column(s) are
            *retained* in the returned DataFrame so callers can compute
            secondary-attribute fairness metrics.  When False (default) the
            primary sensitive-attribute column is dropped from X.

    Returns:
        (X_df, y, A) where
            X_df  – pd.DataFrame of features (possibly with sensitive cols),
            y     – np.ndarray of integer labels,
            A     – np.ndarray of binary sensitive-attribute values.
    """
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
            if not keep_sensitive_cols:
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


def _load_year_raw(task_cls, data_source, states, sensitive_attribute, keep_sensitive_cols):
    """Load data for all states in a single survey year."""
    all_X = []
    all_y = []
    all_A = []

    for state in states:
        acs_data = data_source.get_data(states=[state], download=True)
        X_df, y, _ = task_cls.df_to_pandas(acs_data)
        A = extract_sensitive_attribute(X_df, sensitive_attribute)
        if not keep_sensitive_cols:
            X_df = X_df.drop(columns=[sensitive_attribute], errors="ignore")

        all_X.append(X_df)
        all_y.append(y.values)
        all_A.append(A.values)

    X_year = pd.concat(all_X, axis=0).reset_index(drop=True)
    y_year = np.concatenate(all_y).astype(int).ravel()
    A_year = np.concatenate(all_A).astype(int)
    return X_year, y_year, A_year


def _load_folktables_by_year(
    task: str,
    states,
    years,
    sensitive_attribute: str,
    max_samples_per_year: int = None,
    random_state: int = 42,
    keep_sensitive_cols: bool = False,
):
    """Load Folktables data split by survey year."""
    task_cls = ACSIncome if task == "income" else ACSEmployment
    datasets = {}
    base_columns = None

    for year in years:
        data_source = ACSDataSource(
            survey_year=str(year), horizon="1-Year", survey="person"
        )
        X_year, y_year, A_year = _load_year_raw(
            task_cls, data_source, states, sensitive_attribute, keep_sensitive_cols
        )

        if base_columns is None:
            base_columns = X_year.columns
        else:
            X_year = X_year.reindex(columns=base_columns, fill_value=0)

        if max_samples_per_year is not None and max_samples_per_year < len(y_year):
            rng = np.random.RandomState(random_state + int(year))
            idx = rng.choice(len(y_year), size=max_samples_per_year, replace=False)
            X_year = X_year.iloc[idx].reset_index(drop=True)
            y_year = y_year[idx]
            A_year = A_year[idx]

        datasets[year] = (X_year, y_year, A_year)

    return datasets, base_columns


def _split_into_quarters(X_year, y_year, A_year, random_state: int):
    """Split a year of data into four stratified quarters.

    Stratification uses a combined label/sensitive-attribute stratum so that
    each quarter preserves the overall class and group balance.
    """
    strata = _create_strata(y_year, A_year)
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
    keep_sensitive_cols: bool = False,
):
    """Load Folktables data keyed by year or simulated quarterly periods.

    Args:
        task: 'income' or 'employment'.
        states: Iterable of state abbreviations.
        years: Iterable of survey years to include.
        sensitive_attribute: Column name used as sensitive attribute.
        frequency: 'year' (one entry per survey year) or 'quarter'
            (four stratified splits per survey year, keyed by
            year + (quarter-1)/4, e.g. 2015.0, 2015.25, 2015.5, 2015.75).
        max_samples_per_year: Optional cap on samples per period.
        random_state: RNG seed.
        keep_sensitive_cols: When True the sensitive-attribute column(s) are
            retained in the returned DataFrames.

    Returns:
        (datasets, base_columns) where
            datasets – dict mapping period key → (X_df, y, A),
            base_columns – pd.Index of feature column names.
    """
    if frequency == "year":
        return _load_folktables_by_year(
            task=task,
            states=states,
            years=years,
            sensitive_attribute=sensitive_attribute,
            max_samples_per_year=max_samples_per_year,
            random_state=random_state,
            keep_sensitive_cols=keep_sensitive_cols,
        )

    if frequency != "quarter":
        raise ValueError(f"Unsupported frequency: {frequency!r}. Choose 'year' or 'quarter'.")

    task_cls = ACSIncome if task == "income" else ACSEmployment
    datasets = {}
    base_columns = None

    for year in years:
        data_source = ACSDataSource(
            survey_year=str(year), horizon="1-Year", survey="person"
        )
        X_year, y_year, A_year = _load_year_raw(
            task_cls, data_source, states, sensitive_attribute, keep_sensitive_cols
        )

        if base_columns is None:
            base_columns = X_year.columns
        else:
            X_year = X_year.reindex(columns=base_columns, fill_value=0)

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
    """Perform a stratified train/validation/test split.

    Stratification is done on a combined stratum of (y, A) to preserve both
    class and sensitive-attribute proportions in every split.

    Args:
        X: Feature array (numpy array or similar).
        y: Label array.
        A: Sensitive-attribute array.
        seed: Random seed for reproducibility.
        split: Iterable of three ratios [train, val, test] that sum to 1.0.

    Returns:
        (X_train, X_val, X_test,
         y_train, y_val, y_test,
         A_train, A_val, A_test)
    """
    train_ratio, val_ratio, test_ratio = split
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio:.4f}"
        )

    strata = _create_strata(y, A)
    X_train, X_temp, y_train, y_temp, A_train, A_temp = train_test_split(
        X,
        y,
        A,
        test_size=1 - train_ratio,
        random_state=seed,
        stratify=strata,
    )

    val_size = val_ratio / (val_ratio + test_ratio)
    strata_temp = _create_strata(y_temp, A_temp)
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
