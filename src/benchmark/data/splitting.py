"""Data splitting utilities for train/validation/test splits."""

import numpy as np
from sklearn.model_selection import train_test_split


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    A: np.ndarray,
    seed: int,
    split: list,
) -> tuple:
    """Stratified train/validation/test split based on labels and sensitive attributes.
    
    Guarantees balanced distribution of labels and sensitive attributes across splits.
    
    Args:
        X: Feature array
        y: Labels array
        A: Sensitive attribute array
        seed: Random seed
        split: [train_ratio, val_ratio, test_ratio] that sum to 1.0
        
    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test, A_train, A_val, A_test)
    """
    train_ratio, val_ratio, test_ratio = split
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    # Create strata based on label and sensitive attribute
    strata = (y.astype(int) * 2 + A.astype(int)).astype(int)
    
    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp, A_train, A_temp = train_test_split(
        X,
        y,
        A,
        test_size=1 - train_ratio,
        random_state=seed,
        stratify=strata,
    )

    # Second split: val vs test
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
