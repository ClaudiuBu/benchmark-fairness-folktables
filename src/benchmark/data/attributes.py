"""Sensitive attribute extraction utilities."""

import pandas as pd
import numpy as np


def extract_sensitive_attribute(X_df: pd.DataFrame, sensitive_attribute: str) -> np.ndarray:
    """Extract a single sensitive attribute from data.
    
    Args:
        X_df: Feature dataframe
        sensitive_attribute: Column name to extract
        
    Returns:
        Binary array (0/1) representing the sensitive attribute
    """
    if sensitive_attribute == "SEX":
        return (X_df["SEX"] == 1).astype(int)
    if sensitive_attribute == "RAC1P":
        return (X_df["RAC1P"] == 1).astype(int)
    return (X_df[sensitive_attribute] > X_df[sensitive_attribute].median()).astype(int)


def extract_multiple_sensitive_attributes(X_df: pd.DataFrame) -> dict:
    """Extract both SEX and RACE (RAC1P) as sensitive attributes.
    
    Returns:
        dict: {"SEX": binary array, "RAC1P": binary array}
            SEX: 1 = Male, 0 = Female
            RAC1P: 1 = White, 0 = Non-White
    """
    result = {}
    
    if "SEX" in X_df.columns:
        result["SEX"] = (X_df["SEX"] == 1).astype(int)
    
    if "RAC1P" in X_df.columns:
        result["RAC1P"] = (X_df["RAC1P"] == 1).astype(int)
    
    return result
