"""Tests for data loading utilities."""

import numpy as np
import pytest

from src.benchmark.data import _create_strata, extract_sensitive_attribute, stratified_split


def _mock_df(size=200, seed=42):
    import pandas as pd

    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "SEX": rng.choice([1, 2], size=size),
            "RAC1P": rng.choice(range(1, 10), size=size),
            "AGEP": rng.randint(18, 65, size=size),
        }
    )


def test_extract_sensitive_attribute_sex():
    """SEX=1 (Male) should map to A=1; SEX=2 (Female) should map to A=0."""
    df = _mock_df()
    A = extract_sensitive_attribute(df, "SEX")
    # Male (SEX==1) -> A=1
    assert (A[df["SEX"] == 1] == 1).all()
    # Female (SEX==2) -> A=0
    assert (A[df["SEX"] == 2] == 0).all()


def test_extract_sensitive_attribute_rac1p():
    """RAC1P=1 (White) should map to A=1; others should map to A=0."""
    df = _mock_df()
    A = extract_sensitive_attribute(df, "RAC1P")
    assert (A[df["RAC1P"] == 1] == 1).all()
    assert (A[df["RAC1P"] != 1] == 0).all()


def test_extract_sensitive_attribute_other():
    """Other columns should use above-median encoding."""
    df = _mock_df()
    A = extract_sensitive_attribute(df, "AGEP")
    median_val = df["AGEP"].median()
    expected = (df["AGEP"] > median_val).astype(int)
    np.testing.assert_array_equal(A, expected)


def test_create_strata_unique_values():
    """_create_strata should produce 4 unique strata for binary y and A."""
    y = np.array([0, 0, 1, 1], dtype=int)
    A = np.array([0, 1, 0, 1], dtype=int)
    strata = _create_strata(y, A)
    assert set(strata.tolist()) == {0, 1, 2, 3}


def test_create_strata_encoding():
    """Strata encoding: y*2 + A gives unique integer per (y, A) pair."""
    for y_val in [0, 1]:
        for a_val in [0, 1]:
            y = np.array([y_val], dtype=int)
            A = np.array([a_val], dtype=int)
            expected = y_val * 2 + a_val
            assert _create_strata(y, A)[0] == expected


def test_stratified_split_sizes():
    """Split sizes should match the requested ratios."""
    rng = np.random.RandomState(0)
    n = 1000
    X = rng.randn(n, 5)
    y = rng.randint(0, 2, size=n)
    A = rng.randint(0, 2, size=n)

    split = [0.6, 0.1, 0.3]
    X_train, X_val, X_test, y_train, y_val, y_test, A_train, A_val, A_test = stratified_split(
        X, y, A, seed=42, split=split
    )

    assert len(y_train) == pytest.approx(n * 0.6, abs=5)
    assert len(y_val) == pytest.approx(n * 0.1, abs=5)
    assert len(y_test) == pytest.approx(n * 0.3, abs=5)


def test_stratified_split_preserves_class_balance():
    """Each split should preserve the overall class proportion."""
    rng = np.random.RandomState(0)
    n = 1000
    X = rng.randn(n, 5)
    y = rng.choice([0, 1], size=n, p=[0.7, 0.3])
    A = rng.choice([0, 1], size=n, p=[0.6, 0.4])

    X_train, X_val, X_test, y_train, y_val, y_test, A_train, A_val, A_test = stratified_split(
        X, y, A, seed=42, split=[0.6, 0.1, 0.3]
    )

    overall_pos_rate = y.mean()
    assert y_train.mean() == pytest.approx(overall_pos_rate, abs=0.05)
    assert y_val.mean() == pytest.approx(overall_pos_rate, abs=0.1)
    assert y_test.mean() == pytest.approx(overall_pos_rate, abs=0.05)


def test_stratified_split_ratios_must_sum_to_one():
    """stratified_split should raise ValueError if ratios don't sum to 1."""
    rng = np.random.RandomState(0)
    X = rng.randn(100, 3)
    y = rng.randint(0, 2, size=100)
    A = rng.randint(0, 2, size=100)

    with pytest.raises(ValueError, match="sum to 1.0"):
        stratified_split(X, y, A, seed=0, split=[0.5, 0.2, 0.2])


def test_stratified_split_no_overlap():
    """All three splits should be disjoint (no overlapping indices)."""
    rng = np.random.RandomState(0)
    n = 300
    X = rng.randn(n, 3)
    y = rng.randint(0, 2, size=n)
    A = rng.randint(0, 2, size=n)

    X_train, X_val, X_test, y_train, y_val, y_test, _, _, _ = stratified_split(
        X, y, A, seed=42, split=[0.6, 0.1, 0.3]
    )

    total = len(y_train) + len(y_val) + len(y_test)
    assert total == n
