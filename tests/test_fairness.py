"""Tests for fairness metric functions."""

import numpy as np
import pytest

from src.fairness import (
    demographic_parity,
    equalized_odds_gap,
    f1_score,
    observed_expected_gap,
    precision_score,
    recall_score,
)


def test_demographic_parity_perfect():
    """DP gap should be 0 when both groups have equal positive rates."""
    y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    A = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    assert demographic_parity(y_pred, A) == pytest.approx(0.0)


def test_demographic_parity_positive_gap():
    """DP gap should be positive when A=1 has higher positive rate."""
    y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    A = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    assert demographic_parity(y_pred, A) == pytest.approx(1.0)


def test_demographic_parity_negative_gap():
    """DP gap should be negative when A=0 has higher positive rate."""
    y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    A = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    assert demographic_parity(y_pred, A) == pytest.approx(-1.0)


def test_demographic_parity_single_group_nan():
    """DP gap should be nan when only one group is present."""
    y_pred = np.array([1, 0, 1, 0])
    A_single = np.ones(4, dtype=int)
    result = demographic_parity(y_pred, A_single)
    assert np.isnan(result)


def test_equalized_odds_gap_perfect():
    """EO gap should be 0 when both groups have equal TPR and FPR."""
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    A = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    assert equalized_odds_gap(y_true, y_pred, A) == pytest.approx(0.0)


def test_equalized_odds_gap_single_group_nan():
    """EO gap should be nan when only one group is present."""
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])
    A_single = np.ones(4, dtype=int)
    assert np.isnan(equalized_odds_gap(y_true, y_pred, A_single))


def test_equalized_odds_gap_nonnegative():
    """EO gap should always be >= 0."""
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, size=100)
    y_pred = rng.randint(0, 2, size=100)
    A = rng.randint(0, 2, size=100)
    gap = equalized_odds_gap(y_true, y_pred, A)
    assert np.isnan(gap) or gap >= 0.0


def test_observed_expected_gap_nan_single_group():
    """OE gap should be nan when only one group is present."""
    y_true = np.array([1, 0, 1, 0])
    y_proba = np.array([0.9, 0.1, 0.8, 0.2])
    A_single = np.ones(4, dtype=int)
    assert np.isnan(observed_expected_gap(y_true, y_proba, A_single))


def test_observed_expected_gap_zero_expected():
    """OE gap should be nan when expected count is zero for a group."""
    y_true = np.array([1, 0, 1, 0])
    y_proba = np.array([0.0, 0.0, 0.8, 0.2])  # A=1 has zero expected
    A = np.array([1, 1, 0, 0])
    result = observed_expected_gap(y_true, y_proba, A)
    assert np.isnan(result)


def test_precision_score_no_positives_predicted():
    """Precision should be 0 when no positives are predicted."""
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0, 0, 0, 0])
    assert precision_score(y_true, y_pred) == 0.0


def test_recall_score_no_positives_actual():
    """Recall should be 0 when there are no actual positives."""
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    assert recall_score(y_true, y_pred) == 0.0


def test_f1_score_perfect():
    """F1 score should be 1.0 for perfect predictions."""
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1])
    assert f1_score(y_true, y_pred) == pytest.approx(1.0)


def test_f1_score_no_predictions():
    """F1 score should be 0 when both precision and recall are 0."""
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0])
    assert f1_score(y_true, y_pred) == 0.0


def test_f1_harmonic_mean():
    """F1 score should equal 2*P*R/(P+R)."""
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 1])
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    expected_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    assert f1_score(y_true, y_pred) == pytest.approx(expected_f1)
