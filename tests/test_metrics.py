"""Tests for benchmark metrics computation."""

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.benchmark.metrics import compute_metrics, METRIC_NAMES


def _make_sample_data(seed=42, n=200):
    rng = np.random.RandomState(seed)
    y_true = rng.randint(0, 2, size=n)
    y_proba = np.clip(y_true * 0.6 + rng.randn(n) * 0.2 + 0.2, 0.0, 1.0)
    y_pred = (y_proba >= 0.5).astype(int)
    A = rng.randint(0, 2, size=n)
    return y_true, y_pred, y_proba, A


def test_compute_metrics_returns_all_keys():
    """compute_metrics should return all expected metric keys."""
    y_true, y_pred, y_proba, A = _make_sample_data()
    metrics = compute_metrics(y_true, y_pred, y_proba, A)
    for name in METRIC_NAMES:
        assert name in metrics, f"Missing metric: {name}"


def test_brier_score_matches_sklearn():
    """Brier score should match sklearn's brier_score_loss."""
    y_true, y_pred, y_proba, A = _make_sample_data()
    metrics = compute_metrics(y_true, y_pred, y_proba, A)
    expected = brier_score_loss(y_true, y_proba)
    assert metrics["brier_score"] == pytest.approx(expected, abs=1e-10)


def test_auc_matches_sklearn():
    """AUC should match sklearn's roc_auc_score."""
    y_true, y_pred, y_proba, A = _make_sample_data()
    metrics = compute_metrics(y_true, y_pred, y_proba, A)
    expected = roc_auc_score(y_true, y_proba)
    assert metrics["auc"] == pytest.approx(expected, abs=1e-10)


def test_accuracy_correct():
    """Accuracy should equal fraction of correct predictions."""
    y_true, y_pred, y_proba, A = _make_sample_data()
    metrics = compute_metrics(y_true, y_pred, y_proba, A)
    expected = float(np.mean(y_pred == y_true))
    assert metrics["accuracy"] == pytest.approx(expected, abs=1e-10)


def test_dp_gap_is_nonnegative():
    """dp_gap should always be non-negative (absolute value)."""
    y_true, y_pred, y_proba, A = _make_sample_data()
    metrics = compute_metrics(y_true, y_pred, y_proba, A)
    assert metrics["dp_gap"] >= 0.0


def test_eo_gap_is_nonnegative():
    """eo_gap should always be non-negative (max of absolute values)."""
    y_true, y_pred, y_proba, A = _make_sample_data()
    metrics = compute_metrics(y_true, y_pred, y_proba, A)
    assert np.isnan(metrics["eo_gap"]) or metrics["eo_gap"] >= 0.0


def test_auc_nan_when_single_class():
    """AUC should be nan when y_true has only one class."""
    y_true = np.zeros(50, dtype=int)
    y_proba = np.random.rand(50)
    y_pred = (y_proba >= 0.5).astype(int)
    A = np.random.randint(0, 2, size=50)
    metrics = compute_metrics(y_true, y_pred, y_proba, A)
    assert np.isnan(metrics["auc"])


def test_metrics_in_valid_range():
    """Accuracy, f1, sensitivity and AUC should be in [0, 1]."""
    y_true, y_pred, y_proba, A = _make_sample_data()
    metrics = compute_metrics(y_true, y_pred, y_proba, A)
    for name in ["accuracy", "f1_score", "sensitivity"]:
        assert 0.0 <= metrics[name] <= 1.0, f"{name} out of range: {metrics[name]}"
    if not np.isnan(metrics["auc"]):
        assert 0.0 <= metrics["auc"] <= 1.0
