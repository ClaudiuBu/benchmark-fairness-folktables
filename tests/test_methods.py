"""Tests for benchmark methods (reweighing, threshold optimization, Lagrangian)."""

import numpy as np
import pytest

from src.benchmark.methods import (
    choose_thresholds_equalized_odds,
    kamiran_calders_weights,
    make_model,
    train_with_lagrangian,
)
from src.fairness import demographic_parity, equalized_odds_gap


def _biased_dataset(seed=42, n=400):
    """Create a synthetic dataset with demographic parity gap."""
    rng = np.random.RandomState(seed)
    A = rng.choice([0, 1], size=n, p=[0.5, 0.5])
    # A=1 group has higher P(Y=1)
    y = np.where(
        A == 1,
        rng.binomial(1, 0.7, n),
        rng.binomial(1, 0.3, n),
    )
    X = rng.randn(n, 5)
    X[A == 1] += 0.5
    return X, y, A


def test_kamiran_calders_weights_shape():
    """Weights should have the same shape as y."""
    _, y, A = _biased_dataset()
    weights = kamiran_calders_weights(y, A)
    assert weights.shape == y.shape


def test_kamiran_calders_weights_formula():
    """Each group weight should equal P(A)*P(Y)/P(A,Y)."""
    _, y, A = _biased_dataset()
    weights = kamiran_calders_weights(y, A)
    for a in [0, 1]:
        for c in [0, 1]:
            mask = (A == a) & (y == c)
            if mask.sum() == 0:
                continue
            p_a = np.mean(A == a)
            p_y = np.mean(y == c)
            p_ay = np.mean(mask)
            expected = np.clip(p_a * p_y / p_ay, 0.1, 10.0)
            actual = weights[mask][0]
            assert actual == pytest.approx(expected, rel=1e-6), (
                f"Weight mismatch for A={a}, Y={c}: expected {expected}, got {actual}"
            )


def test_kamiran_calders_weights_clipped():
    """Weights should be clipped to [0.1, 10.0]."""
    _, y, A = _biased_dataset()
    weights = kamiran_calders_weights(y, A)
    assert weights.min() >= 0.1
    assert weights.max() <= 10.0


def test_choose_thresholds_equalized_odds_reduces_gap():
    """Optimal thresholds should reduce EO gap compared to threshold 0.5."""
    X, y_true, A = _biased_dataset()
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = make_model(42)
    model.fit(X_scaled, y_true)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    grid = np.linspace(0.1, 0.9, 9).tolist()
    thresholds = choose_thresholds_equalized_odds(y_true, y_proba, A, grid)

    y_pred_05 = (y_proba >= 0.5).astype(int)
    y_pred_opt = np.where(A == 0, y_proba >= thresholds[0], y_proba >= thresholds[1]).astype(int)

    eo_05 = equalized_odds_gap(y_true, y_pred_05, A)
    eo_opt = equalized_odds_gap(y_true, y_pred_opt, A)
    assert eo_opt <= eo_05 + 1e-9, f"EO gap not reduced: {eo_opt} vs {eo_05}"


def test_choose_thresholds_equalized_odds_single_group():
    """Should return default thresholds when only one group is present."""
    X, y_true, _ = _biased_dataset()
    A_single = np.ones(len(y_true), dtype=int)
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = make_model(42)
    model.fit(X_scaled, y_true)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    thresholds = choose_thresholds_equalized_odds(y_true, y_proba, A_single, [0.5])
    assert thresholds == {0: 0.5, 1: 0.5}


def test_train_with_lagrangian_reduces_dp_gap():
    """Lagrangian training should reduce the demographic parity gap."""
    X, y, A = _biased_dataset()
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    baseline = make_model(42)
    baseline.fit(X_scaled, y)
    dp_baseline = abs(demographic_parity(baseline.predict(X_scaled), A))

    lag_model = train_with_lagrangian(X_scaled, y, A, seed=42, num_iters=8, lr=0.1)
    dp_lag = abs(demographic_parity(lag_model.predict(X_scaled), A))

    assert dp_lag < dp_baseline, (
        f"Lagrangian did not reduce DP gap: {dp_lag:.4f} vs baseline {dp_baseline:.4f}"
    )


def test_make_model_reproducible():
    """make_model with the same seed should produce reproducible results."""
    X, y, _ = _biased_dataset()
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model1 = make_model(0)
    model1.fit(X_scaled, y)
    pred1 = model1.predict(X_scaled)

    model2 = make_model(0)
    model2.fit(X_scaled, y)
    pred2 = model2.predict(X_scaled)

    np.testing.assert_array_equal(pred1, pred2)
