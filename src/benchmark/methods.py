"""Method-specific helpers for bias-mitigation benchmarks."""

import numpy as np
from sklearn.linear_model import SGDClassifier

from src.fairness import demographic_parity, equalized_odds_gap


def make_model(random_state: int) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        learning_rate="optimal",
        max_iter=1000,
        tol=1e-3,
        random_state=random_state,
    )


def kamiran_calders_weights(y, A, eps: float = 1e-6):
    y = y.astype(int)
    A = A.astype(int)

    p_a = {a: max(np.mean(A == a), eps) for a in [0, 1]}
    p_y = {c: max(np.mean(y == c), eps) for c in [0, 1]}

    p_ay = {}
    for a in [0, 1]:
        for c in [0, 1]:
            p_ay[(a, c)] = max(np.mean((A == a) & (y == c)), eps)

    weights = np.zeros_like(y, dtype=float)
    for a in [0, 1]:
        for c in [0, 1]:
            mask = (A == a) & (y == c)
            weights[mask] = (p_a[a] * p_y[c]) / p_ay[(a, c)]

    return np.clip(weights, 0.1, 10.0)


def choose_thresholds_equalized_odds(y_true, y_proba, A, grid):
    if len(np.unique(A)) < 2:
        return {0: 0.5, 1: 0.5}

    best_gap = None
    best_acc = None
    best_t = {0: 0.5, 1: 0.5}

    for t0 in grid:
        for t1 in grid:
            y_pred = np.where(A == 0, y_proba >= t0, y_proba >= t1).astype(int)
            gap = equalized_odds_gap(y_true, y_pred, A)
            acc = np.mean(y_pred == y_true)

            if best_gap is None or gap < best_gap or (np.isclose(gap, best_gap) and acc > best_acc):
                best_gap = gap
                best_acc = acc
                best_t = {0: float(t0), 1: float(t1)}

    return best_t


def train_with_lagrangian(
    X_train,
    y_train,
    A_train,
    seed: int,
    num_iters: int,
    lr: float,
):
    lambda_mult = 0.0
    weights = np.ones_like(y_train, dtype=float)

    for i in range(num_iters):
        model = make_model(seed + i)
        model.fit(X_train, y_train, sample_weight=weights)
        y_pred = model.predict(X_train)
        dp_gap = demographic_parity(y_pred, A_train)

        lambda_mult += lr * dp_gap
        lambda_mult = float(np.clip(lambda_mult, -2.0, 2.0))

        minority_rate = np.mean(A_train == 1)
        majority_rate = np.mean(A_train == 0)
        if majority_rate == 0 or minority_rate == 0:
            break

        weights = np.ones_like(y_train, dtype=float)
        weights[A_train == 1] *= (1.0 + lambda_mult)
        weights[A_train == 0] *= (1.0 - lambda_mult * minority_rate / max(majority_rate, 1e-6))
        weights = np.clip(weights, 0.1, 10.0)

    final_model = make_model(seed + 999)
    final_model.fit(X_train, y_train, sample_weight=weights)
    return final_model
