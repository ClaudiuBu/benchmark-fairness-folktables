"""Method-specific helpers for bias-mitigation benchmarks."""

import numpy as np
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import DemographicParity, EqualizedOdds, ExponentiatedGradient
from sklearn.linear_model import SGDClassifier

from src.fairness import demographic_parity, equalized_odds_gap


__all__ = [
    "make_model",
    "kamiran_calders_weights",
    "choose_thresholds_equalized_odds",
    "choose_thresholds_demographic_parity",
    "train_with_fairlearn_threshold_optimizer",
    "train_with_fairlearn_reductions",
    "train_with_lagrangian",
]


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


def _classification_utility(y_true, y_pred):
    accuracy = float(np.mean(y_pred == y_true))
    positive_rate = float(np.mean(y_pred))
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"accuracy": accuracy, "f1": float(f1), "positive_rate": positive_rate}


def _threshold_utility_floor(y_true, y_proba, grid, config):
    """Return utility constraints relative to the unconstrained 0.5 threshold."""
    cfg = config or {}
    baseline_pred = (y_proba >= 0.5).astype(int)
    baseline = _classification_utility(y_true, baseline_pred)

    min_positive_rate = max(
        float(cfg.get("min_positive_rate", 0.01)),
        float(cfg.get("min_positive_rate_ratio", 0.25)) * baseline["positive_rate"],
    )
    return {
        "baseline": baseline,
        "max_accuracy_drop": float(cfg.get("max_accuracy_drop", 0.05)),
        "max_f1_drop": float(cfg.get("max_f1_drop", 0.10)),
        "min_positive_rate": min_positive_rate,
        "fallback_to_baseline": bool(cfg.get("fallback_to_baseline", True)),
    }


def _passes_utility_floor(utility, floor):
    return (
        utility["accuracy"] >= floor["baseline"]["accuracy"] - floor["max_accuracy_drop"]
        and utility["f1"] >= floor["baseline"]["f1"] - floor["max_f1_drop"]
        and utility["positive_rate"] >= floor["min_positive_rate"]
    )


def choose_thresholds_equalized_odds(y_true, y_proba, A, grid, utility_config=None):
    if len(np.unique(A)) < 2:
        return {0: 0.5, 1: 0.5}

    best_gap = None
    best_utility = None
    best_t = {0: 0.5, 1: 0.5}
    floor = _threshold_utility_floor(y_true, y_proba, grid, utility_config)

    for t0 in grid:
        for t1 in grid:
            y_pred = np.where(A == 0, y_proba >= t0, y_proba >= t1).astype(int)
            utility = _classification_utility(y_true, y_pred)
            if not _passes_utility_floor(utility, floor):
                continue

            gap = equalized_odds_gap(y_true, y_pred, A)

            if (
                best_gap is None
                or gap < best_gap
                or (
                    np.isclose(gap, best_gap)
                    and (utility["f1"], utility["accuracy"]) > (best_utility["f1"], best_utility["accuracy"])
                )
            ):
                best_gap = gap
                best_utility = utility
                best_t = {0: float(t0), 1: float(t1)}

    if best_gap is None and floor["fallback_to_baseline"]:
        return {0: 0.5, 1: 0.5}
    return best_t


def choose_thresholds_demographic_parity(y_true, y_proba, A, grid, utility_config=None):
    """Find group-specific thresholds that equalize acceptance rates (Demographic Parity).
    
    Demographic Parity: P(Ŷ=1|A=0) = P(Ŷ=1|A=1)
    Minimizes DP gap subject to a utility floor to avoid trivial all-negative
    solutions that equalize acceptance by rejecting everyone.
    """
    if len(np.unique(A)) < 2:
        return {0: 0.5, 1: 0.5}

    best_dp_gap = None
    best_utility = None
    best_t = {0: 0.5, 1: 0.5}
    floor = _threshold_utility_floor(y_true, y_proba, grid, utility_config)

    for t0 in grid:
        for t1 in grid:
            y_pred = np.where(A == 0, y_proba >= t0, y_proba >= t1).astype(int)
            utility = _classification_utility(y_true, y_pred)
            if not _passes_utility_floor(utility, floor):
                continue
            
            # Compute DP gap: |acceptance_rate_A0 - acceptance_rate_A1|
            acc_rate_a0 = np.mean(y_pred[A == 0]) if np.any(A == 0) else 0.5
            acc_rate_a1 = np.mean(y_pred[A == 1]) if np.any(A == 1) else 0.5
            dp_gap = abs(acc_rate_a0 - acc_rate_a1)

            if (
                best_dp_gap is None
                or dp_gap < best_dp_gap
                or (
                    np.isclose(dp_gap, best_dp_gap)
                    and (utility["f1"], utility["accuracy"]) > (best_utility["f1"], best_utility["accuracy"])
                )
            ):
                best_dp_gap = dp_gap
                best_utility = utility
                best_t = {0: float(t0), 1: float(t1)}

    if best_dp_gap is None and floor["fallback_to_baseline"]:
        return {0: 0.5, 1: 0.5}
    return best_t


class FairlearnReductionsWrapper:
    """Expose a sklearn-like probability interface for Fairlearn reductions."""

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "_pmf_predict"):
            pmf = np.asarray(self.model._pmf_predict(X), dtype=float)
            if pmf.ndim == 1:
                p1 = pmf
            elif pmf.shape[1] >= 2:
                p1 = pmf[:, 1]
            else:
                p1 = pmf[:, 0]
        else:
            p1 = np.asarray(self.model.predict(X), dtype=float)

        p1 = np.clip(p1, 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])


class FairlearnThresholdOptimizerWrapper:
    """Wrap Fairlearn post-processing with the benchmark prediction interface."""

    def __init__(
        self,
        postprocessor,
        base_estimator,
        constraint: str,
        objective: str,
        grid_size: int,
        random_state: int,
    ):
        self.postprocessor = postprocessor
        self.base_estimator = base_estimator
        self.constraint = constraint
        self.objective = objective
        self.grid_size = int(grid_size)
        self.random_state = int(random_state)

    def predict_with_sensitive(self, X, A):
        return self.postprocessor.predict(
            X,
            sensitive_features=np.asarray(A).astype(int),
            random_state=self.random_state,
        )

    def predict_proba_with_sensitive(self, X, A):
        if hasattr(self.postprocessor, "_pmf_predict"):
            pmf = np.asarray(
                self.postprocessor._pmf_predict(X, sensitive_features=np.asarray(A).astype(int)),
                dtype=float,
            )
            if pmf.ndim == 1:
                p1 = pmf
            elif pmf.shape[1] >= 2:
                p1 = pmf[:, 1]
            else:
                p1 = pmf[:, 0]
            p1 = np.clip(p1, 0.0, 1.0)
            return np.column_stack([1.0 - p1, p1])
        return self.base_estimator.predict_proba(X)

    def predict(self, X):
        return self.base_estimator.predict(X)

    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)

    def recalibrate(self, X_cal, y_cal, A_cal):
        return _fit_threshold_optimizer(
            self.base_estimator,
            X_cal,
            y_cal,
            A_cal,
            constraint=self.constraint,
            objective=self.objective,
            grid_size=self.grid_size,
            random_state=self.random_state,
        )


def _fit_threshold_optimizer(
    base_estimator,
    X_cal,
    y_cal,
    A_cal,
    constraint: str,
    objective: str,
    grid_size: int,
    random_state: int,
):
    postprocessor = ThresholdOptimizer(
        estimator=base_estimator,
        constraints=constraint,
        objective=objective,
        grid_size=int(grid_size),
        prefit=True,
        predict_method="predict_proba",
    )
    postprocessor.fit(X_cal, y_cal, sensitive_features=np.asarray(A_cal).astype(int))
    return FairlearnThresholdOptimizerWrapper(
        postprocessor,
        base_estimator,
        constraint=constraint,
        objective=objective,
        grid_size=grid_size,
        random_state=random_state,
    )


def train_with_fairlearn_threshold_optimizer(
    X_train,
    y_train,
    X_val,
    y_val,
    A_val,
    seed: int,
    constraint: str,
    objective: str = "balanced_accuracy_score",
    grid_size: int = 100,
):
    """Train a base scorer and fit a standard Fairlearn threshold optimizer."""
    base_estimator = make_model(seed)
    base_estimator.fit(X_train, y_train)
    return _fit_threshold_optimizer(
        base_estimator,
        X_val,
        y_val,
        A_val,
        constraint=constraint,
        objective=objective,
        grid_size=int(grid_size),
        random_state=seed,
    )


def train_with_fairlearn_reductions(
    X_train,
    y_train,
    A_train,
    seed: int,
    constraint: str = "demographic_parity",
    eps: float = 0.02,
    max_iter: int = 50,
):
    """Train a standard Fairlearn reductions in-processing model."""
    constraint_key = constraint.lower().replace("-", "_")
    if constraint_key in {"demographic_parity", "dp"}:
        moment = DemographicParity()
    elif constraint_key in {"equalized_odds", "eo"}:
        moment = EqualizedOdds()
    else:
        raise ValueError(f"Unsupported Fairlearn constraint: {constraint}")

    model = ExponentiatedGradient(
        make_model(seed),
        moment,
        eps=float(eps),
        max_iter=int(max_iter),
        sample_weight_name="sample_weight",
    )
    model.fit(X_train, y_train, sensitive_features=np.asarray(A_train).astype(int))
    return FairlearnReductionsWrapper(model)


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
