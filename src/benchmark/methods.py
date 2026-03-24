"""Method-specific helpers for bias-mitigation benchmarks."""

import numpy as np
from scipy.optimize import linprog
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


# ---------------------------------------------------------------------------
# Method 5: Calibrated Equalized Odds (Hardt et al., NeurIPS 2016)
# ---------------------------------------------------------------------------

def _roc_points_for_group(y_true_g: np.ndarray, y_proba_g: np.ndarray, grid: list) -> dict:
    """Compute (tpr, fpr, error_rate) for each threshold in *grid* for one group.

    Returns a dict with keys ``thresholds``, ``tpr``, ``fpr``, ``error`` – each
    a 1-D numpy array of length ``len(grid)``.
    """
    thresholds = np.array(sorted(grid), dtype=float)
    n = len(y_true_g)
    tprs, fprs, errs = [], [], []

    pos_mask = y_true_g == 1
    neg_mask = ~pos_mask
    pos_total = int(pos_mask.sum())
    neg_total = int(neg_mask.sum())

    for t in thresholds:
        pred_pos = y_proba_g >= t
        tp = int((pos_mask & pred_pos).sum())
        fp = int((neg_mask & pred_pos).sum())
        fn = pos_total - tp
        tprs.append(tp / pos_total if pos_total > 0 else 0.0)
        fprs.append(fp / neg_total if neg_total > 0 else 0.0)
        errs.append((fp + fn) / n if n > 0 else 1.0)

    return {
        "thresholds": thresholds,
        "tpr": np.array(tprs),
        "fpr": np.array(fprs),
        "error": np.array(errs),
    }


def choose_calibrated_equalized_odds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    A: np.ndarray,
    grid: list,
    seed: int = 0,
) -> dict:
    """Find the globally optimal randomized post-processing strategy for equalized odds.

    Uses the LP formulation from Hardt et al. (NeurIPS 2016) "Equality of
    Opportunity in Supervised Learning".  For each group a convex combination of
    per-threshold predictors (mixing weights λ_g) is found such that:

      - Expected TPR is equal across groups.
      - Expected FPR is equal across groups.
      - Total expected error rate is minimised.

    The mixing weights are then used at prediction time: for a sample with score
    ``s`` in group ``g`` the model predicts 1 with probability
    ``sum_i λ_{g,i} * I(s ≥ t_i)``, and a seeded RNG converts that probability
    to a deterministic binary prediction.

    Returns a dict ``{0: {...}, 1: {...}, "seed": seed}`` suitable for
    ``_predict_with_method``.
    """
    if len(np.unique(A)) < 2:
        n = len(grid)
        uniform = np.zeros(n)
        mid = n // 2
        uniform[mid] = 1.0
        g = {"thresholds": np.array(sorted(grid), dtype=float), "weights": uniform}
        return {0: g, 1: g, "seed": seed}

    roc = {g: _roc_points_for_group(y_true[A == g], y_proba[A == g], grid) for g in [0, 1]}
    n0 = len(roc[0]["thresholds"])
    n1 = len(roc[1]["thresholds"])
    # Variables: [λ_{0,0}, …, λ_{0,n0-1}, λ_{1,0}, …, λ_{1,n1-1}]
    # Objective: minimise weighted sum of per-threshold error rates
    c = np.concatenate([roc[0]["error"], roc[1]["error"]])

    # Equality constraints (4 rows):
    #   Row 0: sum λ_0 = 1
    #   Row 1: sum λ_1 = 1
    #   Row 2: sum λ_0 * fpr0 - sum λ_1 * fpr1 = 0  (equal FPR)
    #   Row 3: sum λ_0 * tpr0 - sum λ_1 * tpr1 = 0  (equal TPR)
    A_eq = np.zeros((4, n0 + n1))
    A_eq[0, :n0] = 1.0
    A_eq[1, n0:] = 1.0
    A_eq[2, :n0] = roc[0]["fpr"]
    A_eq[2, n0:] = -roc[1]["fpr"]
    A_eq[3, :n0] = roc[0]["tpr"]
    A_eq[3, n0:] = -roc[1]["tpr"]
    b_eq = np.array([1.0, 1.0, 0.0, 0.0])

    bounds = [(0.0, None)] * (n0 + n1)

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if result.success:
        lam0 = np.maximum(result.x[:n0], 0.0)
        lam1 = np.maximum(result.x[n0:], 0.0)
        # Renormalise in case of tiny numerical errors; fall back to uniform
        # if the LP returned an all-zero vector (degenerate solution).
        lam0 = lam0 / lam0.sum() if lam0.sum() > 0 else np.ones(n0) / n0
        lam1 = lam1 / lam1.sum() if lam1.sum() > 0 else np.ones(n1) / n1
    else:
        # Fallback: equal-weight mixing (degenerate – same as grid equalized odds)
        lam0 = np.ones(n0) / n0
        lam1 = np.ones(n1) / n1

    return {
        0: {"thresholds": roc[0]["thresholds"], "weights": lam0},
        1: {"thresholds": roc[1]["thresholds"], "weights": lam1},
        "seed": seed,
    }


def apply_calibrated_equalized_odds(
    y_proba: np.ndarray,
    A: np.ndarray,
    mixing: dict,
) -> np.ndarray:
    """Convert calibrated-EO mixing weights into a binary prediction array.

    For sample ``i`` in group ``g``:
      p_i = sum_j λ_{g,j} * I(score_i ≥ threshold_j)

    A seeded RNG converts the probability ``p_i`` to a deterministic binary
    prediction.  Because the seed is fixed per run, repeated calls with the
    same arguments give identical results.
    """
    y_pred = np.zeros(len(y_proba), dtype=int)
    rng = np.random.default_rng(int(mixing.get("seed", 0)))
    random_draws = rng.random(len(y_proba))

    for g in [0, 1]:
        mask = A == g
        if not np.any(mask):
            continue
        info = mixing[g]
        t = info["thresholds"]  # shape (n,)
        w = info["weights"]     # shape (n,)
        # indicator: (n_samples_g, n_thresholds)
        indicator = y_proba[mask, np.newaxis] >= t[np.newaxis, :]
        prob1 = indicator.astype(float) @ w  # (n_samples_g,)
        prob1 = np.clip(prob1, 0.0, 1.0)
        y_pred[mask] = (random_draws[mask] < prob1).astype(int)

    return y_pred


# ---------------------------------------------------------------------------
# Method 6: Reject Option Classification (Kamiran et al., 2012)
# ---------------------------------------------------------------------------

def choose_reject_option_margin(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    A: np.ndarray,
    n_margin_steps: int = 20,
) -> dict:
    """Find the uncertainty-band margin that best reduces the DP gap (validation set).

    The reject option rule (Kamiran et al., ICDM 2012):
      - Determine which group is *disadvantaged* (lower positive prediction rate).
      - In the critical region ``[0.5 − θ, 0.5 + θ]``:
          * Disadvantaged group → predict 1.
          * Privileged group    → predict 0.
      - Outside the critical region: use standard threshold 0.5.

    The margin ``θ`` is tuned on the validation set by minimising ``|DP gap|``.

    Returns a dict ``{"margin": θ, "disadvantaged_group": g}``.
    """
    margins = np.linspace(0.0, 0.5, n_margin_steps + 1)[1:]  # skip θ=0 (no change)

    # Which group is disadvantaged?  The one with the lower baseline positive rate.
    rate0 = float((y_proba[A == 0] >= 0.5).mean()) if np.any(A == 0) else 0.5
    rate1 = float((y_proba[A == 1] >= 0.5).mean()) if np.any(A == 1) else 0.5
    disadvantaged = 1 if rate1 <= rate0 else 0

    best_margin = float(margins[0])
    best_gap = np.inf

    for margin in margins:
        y_pred = _apply_reject_option(y_proba, A, margin, disadvantaged)
        gap = abs(float(demographic_parity(y_pred, A)))
        if gap < best_gap:
            best_gap = gap
            best_margin = float(margin)

    return {"margin": best_margin, "disadvantaged_group": disadvantaged}


def _apply_reject_option(
    y_proba: np.ndarray,
    A: np.ndarray,
    margin: float,
    disadvantaged_group: int,
) -> np.ndarray:
    """Apply the reject option rule to a score array.

    Samples whose score falls in the critical region ``(0.5 − margin, 0.5 + margin]``
    receive a group-specific override; all other samples use the standard 0.5 threshold.
    """
    y_pred = (y_proba >= 0.5).astype(int)
    in_margin = (y_proba > 0.5 - margin) & (y_proba <= 0.5 + margin)
    disadvantaged = A == disadvantaged_group
    privileged = ~disadvantaged
    # Disadvantaged group in critical zone → predict positive
    y_pred[in_margin & disadvantaged] = 1
    # Privileged group in critical zone → predict negative
    y_pred[in_margin & privileged] = 0
    return y_pred
