"""Metric computation for benchmark evaluation."""

import numpy as np
from sklearn.metrics import roc_auc_score

from src.fairness import demographic_parity, equalized_odds_gap


def compute_metrics(y_true, y_pred, y_proba, A):
    dp_gap = demographic_parity(y_pred, A)
    eo_gap = equalized_odds_gap(y_true, y_pred, A)
    acc = np.mean(y_pred == y_true)

    auc = np.nan
    try:
        auc = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        auc = np.nan

    return {
        "dp_gap": float(abs(dp_gap)),
        "eo_gap": float(eo_gap),
        "accuracy": float(acc),
        "auc": auc,
    }
