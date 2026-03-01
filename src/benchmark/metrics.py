"""Metric computation for benchmark evaluation."""

import numpy as np
from sklearn.metrics import roc_auc_score

from src.fairness import demographic_parity, equalized_odds_gap, observed_expected_gap, recall_score, f1_score


METRIC_REGISTRY = [
    ("dp_gap", "DP Gap"),
    ("eo_gap", "EO Gap"),
    ("accuracy", "Accuracy"),
    ("f1_score", "F1 Score"),
    ("sensitivity", "Sensitivity"),
    ("auc", "AUC"),
    ("brier_score", "Brier Score"),
    ("oe_gap", "O/E Gap")
]

METRIC_NAMES = [name for name, _ in METRIC_REGISTRY]
METRIC_LABELS = {name: label for name, label in METRIC_REGISTRY}


def compute_metrics(y_true, y_pred, y_proba, A):
    dp_gap = demographic_parity(y_pred, A)
    eo_gap = equalized_odds_gap(y_true, y_pred, A)
    oe_gap = observed_expected_gap(y_true, y_proba, A)
    acc = np.mean(y_pred == y_true)
    f1 = f1_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    brier_score = np.mean((y_proba - y_true) ** 2)
    
    auc = np.nan
    try:
        auc = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        auc = np.nan

    metric_values = {
        "auc": auc,
        "brier_score": float(brier_score),
        "oe_gap": float(oe_gap) if not np.isnan(oe_gap) else np.nan,
        "accuracy": float(acc),
        "f1_score": float(f1),
        "sensitivity": float(sensitivity),
        "dp_gap": float(abs(dp_gap)),
        "eo_gap": float(eo_gap),
    }
    return {name: metric_values[name] for name in METRIC_NAMES if name in metric_values}
