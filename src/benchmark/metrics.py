"""Metric computation for benchmark evaluation."""

import numpy as np
from sklearn.metrics import roc_auc_score

from src.fairness import demographic_parity, equalized_odds_gap, observed_expected_gap, recall_score, f1_score


METRIC_REGISTRY = [
    ("dp_gap", "DP Gap"),
    ("eo_gap", "EO Gap"),
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Balanced Accuracy"),
    ("f1_score", "F1 Score"),
    ("sensitivity", "Sensitivity"),
    ("specificity", "Specificity"),
    ("fpr", "FPR"),
    ("fnr", "FNR"),
    ("base_rate", "Base Rate"),
    ("positive_rate", "Positive Rate"),
    ("auc", "AUC"),
    ("brier_score", "Brier Score"),
    ("oe_gap", "O/E Gap"),
    ("n", "N"),
    ("tp", "TP"),
    ("fp", "FP"),
    ("tn", "TN"),
    ("fn", "FN"),
]

METRIC_NAMES = [name for name, _ in METRIC_REGISTRY]
METRIC_LABELS = {name: label for name, label in METRIC_REGISTRY}


def compute_metrics(y_true, y_pred, y_proba, A):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_proba = np.asarray(y_proba)
    A = np.asarray(A).astype(int)

    dp_gap = demographic_parity(y_pred, A)
    eo_gap = equalized_odds_gap(y_true, y_pred, A)
    oe_gap = observed_expected_gap(y_true, y_proba, A)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    positives = tp + fn
    negatives = tn + fp

    acc = np.mean(y_pred == y_true)
    f1 = f1_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    specificity = tn / negatives if negatives > 0 else np.nan
    fpr = fp / negatives if negatives > 0 else np.nan
    fnr = fn / positives if positives > 0 else np.nan
    balanced_accuracy = np.nanmean([sensitivity, specificity])
    base_rate = np.mean(y_true)
    positive_rate = np.mean(y_pred)
    brier_score = np.mean((y_proba - y_true) ** 2)
    
    auc = np.nan
    try:
        auc = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        auc = np.nan

    metric_values = {
        "auc": auc,
        "brier_score": float(brier_score),
        "oe_gap": float(abs(oe_gap)) if not np.isnan(oe_gap) else np.nan,
        "accuracy": float(acc),
        "balanced_accuracy": float(balanced_accuracy),
        "f1_score": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "base_rate": float(base_rate),
        "positive_rate": float(positive_rate),
        "dp_gap": float(abs(dp_gap)),
        "eo_gap": float(eo_gap),
        "n": int(len(y_true)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
    return {name: metric_values[name] for name in METRIC_NAMES if name in metric_values}
