import numpy as np

def demographic_parity(y_pred, A):
    """
    P(y=1 | A=1) - P(y=1 | A=0)
    """
    p1 = y_pred[A == 1].mean()
    p0 = y_pred[A == 0].mean()
    return p1 - p0


def precision_score(y_true, y_pred):
    """Precision: TP / (TP + FP)"""
    if y_pred.sum() == 0:
        return 0.0
    return (y_true[y_pred == 1] == 1).sum() / y_pred.sum()


def recall_score(y_true, y_pred):
    """Recall: TP / (TP + FN)"""
    if y_true.sum() == 0:
        return 0.0
    return (y_pred[y_true == 1] == 1).sum() / y_true.sum()


def f1_score(y_true, y_pred):
    """F1: Harmonic mean of precision and recall"""
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def equalized_odds_gap(y_true, y_pred, A):
    """Equalized Odds gap: max(|TPR_1 - TPR_0|, |FPR_1 - FPR_0|)."""
    def _rates(mask):
        y_t = y_true[mask]
        y_p = y_pred[mask]
        tp = ((y_t == 1) & (y_p == 1)).sum()
        fp = ((y_t == 0) & (y_p == 1)).sum()
        fn = ((y_t == 1) & (y_p == 0)).sum()
        tn = ((y_t == 0) & (y_p == 0)).sum()

        tpr_den = tp + fn
        fpr_den = fp + tn
        tpr = tp / tpr_den if tpr_den > 0 else 0.0
        fpr = fp / fpr_den if fpr_den > 0 else 0.0
        return tpr, fpr

    mask0 = A == 0
    mask1 = A == 1
    tpr0, fpr0 = _rates(mask0)
    tpr1, fpr1 = _rates(mask1)

    return max(abs(tpr1 - tpr0), abs(fpr1 - fpr0))