"""Classification evaluation helpers."""

from __future__ import annotations

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    """Return the main binary-classification metrics.

    For spam, recall on the spam class matters most (a missed spam is a
    nuisance) but precision matters too (a ham wrongly flagged is a lost
    message), so both are reported alongside their F1 balance.
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
