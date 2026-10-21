import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_report(y_true, probabilities, threshold=0.5):
    """Calculate binary classification metrics from probabilities."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have equal lengths.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1.")

    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(
            y_true, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, predictions, zero_division=0
        )),
        "f1": float(f1_score(
            y_true, predictions, zero_division=0
        )),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def risk_bands(probabilities):
    """Assign High to the top 10%, Medium to the next 20%, and Low to the rest."""
    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim != 1 or len(probabilities) == 0:
        raise ValueError("Provide a non-empty one-dimensional score array.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Risk scores must be finite.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Risk scores must be between 0 and 1.")

    # Stable ranking gives reproducible bands even when scores tie.
    order = np.argsort(-probabilities, kind="stable")
    bands = np.full(len(probabilities), "Low", dtype=object)

    high_count = int(np.ceil(len(probabilities) * 0.10))
    medium_end = int(np.ceil(len(probabilities) * 0.30))

    bands[order[:high_count]] = "High"
    bands[order[high_count:medium_end]] = "Medium"

    return bands
