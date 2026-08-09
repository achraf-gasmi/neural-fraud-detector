"""
FraudShield — Evaluation Metrics
==================================
AUPRC is the primary metric for fraud detection (not AUROC).
Reason: class imbalance makes AUROC optimistic; AUPRC is more diagnostic.
"""


import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def compute_auprc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Area Under Precision-Recall Curve."""
    return average_precision_score(y_true, y_scores)


def compute_auroc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Area Under ROC Curve."""
    try:
        return roc_auc_score(y_true, y_scores)
    except ValueError:
        return 0.0


def find_best_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric: str = "f1",
) -> tuple[float, float]:
    """
    Sweep thresholds on precision-recall curve to find optimal F1 threshold.

    Returns:
        (best_threshold, best_metric_value)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

    if metric == "f1":
        # F1 = 2 * P * R / (P + R)
        with np.errstate(divide="ignore", invalid="ignore"):
            f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores[:-1])  # last element has no threshold
        return float(thresholds[best_idx]), float(f1_scores[best_idx])

    elif metric == "precision_at_recall_80":
        # Find threshold that achieves ≥80% recall, maximize precision
        recall_mask = recalls >= 0.80
        if recall_mask.sum() == 0:
            return 0.5, 0.0
        best_idx = np.argmax(precisions[recall_mask])
        thresh_idx = np.where(recall_mask)[0][best_idx]
        return float(thresholds[min(thresh_idx, len(thresholds) - 1)]), float(precisions[thresh_idx])


def compute_all_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float | None = None,
) -> dict:
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: Binary ground truth labels
        y_scores: Predicted fraud probabilities
        threshold: Decision threshold (if None, finds optimal F1 threshold)

    Returns:
        Dict with all metrics
    """
    auprc = compute_auprc(y_true, y_scores)
    auroc = compute_auroc(y_true, y_scores)

    if threshold is None:
        threshold, _ = find_best_threshold(y_true, y_scores, metric="f1")

    y_pred = (y_scores >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    fpr = fp / (fp + tn + 1e-8)   # False positive rate

    return {
        "auprc": round(auprc, 4),
        "auroc": round(auroc, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "fpr": round(fpr, 4),
        "threshold": round(threshold, 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "total_fraud": int(y_true.sum()),
        "total_legit": int((y_true == 0).sum()),
    }


def print_metrics(metrics: dict, prefix: str = ""):
    """Pretty print metrics table."""
    tag = f"[{prefix}] " if prefix else ""
    print(f"\n{'='*50}")
    print(f"{tag}Evaluation Results")
    print(f"{'='*50}")
    print(f"  AUPRC  (primary):  {metrics['auprc']:.4f}")
    print(f"  AUROC:             {metrics['auroc']:.4f}")
    print(f"  F1 Score:          {metrics['f1']:.4f}")
    print(f"  Precision:         {metrics['precision']:.4f}")
    print(f"  Recall:            {metrics['recall']:.4f}")
    print(f"  FPR:               {metrics['fpr']:.4f}")
    print(f"  Threshold:         {metrics['threshold']:.4f}")
    print(f"  TP/FP/TN/FN:       {metrics['tp']}/{metrics['fp']}/{metrics['tn']}/{metrics['fn']}")
    print(f"  Fraud/Total:       {metrics['total_fraud']}/{metrics['total_fraud'] + metrics['total_legit']}")
    print(f"{'='*50}\n")


class MetricTracker:
    """Track metrics across epochs for early stopping and logging."""

    def __init__(self, primary_metric: str = "auprc", patience: int = 10):
        self.primary_metric = primary_metric
        self.patience = patience
        self.best_value = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.history = []

    def update(self, metrics: dict, epoch: int) -> bool:
        """
        Update tracker. Returns True if this is a new best.
        """
        value = metrics[self.primary_metric]
        self.history.append({"epoch": epoch, **metrics})

        if value > self.best_value:
            self.best_value = value
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            return True
        else:
            self.epochs_without_improvement += 1
            return False

    def should_stop(self) -> bool:
        return self.epochs_without_improvement >= self.patience

    def get_history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)
