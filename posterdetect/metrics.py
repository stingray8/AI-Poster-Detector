from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def aggregate_by_sample(sample_idx: np.ndarray, scores: np.ndarray, n_samples: int | None = None):
    """Mean of per-tile scores for each image/sample."""
    n = int(sample_idx.max()) + 1 if n_samples is None else n_samples
    sums = np.bincount(sample_idx, weights=scores, minlength=n)
    counts = np.bincount(sample_idx, minlength=n)
    return sums / np.maximum(counts, 1)


def tpr_at_fpr(y_true, scores, fpr_target: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.interp(fpr_target, fpr, tpr))


def summarize(y_true, scores, threshold: float = 0.0) -> dict:
    """`scores` are logits (threshold 0 == p=0.5)."""
    y_true = np.asarray(y_true)
    return {
        "n": int(len(y_true)),
        "auc": float(roc_auc_score(y_true, scores)),
        "ap": float(average_precision_score(y_true, scores)),
        "tpr@1%fpr": tpr_at_fpr(y_true, scores, 0.01),
        "tpr@5%fpr": tpr_at_fpr(y_true, scores, 0.05),
        "acc": float(((scores > threshold).astype(int) == y_true).mean()),
    }
