"""Streaming evaluation metrics.

Primary metric is PR-AUC (average precision) because fraud prevalence is 0.1-2%:
- accuracy is meaningless (predict-all-legitimate gives ~99% accuracy, fraud recall = 0);
- ROC-AUC is too optimistic under severe imbalance (enormous true-negative mass inflates it);
- PR-AUC focuses on the positive class and is sensitive to both precision and recall.

Rolling-window PR-AUC tracks quality degradation and recovery around drift points.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.metrics import average_precision_score, f1_score


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the Precision-Recall curve (average precision).

    Returns float('nan') if ``y_true`` contains a single class.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def rolling_pr_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    window: int,
    min_samples: int = 10,
) -> np.ndarray:
    """Rolling-window PR-AUC, one value per batch.

    For batch ``i`` the window covers batches ``max(0, i-window+1)..i``.
    Returns NaN where the window has fewer than ``min_samples`` or a single class.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = y_true.shape[0]
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        start = max(0, i - window + 1)
        yw = y_true[start : i + 1]
        sw = y_score[start : i + 1]
        if yw.size >= min_samples and len(np.unique(yw)) >= 2:
            out[i] = pr_auc(yw, sw)
    return out


def f1_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """F1 score at a fixed decision threshold on P(fraud)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.size == 0:
        return float("nan")
    y_pred = (y_score >= threshold).astype(np.int64)
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return float("nan")
    return float(f1_score(y_true, y_pred, zero_division=0))


def recall_at_budget(
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels_spent: int | np.ndarray,
) -> float:
    """Recall achievable under a fixed labeling budget.

    ``labels_spent`` is either an int (top-``labels_spent`` samples by score are
    labeled) or a boolean mask of labeled samples. Recall is the share of all
    fraud instances whose labels were acquired within the budget:

        recall = TP_labeled / total_fraud_in_batch.

    Returns 0.0 when there is no fraud in the batch.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    total_fraud = int((y_true == 1).sum())
    if total_fraud == 0:
        return 0.0
    if isinstance(labels_spent, int):
        n = min(max(labels_spent, 0), y_true.size)
        top = np.argsort(-y_score, kind="mergesort")[:n]
        labeled = np.zeros(y_true.size, dtype=bool)
        labeled[top] = True
    else:
        labeled = np.asarray(labels_spent, dtype=bool)
    tp_labeled = int(((y_true == 1) & labeled).sum())
    return tp_labeled / total_fraud


def recovery_time(
    pr_auc_series: list[float] | np.ndarray,
    drift_batch: int,
    level: float = 0.95,
) -> int | None:
    """Batches until PR-AUC returns to ``level`` of its pre-drift value.

    The pre-drift reference is the mean PR-AUC over batches before ``drift_batch``.
    Returns the index of the first batch at/after ``drift_batch`` where
    ``pr_auc >= level * pre_drift``, or None if never recovered.
    """
    series = np.asarray(pr_auc_series, dtype=float)
    pre = series[:drift_batch]
    pre = pre[~np.isnan(pre)]
    if pre.size == 0:
        return None
    reference = float(np.mean(pre))
    target = level * reference
    post = series[drift_batch:]
    for i, v in enumerate(post):
        if not np.isnan(v) and v >= target:
            return drift_batch + i
    return None


def detection_delay(detected_batches: list[int], ground_truth: list[int]) -> list[float | None]:
    """Detection delay per ground-truth drift point.

    For each ground-truth point, finds the first detected alarm at or after it
    and returns ``alarm_batch - drift_batch``. Returns None if no alarm follows.
    """
    delays: list[float | None] = []
    alarms = sorted(detected_batches)
    for gt in ground_truth:
        delay: float | None = None
        for a in alarms:
            if a >= gt:
                delay = float(a - gt)
                break
        delays.append(delay)
    return delays


def false_alarm_count(
    alarms: list[int],
    ground_truth: list[int],
    tolerance: int = 0,
) -> int:
    """Number of alarms not matching any ground-truth drift point.

    An alarm matches a drift point if ``|alarm - gt| <= tolerance``.
    """
    count = 0
    for a in alarms:
        if not any(abs(a - gt) <= tolerance for gt in ground_truth):
            count += 1
    return count


def labels_spent_series(
    decisions: list[dict],
    batch_indices: list[int],
) -> list[int]:
    """Cumulative labels spent per batch from labeling decisions."""
    spent_by_batch: dict[int, int] = {}
    for d in decisions:
        spent_by_batch[d["batch_index"]] = d.get("labels_spent", 0)
    cumulative = 0
    out: list[int] = []
    for bi in batch_indices:
        cumulative += spent_by_batch.get(bi, 0)
        out.append(cumulative)
    return out
