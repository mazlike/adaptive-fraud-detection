"""Active learning loop with labeling budget and three selection strategies.

Strategies:
- random: uniform sampling among batch samples.
- uncertainty: least confidence / margin around p=0.5 / entropy of model predictions.
- rare_aware: share of budget reserved for top-p(fraud) samples (exploitation of
  the rare class) plus the remaining share for uncertainty sampling (exploration).
  Parameters `fraud_score_share` and `exploration_share` are configurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class LabelingDecision:
    """Which samples of a batch are sent for labeling."""

    batch_index: int
    labeled_indices: np.ndarray  # positions within the batch
    strategy: str
    labels_spent: int


class LabelingBudget:
    """Tracks labeling budget consumption (per-batch fraction + optional total cap)."""

    def __init__(
        self, budget_fraction: float, total_budget: int | None = None
    ) -> None:
        self.budget_fraction = budget_fraction
        self.total_budget = total_budget
        self.spent = 0

    def allowance(self, n_in_batch: int) -> int:
        """Max labels we may acquire this batch without exceeding the total cap.

        The per-batch allowance is ``budget_fraction * n_in_batch``, rounded down,
        further bounded by the remaining total budget (if set).
        """
        per_batch = int(self.budget_fraction * n_in_batch)
        if self.total_budget is not None:
            remaining = self.total_budget - self.spent
            per_batch = min(per_batch, max(remaining, 0))
        return max(per_batch, 0)

    def spend(self, n: int) -> None:
        self.spent += n


def select_samples(
    scores: np.ndarray,
    probabilities: np.ndarray,
    n_to_select: int,
    strategy: Literal["random", "uncertainty", "rare_aware"] = "uncertainty",
    fraud_score_share: float = 0.5,
    exploration_share: float = 0.5,
    uncertainty_measure: Literal[
        "margin", "least_confidence", "entropy"
    ] = "margin",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Select indices of samples to label within one batch.

    ``scores`` is treated as the model's fraud score. For this prototype it is
    expected to be P(fraud). ``probabilities`` is accepted for compatibility and
    defaults to ``scores`` when empty.
    """
    s = np.asarray(scores, dtype=float).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if p.size == 0:
        p = s
    if p.shape[0] != s.shape[0]:
        raise ValueError("scores and probabilities must have the same length")

    n = int(max(0, min(n_to_select, s.shape[0])))
    if n == 0:
        return np.array([], dtype=int)

    if rng is None:
        rng = np.random.default_rng(42)

    if strategy == "random":
        return np.sort(rng.choice(s.shape[0], size=n, replace=False)).astype(int)

    p_clipped = np.clip(p, 1e-12, 1.0 - 1e-12)
    if uncertainty_measure == "margin":
        uncertainty = -np.abs(p_clipped - 0.5)
    elif uncertainty_measure == "least_confidence":
        uncertainty = 1.0 - np.maximum(p_clipped, 1.0 - p_clipped)
    elif uncertainty_measure == "entropy":
        uncertainty = -(p_clipped * np.log(p_clipped) + (1.0 - p_clipped) * np.log(1.0 - p_clipped))
    else:
        raise ValueError(f"unknown uncertainty_measure: {uncertainty_measure!r}")

    if strategy == "uncertainty":
        order = np.argsort(-uncertainty, kind="mergesort")
        return order[:n].astype(int)

    if strategy == "rare_aware":
        w_fraud = max(float(fraud_score_share), 0.0)
        w_explore = max(float(exploration_share), 0.0)
        if w_fraud == 0.0 and w_explore == 0.0:
            w_fraud, w_explore = 0.5, 0.5
        total = w_fraud + w_explore
        n_fraud = int(np.floor(n * (w_fraud / total)))
        n_explore = n - n_fraud

        top_fraud = np.argsort(-p_clipped, kind="mergesort")[:n_fraud]
        selected = list(top_fraud.astype(int))
        selected_set = set(selected)

        if n_explore > 0:
            explore_pool = [i for i in np.argsort(-uncertainty, kind="mergesort") if int(i) not in selected_set]
            selected.extend([int(i) for i in explore_pool[:n_explore]])

        # Fill if duplicates/overlap reduced size.
        if len(selected) < n:
            ranked_pool = np.argsort(-p_clipped, kind="mergesort")
            for i in ranked_pool:
                ii = int(i)
                if ii not in selected_set:
                    selected.append(ii)
                    selected_set.add(ii)
                if len(selected) == n:
                    break

        return np.asarray(selected[:n], dtype=int)

    raise ValueError(f"unknown strategy: {strategy!r}")
