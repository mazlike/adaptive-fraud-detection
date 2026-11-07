"""Active learning loop with labeling budget and three selection strategies.

Stage 0: stubs. Implemented in Stage 4.

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
    """Select indices of samples to label within one batch."""
    raise NotImplementedError("Stage 4: implement select_samples strategies")
