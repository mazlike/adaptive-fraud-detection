"""Prequential (test-then-train) stream simulator.

CRITICAL invariant: evaluation uses the TRUE labels of all samples in a batch,
while parameter updates use only labels actually acquired under the labeling
budget. No future information ever enters features, thresholds, or weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from afd.active_learning import LabelingBudget, select_samples
from afd.detector import DriftRetrainer
from afd.metrics import f1_at_threshold, pr_auc, recall_at_budget


@dataclass
class BatchResult:
    """Metrics recorded for one mini-batch."""

    batch_index: int
    n_samples: int
    n_fraud: int
    labels_acquired: int
    labels_cumulative: int
    y_true: np.ndarray = field(repr=False)
    y_score: np.ndarray = field(repr=False)
    labeled_mask: np.ndarray = field(repr=False)
    pr_auc: float = float("nan")
    f1: float = float("nan")
    recall: float = float("nan")


LabelSelector = Callable[
    [np.ndarray, int, dict],
    np.ndarray,
]


class StreamSimulator:
    """Process a stream in mini-batches: predict -> record metrics -> update.

    Parameters
    ----------
    model:
        Object with ``fit``/``predict_proba`` and optionally ``partial_fit``.
    batch_size:
        Samples per mini-batch.
    warmup_batches:
        Batches used for initial training (all labels available). They are
        recorded for traceability but excluded from evaluation metrics.
    budget_fraction:
        Max fraction of each batch sent for labeling.
    total_budget:
        Optional hard cap on total labels across the whole stream.
    select_labels:
        Callable ``(probabilities, n_to_select, context) -> indices`` selecting
        which samples to label within a batch.
    strategy_context:
        Extra config passed to the label selector (strategy name, shares, etc.).
    seed:
        RNG seed for tie-breaking / random strategy.
    adaptation_enabled:
        Whether acquired post-warm-up labels may update the model. Set to
        ``False`` for the static no-adaptation baseline.
    """

    def __init__(
        self,
        model: object,
        batch_size: int,
        warmup_batches: int,
        budget_fraction: float,
        total_budget: int | None = None,
        select_labels: LabelSelector | None = None,
        strategy_context: dict | None = None,
        seed: int = 42,
        adaptation_enabled: bool = True,
        drift_retrainer: DriftRetrainer | None = None,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.warmup_batches = warmup_batches
        self.budget_fraction = budget_fraction
        self.total_budget = total_budget
        self.select_labels = select_labels
        self.strategy_context = strategy_context or {}
        self.seed = seed
        self.adaptation_enabled = adaptation_enabled
        self.drift_retrainer = drift_retrainer
        self.drift_alarms = []
        self.retrain_batches: list[int] = []
        self.rng = np.random.default_rng(seed)
        self.labels_spent_total = 0
        self.warmup_labels_total = 0
        self.results: list[BatchResult] = []
        self._budget = None

    # ------------------------------------------------------------------ #
    # budget accounting
    # ------------------------------------------------------------------ #
    def _ensure_budget(self) -> LabelingBudget:
        if self._budget is None:
            self._budget = LabelingBudget(self.budget_fraction, self.total_budget)
        return self._budget

    def labels_allowed(self, n_in_batch: int) -> int:
        """Max labels we may acquire this batch without exceeding the total cap."""
        return self._ensure_budget().allowance(n_in_batch)

    def _default_select_labels(
        self,
        probabilities: np.ndarray,
        n_to_select: int,
        context: dict,
    ) -> np.ndarray:
        """Default active-learning selector driven by ``strategy_context``."""
        strategy = context.get("strategy", "uncertainty")
        rare = context.get("rare_aware", {})
        fraud_score_share = float(rare.get("fraud_score_share", context.get("fraud_score_share", 0.5)))
        exploration_share = float(rare.get("exploration_share", context.get("exploration_share", 0.5)))
        uncertainty_measure = context.get("uncertainty_measure", "margin")
        return select_samples(
            scores=probabilities,
            probabilities=probabilities,
            n_to_select=n_to_select,
            strategy=strategy,
            fraud_score_share=fraud_score_share,
            exploration_share=exploration_share,
            uncertainty_measure=uncertainty_measure,
            rng=self.rng,
        )

    # ------------------------------------------------------------------ #
    # main loop
    # ------------------------------------------------------------------ #
    def run(self, X: pd.DataFrame, y: pd.Series) -> list[BatchResult]:
        """Run prequential evaluation over the whole stream.

        For each batch after warmup:
          1. predict P(fraud) for all samples,
          2. record metrics on TRUE labels of all samples,
          3. select labels within budget,
          4. update model on labeled samples only.
        """
        X_arr = X
        y_arr = np.asarray(y).astype(np.int64).reshape(-1)
        n = y_arr.shape[0]
        n_batches = int(np.ceil(n / self.batch_size))

        self.results = []
        self.labels_spent_total = 0
        self.warmup_labels_total = 0
        self._budget = None
        self.drift_alarms = []
        self.retrain_batches = []

        warmup_X_parts: list[np.ndarray] = []
        warmup_y_parts: list[np.ndarray] = []
        labeled_X_parts: list[np.ndarray] = []
        labeled_y_parts: list[np.ndarray] = []

        for bi in range(n_batches):
            start = bi * self.batch_size
            end = min(start + self.batch_size, n)
            Xb = X_arr.iloc[start:end] if isinstance(X_arr, pd.DataFrame) else X_arr[start:end]
            yb = y_arr[start:end]

            # ---- 1. predict -------------------------------------------
            if bi < self.warmup_batches:
                # Warmup: train first (all labels), then evaluate on same batch.
                warmup_X_parts.append(Xb if isinstance(Xb, np.ndarray) else Xb.to_numpy(dtype=float))
                warmup_y_parts.append(yb)
                self.model.fit(
                    np.concatenate(warmup_X_parts),
                    np.concatenate(warmup_y_parts),
                )
                y_score = self.model.predict_proba(Xb)
                labeled_mask = np.ones(yb.shape[0], dtype=bool)
                labels_acquired = int(yb.shape[0])
                self.warmup_labels_total += labels_acquired
            else:
                # Test-then-train: predict BEFORE any update on this batch.
                y_score = self.model.predict_proba(Xb)

                # ---- 2. evaluate on TRUE labels of ALL samples -------
                labeled_mask = np.zeros(yb.shape[0], dtype=bool)
                labels_acquired = 0

                # ---- 3. select labels within budget -------------------
                n_allowed = self.labels_allowed(yb.shape[0])
                idx = np.array([], dtype=int)
                if n_allowed > 0:
                    ctx = dict(self.strategy_context)
                    selector = self.select_labels or self._default_select_labels
                    idx = selector(y_score, n_allowed, ctx)
                    idx = np.asarray(idx, dtype=int)
                    idx = idx[(idx >= 0) & (idx < yb.shape[0])]
                    idx = np.unique(idx)
                    labels_acquired = int(idx.size)
                    labeled_mask[idx] = True
                    self.labels_spent_total += labels_acquired
                    self._ensure_budget().spend(labels_acquired)

                # ---- 4. update detector ---------------------------------
                alarm = False
                if self.drift_retrainer is not None:
                    detector = self.drift_retrainer.detector
                    if bi >= self.drift_retrainer.warmup_batches:
                        detector.arm()
                    if detector.input_signal == "score":
                        for offset, score in enumerate(y_score):
                            alarm = detector.update(
                                float(score), bi, start + offset
                            ) or alarm
                    else:
                        for offset in idx:
                            error = float(int((y_score[offset] >= 0.5) != yb[offset]))
                            alarm = detector.update(
                                error, bi, start + int(offset)
                            ) or alarm
                    self.drift_alarms = list(detector.alarms)

                retrain = (
                    self.drift_retrainer is not None
                    and self.drift_retrainer.should_retrain(bi)
                )
                self.retrain_batches = (
                    list(self.drift_retrainer.retrain_batches)
                    if self.drift_retrainer is not None
                    else []
                )

                # ---- 5. update model on labeled samples only -------
                if labels_acquired > 0:
                    Xl = (
                        Xb.iloc[idx] if isinstance(Xb, pd.DataFrame) else Xb[idx]
                    )
                    yl = yb[idx]
                    Xl_arr = Xl.to_numpy(dtype=float) if isinstance(Xl, pd.DataFrame) else np.asarray(Xl)
                    labeled_X_parts.append(Xl_arr)
                    labeled_y_parts.append(yl)
                    if self.adaptation_enabled:
                        if retrain:
                            X_recent = np.concatenate(labeled_X_parts)
                            y_recent = np.concatenate(labeled_y_parts)
                            X_recent = X_recent[-self.drift_retrainer.retrain_window :]
                            y_recent = y_recent[-self.drift_retrainer.retrain_window :]
                            self.model.partial_fit(X_recent, y_recent)
                        elif hasattr(self.model, "partial_fit"):
                            self.model.partial_fit(Xl, yl)
                        else:
                            self.model.fit(Xl, yl)

            # ---- record metrics --------------------------------------
            if bi < self.warmup_batches:
                # Warm-up labels initialize the model but are never evaluation data.
                pr = float("nan")
                f1 = float("nan")
                rec = float("nan")
            else:
                pr = pr_auc(yb, y_score)
                f1 = f1_at_threshold(
                    yb,
                    y_score,
                    threshold=self.strategy_context.get("f1_threshold", 0.5),
                )
                rec = recall_at_budget(yb, y_score, labeled_mask)

            self.results.append(
                BatchResult(
                    batch_index=bi,
                    n_samples=int(yb.shape[0]),
                    n_fraud=int((yb == 1).sum()),
                    labels_acquired=labels_acquired,
                    labels_cumulative=int(self.labels_spent_total),
                    y_true=yb,
                    y_score=y_score,
                    labeled_mask=labeled_mask,
                    pr_auc=pr,
                    f1=f1,
                    recall=rec,
                )
            )

        return self.results

    # ------------------------------------------------------------------ #
    # convenience
    # ------------------------------------------------------------------ #
    def results_frame(self) -> pd.DataFrame:
        """Return recorded metrics as a DataFrame (one row per batch)."""
        rows = []
        for r in self.results:
            rows.append(
                {
                    "batch_index": r.batch_index,
                    "n_samples": r.n_samples,
                    "n_fraud": r.n_fraud,
                    "labels_acquired": r.labels_acquired,
                    "labels_cumulative": r.labels_cumulative,
                    "pr_auc": r.pr_auc,
                    "f1": r.f1,
                    "recall": r.recall,
                }
            )
        return pd.DataFrame(rows)

    def overall_metrics(self) -> dict[str, float]:
        """Aggregate metrics over all evaluated batches."""
        if not self.results:
            return {}
        evaluated = self.results[self.warmup_batches :]
        if not evaluated:
            return {}
        y_true = np.concatenate([r.y_true for r in evaluated])
        y_score = np.concatenate([r.y_score for r in evaluated])
        labeled = np.concatenate([r.labeled_mask for r in evaluated])
        return {
            "pr_auc_overall": pr_auc(y_true, y_score),
            "f1_overall": f1_at_threshold(y_true, y_score),
            "recall_at_budget_overall": recall_at_budget(y_true, y_score, labeled),
            "labels_spent_total": float(self.labels_spent_total),
            "n_batches_evaluated": float(len(evaluated)),
        }
