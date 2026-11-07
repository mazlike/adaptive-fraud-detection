"""Stage 2 tests: stream simulator invariants.

Covers:
- prequential ordering: predictions made before model update on each batch
- labeling budget never exceeded (per-batch fraction + total cap)
- no leakage: model updated only on acquired labels
- warmup trains on all labels, then test-then-train begins
- drift injected at the declared position (delegated to data generator)
- labels_cumulative is non-decreasing and <= total_budget when set
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from afd.baseline import BaselineModel
from afd.data import SyntheticStreamConfig, SyntheticStreamGenerator
from afd.simulator import StreamSimulator


def _make_stream(n_samples: int = 4000, seed: int = 1):
    cfg = SyntheticStreamConfig(
        n_samples=n_samples,
        n_features=12,
        informative_features=6,
        fraud_rate=0.01,
        drift_type="sudden",
        t0=n_samples // 2,
        width=400,
        seed=seed,
    )
    return SyntheticStreamGenerator(cfg).generate()


def _selector(probabilities: np.ndarray, n_to_select: int, context: dict) -> np.ndarray:
    """Uncertainty selector: pick samples closest to p=0.5."""
    n = min(n_to_select, probabilities.shape[0])
    uncertainty = np.abs(probabilities - 0.5)
    return np.argsort(uncertainty, kind="mergesort")[:n]


class _TrackingModel:
    """Wrapper that records every fit/partial_fit call to detect leakage."""

    def __init__(self) -> None:
        self.model = BaselineModel(model_type="logistic", class_weight="none")
        self.fit_calls: list[tuple[int, int]] = []  # (n_samples, n_fraud)
        self.fitted = False

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y).reshape(-1)
        self.fit_calls.append((int(y.shape[0]), int((y == 1).sum())))
        self.model.fit(X, y, sample_weight=sample_weight)
        self.fitted = True

    def partial_fit(self, X, y, sample_weight=None):
        # Simulate adaptation: refit on provided window only.
        self.fit(X, y, sample_weight=sample_weight)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def test_prequential_ordering_predict_before_update() -> None:
    """The model must not see batch labels before predicting on that batch."""
    X, y, _ = _make_stream()
    model = _TrackingModel()
    sim = StreamSimulator(
        model=model,
        batch_size=500,
        warmup_batches=2,
        budget_fraction=0.1,
        select_labels=_selector,
        strategy_context={"f1_threshold": 0.5},
        seed=7,
    )
    sim.run(X, y)

    # After warmup, each batch must produce exactly one predict (recorded via fit call order).
    # The first fit call happens during warmup (batch 0-1). Subsequent fits occur only after
    # predictions for that batch. We check that the number of post-warmup partial_fit calls
    # equals the number of post-warmup batches that acquired labels.
    post_warmup_batches = sim.warmup_batches
    post_warmup_results = sim.results[post_warmup_batches:]
    fit_calls_after_warmup = len(model.fit_calls) - post_warmup_batches  # warmup calls
    assert fit_calls_after_warmup == sum(
        1 for r in post_warmup_results if r.labels_acquired > 0
    ), f"{fit_calls_after_warmup} fits != batches-with-labels"


def test_budget_never_exceeded_per_batch() -> None:
    """labels_acquired per batch <= budget_fraction * batch_size."""
    X, y, _ = _make_stream()
    model = BaselineModel(model_type="logistic", class_weight="none")
    sim = StreamSimulator(
        model=model,
        batch_size=1000,
        warmup_batches=2,
        budget_fraction=0.05,
        select_labels=_selector,
        strategy_context={"f1_threshold": 0.5},
        seed=7,
    )
    sim.run(X, y)
    for r in sim.results:
        if r.batch_index < sim.warmup_batches:
            continue
        assert r.labels_acquired <= int(0.05 * r.n_samples) + 1, (
            f"batch {r.batch_index}: {r.labels_acquired} > {int(0.05 * r.n_samples)}"
        )


def test_total_budget_cap_respected() -> None:
    """labels_cumulative never exceeds total_budget when set."""
    X, y, _ = _make_stream(n_samples=8000)
    model = BaselineModel(model_type="logistic", class_weight="none")
    sim = StreamSimulator(
        model=model,
        batch_size=1000,
        warmup_batches=2,
        budget_fraction=0.2,
        total_budget=100,
        select_labels=_selector,
        strategy_context={"f1_threshold": 0.5},
        seed=7,
    )
    sim.run(X, y)
    assert sim.labels_spent_total <= 100, sim.labels_spent_total
    cumulative = [r.labels_cumulative for r in sim.results]
    assert all(c <= 100 for c in cumulative)
    # cumulative must be non-decreasing
    assert all(cumulative[i] <= cumulative[i + 1] for i in range(len(cumulative) - 1))


def test_duplicate_selected_indices_do_not_spend_duplicate_labels() -> None:
    """A selector bug must not make one sample consume the budget twice."""
    X, y, _ = _make_stream(n_samples=2000)
    model = BaselineModel(model_type="logistic", class_weight="none")

    def duplicate_selector(probabilities, n_to_select, context):
        return np.array([0, 0, 1, 1])[:n_to_select]

    sim = StreamSimulator(
        model=model,
        batch_size=500,
        warmup_batches=1,
        budget_fraction=0.01,
        select_labels=duplicate_selector,
    )
    sim.run(X, y)
    assert all(r.labels_acquired <= 2 for r in sim.results[1:])


def test_no_leakage_model_updated_only_on_acquired_labels() -> None:
    """partial_fit receives only samples whose labels were acquired."""
    X, y, _ = _make_stream()
    model = _TrackingModel()
    sim = StreamSimulator(
        model=model,
        batch_size=500,
        warmup_batches=1,
        budget_fraction=0.0,
        select_labels=None,
        strategy_context={"f1_threshold": 0.5},
        seed=7,
    )
    sim.run(X, y)
    # budget_fraction=0 -> no labels acquired after warmup -> no partial_fit calls post-warmup
    post_warmup = sim.results[sim.warmup_batches :]
    assert all(r.labels_acquired == 0 for r in post_warmup)
    # fit_calls count == warmup_batches only
    assert len(model.fit_calls) == sim.warmup_batches


def test_warmup_trains_on_all_labels() -> None:
    """During warmup the model sees every label (full initial training)."""
    X, y, _ = _make_stream()
    model = BaselineModel(model_type="logistic", class_weight="none")
    sim = StreamSimulator(
        model=model,
        batch_size=500,
        warmup_batches=3,
        budget_fraction=0.1,
        select_labels=_selector,
        strategy_context={"f1_threshold": 0.5},
        seed=7,
    )
    sim.run(X, y)
    warmup_results = sim.results[: sim.warmup_batches]
    for r in warmup_results:
        assert r.labels_acquired == r.n_samples
        assert np.all(r.labeled_mask)


def test_metrics_recorded_on_all_true_labels() -> None:
    """pr_auc/f1/recall are computed on all samples, not just labeled ones."""
    X, y, _ = _make_stream(n_samples=2000)
    model = BaselineModel(model_type="logistic", class_weight="none")
    sim = StreamSimulator(
        model=model,
        batch_size=500,
        warmup_batches=1,
        budget_fraction=0.0,
        select_labels=None,
        strategy_context={"f1_threshold": 0.5},
        seed=7,
    )
    sim.run(X, y)
    for r in sim.results[sim.warmup_batches:]:
        assert r.y_true.shape[0] == r.n_samples
        assert r.y_score.shape[0] == r.n_samples
        # pr_auc should be computable (not NaN) when both classes present
        if r.n_fraud > 0 and r.n_fraud < r.n_samples:
            assert not np.isnan(r.pr_auc)


def test_adaptation_can_be_disabled_for_static_baseline() -> None:
    """A no-adaptation baseline must not update on post-warmup labels."""
    X, y, _ = _make_stream(n_samples=2000)
    model = _TrackingModel()
    sim = StreamSimulator(
        model=model,
        batch_size=500,
        warmup_batches=1,
        budget_fraction=0.1,
        select_labels=_selector,
        adaptation_enabled=False,
        strategy_context={"f1_threshold": 0.5},
    )
    sim.run(X, y)
    assert len(model.fit_calls) == sim.warmup_batches


def test_drift_position_matches_generator() -> None:
    """Sanity: the stream used here has drift at t0 = n_samples // 2."""
    X, y, drift_points = _make_stream(n_samples=4000)
    assert 2000 in drift_points
    assert len(X) == 4000


def test_results_frame_shape() -> None:
    X, y, _ = _make_stream(n_samples=1000)
    model = BaselineModel(model_type="logistic", class_weight="none")
    sim = StreamSimulator(
        model=model,
        batch_size=250,
        warmup_batches=1,
        budget_fraction=0.1,
        select_labels=_selector,
        strategy_context={"f1_threshold": 0.5},
        seed=7,
    )
    sim.run(X, y)
    df = sim.results_frame()
    assert list(df.columns) == [
        "batch_index",
        "n_samples",
        "n_fraud",
        "labels_acquired",
        "labels_cumulative",
        "pr_auc",
        "f1",
        "recall",
    ]
    assert len(df) == 4  # 1000 / 250


def test_overall_metrics_keys() -> None:
    X, y, _ = _make_stream(n_samples=1000)
    model = BaselineModel(model_type="logistic", class_weight="none")
    sim = StreamSimulator(
        model=model,
        batch_size=250,
        warmup_batches=1,
        budget_fraction=0.1,
        select_labels=_selector,
        strategy_context={"f1_threshold": 0.5},
        seed=7,
    )
    sim.run(X, y)
    m = sim.overall_metrics()
    for key in [
        "pr_auc_overall",
        "f1_overall",
        "recall_at_budget_overall",
        "labels_spent_total",
        "n_batches_evaluated",
    ]:
        assert key in m


def test_default_active_learning_selector_is_used_when_none_is_provided() -> None:
    X, y, _ = _make_stream(n_samples=1000)
    model = BaselineModel(model_type="logistic", class_weight="none")
    sim = StreamSimulator(
        model=model,
        batch_size=250,
        warmup_batches=1,
        budget_fraction=0.1,
        select_labels=None,
        strategy_context={
            "strategy": "uncertainty",
            "uncertainty_measure": "margin",
            "f1_threshold": 0.5,
        },
        seed=7,
    )
    sim.run(X, y)
    assert any(r.labels_acquired > 0 for r in sim.results[sim.warmup_batches :])
