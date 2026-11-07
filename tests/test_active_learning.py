"""Stage 4 tests: active learning strategies and budget behavior."""

from __future__ import annotations

import numpy as np

from afd.active_learning import LabelingBudget, select_samples


def test_labeling_budget_respects_total_cap() -> None:
    budget = LabelingBudget(budget_fraction=0.2, total_budget=15)
    assert budget.allowance(100) == 15
    budget.spend(10)
    assert budget.allowance(100) == 5
    budget.spend(5)
    assert budget.allowance(100) == 0


def test_random_selection_is_unique_and_reproducible() -> None:
    p = np.linspace(0.01, 0.99, 100)
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    a = select_samples(
        scores=p,
        probabilities=p,
        n_to_select=10,
        strategy="random",
        rng=rng_a,
    )
    b = select_samples(
        scores=p,
        probabilities=p,
        n_to_select=10,
        strategy="random",
        rng=rng_b,
    )
    assert len(a) == 10
    assert len(np.unique(a)) == 10
    assert np.array_equal(a, b)


def test_uncertainty_margin_prefers_probabilities_near_half() -> None:
    p = np.array([0.01, 0.2, 0.49, 0.51, 0.9, 0.8])
    idx = select_samples(
        scores=p,
        probabilities=p,
        n_to_select=2,
        strategy="uncertainty",
        uncertainty_measure="margin",
    )
    assert set(idx.tolist()) == {2, 3}


def test_uncertainty_entropy_prefers_high_entropy_points() -> None:
    p = np.array([0.01, 0.1, 0.3, 0.5, 0.7, 0.95])
    idx = select_samples(
        scores=p,
        probabilities=p,
        n_to_select=2,
        strategy="uncertainty",
        uncertainty_measure="entropy",
    )
    assert set(idx.tolist()) == {2, 3}


def test_rare_aware_combines_exploitation_and_exploration() -> None:
    p = np.array([0.99, 0.97, 0.6, 0.5, 0.49, 0.05])
    idx = select_samples(
        scores=p,
        probabilities=p,
        n_to_select=4,
        strategy="rare_aware",
        fraud_score_share=0.5,
        exploration_share=0.5,
        uncertainty_measure="margin",
    )
    # Exploitation should include top fraud scores.
    assert 0 in idx and 1 in idx
    # Exploration should include near-boundary points.
    assert 3 in idx or 4 in idx


def test_rare_aware_all_exploitation_matches_top_scores() -> None:
    p = np.array([0.99, 0.2, 0.7, 0.6, 0.1])
    idx = select_samples(
        scores=p,
        probabilities=p,
        n_to_select=2,
        strategy="rare_aware",
        fraud_score_share=1.0,
        exploration_share=0.0,
    )
    assert set(idx.tolist()) == {0, 2}
