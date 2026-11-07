"""Stage 1 tests: data loading + synthetic drift generator.

Covers:
- synthetic generator output shapes / dtypes
- realised fraud rate close to target
- drift injected at the declared position t0 (drift_points)
- chronological order preserved (no shuffling)
- sudden vs gradual produce different post-drift distributions
- generation is deterministic given a seed
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from afd.data import SyntheticStreamConfig, SyntheticStreamGenerator, fraud_rate


def _make_cfg(**kw) -> SyntheticStreamConfig:
    base = dict(
        n_samples=2000,
        n_features=20,
        informative_features=8,
        fraud_rate=0.01,
        drift_type="sudden",
        t0=1000,
        width=400,
        seed=3,
    )
    base.update(kw)
    return SyntheticStreamConfig(**base)


def test_generator_shapes_and_dtypes() -> None:
    cfg = _make_cfg()
    X, y, drift_points = SyntheticStreamGenerator(cfg).generate()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape == (cfg.n_samples, cfg.n_features)
    assert y.shape == (cfg.n_samples,)
    assert set(X.columns) == {f"f{i}" for i in range(cfg.n_features)}
    assert set(np.unique(y)).issubset({0, 1})


def test_fraud_rate_close_to_target_pre_drift() -> None:
    # Bias is calibrated on concept 0, so the pre-drift realised rate should match target.
    cfg = _make_cfg(fraud_rate=0.01, n_samples=20_000, t0=10_000)
    _, y, dp = SyntheticStreamGenerator(cfg).generate()
    pre = y[y.index < dp[0]]
    assert abs(fraud_rate(pre) - cfg.fraud_rate) < 0.005, fraud_rate(pre)


def test_drift_injected_at_declared_position() -> None:
    cfg = _make_cfg(drift_type="sudden", t0=1000)
    _, _, drift_points = SyntheticStreamGenerator(cfg).generate()
    assert cfg.t0 in drift_points, f"drift point {cfg.t0} not in {drift_points}"


def test_gradual_drift_at_declared_position() -> None:
    cfg = _make_cfg(drift_type="gradual", t0=1000, width=400)
    _, _, drift_points = SyntheticStreamGenerator(cfg).generate()
    assert cfg.t0 in drift_points


def test_chronological_order_preserved() -> None:
    cfg = _make_cfg()
    X, _, _ = SyntheticStreamGenerator(cfg).generate()
    # Index is a plain RangeIndex in generation order; values increase left to right.
    assert list(X.index) == list(range(cfg.n_samples))


def test_sudden_vs_gradual_change_post_drift_distribution() -> None:
    cfg_sud = _make_cfg(drift_type="sudden", t0=1000)
    cfg_grad = _make_cfg(drift_type="gradual", t0=1000, width=400)
    Xs, ys, _ = SyntheticStreamGenerator(cfg_sud).generate()
    Xg, yg, _ = SyntheticStreamGenerator(cfg_grad).generate()

    pre_s = ys[ys.index < 1000].mean()
    # Sudden: fully post-drift right after t0.
    post_s = ys[ys.index >= 1000].mean()
    assert abs(post_s - pre_s) > 1e-4

    # Gradual: only fully concept-1 after the interpolation window ends.
    post_g = yg[yg.index >= 1000 + 400].mean()
    pre_g = yg[yg.index < 1000].mean()
    assert abs(post_g - pre_g) > 1e-4


def test_deterministic_given_seed() -> None:
    cfg = _make_cfg(seed=11)
    X1, y1, dp1 = SyntheticStreamGenerator(cfg).generate()
    X2, y2, dp2 = SyntheticStreamGenerator(cfg).generate()
    pd.testing.assert_frame_equal(X1, X2)
    pd.testing.assert_series_equal(y1, y2)
    assert dp1 == dp2


def test_no_shuffling_in_generator() -> None:
    """Generation order must equal row order (test-then-train relies on this)."""
    cfg = _make_cfg(n_samples=500, t0=250)
    X, _, _ = SyntheticStreamGenerator(cfg).generate()
    first_vals = X.iloc[0].to_numpy()
    last_vals = X.iloc[-1].to_numpy()
    assert not np.array_equal(first_vals, last_vals)
