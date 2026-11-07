"""Data loading and synthetic stream generation.

Supports two sources:
1. Real IEEE-CIS Fraud Detection via ``kagglehub.competition_download('ieee-fraud-detection')``
   (or a local CSV path). Rows are sorted by the time axis (``TransactionDT``) and never shuffled.
2. A synthetic stream generator with injected concept drift. Ground-truth drift points are stored
   for evaluating the drift detector (detection delay, false alarm count).

All preprocessing decisions (scaling, encoding, thresholds) are deferred to the simulator/model
stage and are fit on past data only — this module only returns chronological (X, y).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

DriftType = Literal["none", "sudden", "gradual"]
Interpolation = Literal["linear", "sigmoid"]


@dataclass
class SyntheticStreamConfig:
    """Configuration for the synthetic stream generator.

    Attributes:
        n_samples: total number of samples in the stream.
        n_features: total number of features.
        informative_features: number of features carrying signal (<= n_features).
        fraud_rate: target share of positive (fraud) class, 0.001-0.02.
        drift_type: "none" | "sudden" | "gradual".
        t0: drift start position (sample index). Ground truth for detector eval.
        width: interpolation window for gradual drift (samples).
        interpolation: "linear" | "sigmoid" (used when drift_type == "gradual").
        seed: RNG seed.
        target_column: name of the returned target Series.
    """

    n_samples: int = 100_000
    n_features: int = 30
    informative_features: int = 10
    fraud_rate: float = 0.01
    drift_type: DriftType = "sudden"
    t0: int = 50_000
    width: int = 5_000
    interpolation: Interpolation = "linear"
    seed: int = 7
    target_column: str = "isFraud"


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class SyntheticStreamGenerator:
    """Generate a chronological stream with injected concept drift.

    Two class-conditional concepts are defined over the informative features:

    * Concept 0 (pre-drift): fraud probability rises with the first half of the
      informative features.
    * Concept 1 (post-drift): the relationship flips — the first half becomes
      negatively associated with fraud and the second half positively associated.

    A model trained on concept 0 therefore learns a decision boundary that is
    invalid after the drift, which is exactly what the adaptation experiments
    are designed to measure.

    ``drift_points`` (the ground truth) is populated with the drift start index
    ``t0`` and is used by the detector evaluation (detection delay, false alarms).

    The fraud rate is calibrated on concept 0 via a bias term (bisection on the
    sigmoid mean) so the realised share of fraud is close to ``fraud_rate``.
    """

    def __init__(self, cfg: SyntheticStreamConfig) -> None:
        if not 0 < cfg.fraud_rate < 1:
            raise ValueError("fraud_rate must be in (0, 1)")
        if cfg.informative_features > cfg.n_features:
            raise ValueError("informative_features cannot exceed n_features")
        if not 0 <= cfg.t0 < cfg.n_samples:
            raise ValueError("t0 must be within [0, n_samples)")
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.drift_points: list[int] = []

    def generate(self) -> tuple[pd.DataFrame, pd.Series, list[int]]:
        """Generate (X, y, drift_points) in chronological order.

        Returns:
            X: feature DataFrame (n_samples, n_features), columns f0..f{n-1}.
            y: target Series (n_samples,), 1 = fraud.
            drift_points: list of sample indices where drift starts.
        """
        cfg = self.cfg
        rng = self.rng
        n = cfg.n_samples

        # Features: i.i.d. standard normal, chronological by construction.
        X = rng.standard_normal((n, cfg.n_features))
        X_inf = X[:, : cfg.informative_features]

        # --- class-conditional concepts -------------------------------------
        half = max(1, cfg.informative_features // 2)
        w0 = np.zeros(cfg.informative_features)
        w0[:half] = 1.5  # concept 0: first half informative -> fraud

        w1 = np.zeros(cfg.informative_features)
        w1[half:] = 1.5  # concept 1: second half informative -> fraud
        w1[:half] = -1.0  # concept 1: first half now anti-predictive (boundary flips)

        noise_scale = 0.5
        s0 = X_inf @ w0 + rng.normal(0.0, noise_scale, n)
        s1 = X_inf @ w1 + rng.normal(0.0, noise_scale, n)

        # --- calibrate bias so realised fraud rate ~= target (concept 0) ----
        lo, hi = -30.0, 30.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if _sigmoid(s0 + mid).mean() < cfg.fraud_rate:
                lo = mid
            else:
                hi = mid
        bias = 0.5 * (lo + hi)

        # --- drift interpolation weight alpha(t) ----------------------------
        idx = np.arange(n)
        alpha = np.zeros(n)
        if cfg.drift_type == "sudden":
            alpha[idx >= cfg.t0] = 1.0
            self.drift_points = [int(cfg.t0)]
        elif cfg.drift_type == "gradual":
            end = min(cfg.t0 + cfg.width, n)
            if cfg.interpolation == "linear":
                ramp = np.clip((idx - cfg.t0) / cfg.width, 0.0, 1.0)
            else:  # sigmoid
                ramp = np.clip(
                    1.0 / (1.0 + np.exp(-6.0 * ((idx - cfg.t0) / max(cfg.width, 1)) - 3.0)),
                    0.0,
                    1.0,
                )
            alpha = np.where(idx < cfg.t0, 0.0, ramp)
            alpha[idx >= end] = 1.0
            self.drift_points = [int(cfg.t0)]
        else:
            self.drift_points = []

        # --- labels ---------------------------------------------------------
        s = (1.0 - alpha) * s0 + alpha * s1
        p = _sigmoid(s + bias)
        y = (rng.random(n) < p).astype(np.int64)

        columns = [f"f{i}" for i in range(cfg.n_features)]
        Xdf = pd.DataFrame(X, columns=columns)
        yser = pd.Series(y, name=cfg.target_column)
        return Xdf, yser, list(self.drift_points)


def _kagglehub_ieee_path() -> Path | None:
    """Return the path to train_transaction.csv via kagglehub, or None.

    Requires Kaggle API credentials (KAGGLE_USERNAME / KAGGLE_KEY or
    ~/.kaggle/kaggle.json). Returns None if kagglehub is unavailable or
    authentication fails.
    """
    try:
        import kagglehub  # type: ignore
    except ImportError:
        return None
    try:
        dl_dir = Path(kagglehub.competition_download("ieee-fraud-detection"))
    except Exception:
        return None
    csv_path = dl_dir / "train_transaction.csv"
    return csv_path if csv_path.exists() else None


def _drop_identifier_columns(df: pd.DataFrame) -> list[str]:
    """Return identifier-like columns that carry no predictive signal."""
    drop = []
    for c in df.columns:
        cl = c.lower()
        if cl in {"transactionid"} or cl.endswith("id"):
            drop.append(c)
    return drop


def load_real_dataset(
    path: str | Path | None = None,
    source: str = "ieee-cis",
    time_column: str = "TransactionDT",
    target_column: str = "isFraud",
) -> tuple[pd.DataFrame, pd.Series]:
    """Load a real chronological fraud dataset.

    For ``source="ieee-cis"`` this prefers
    ``kagglehub.competition_download('ieee-fraud-detection')`` and falls back to
    ``path`` if provided. For ``source="creditcard"``, ``path`` is required.
    The frame is sorted by ``time_column`` ascending and never shuffled.
    Identifier columns are dropped; all remaining columns are returned as ``X``
    (preprocessing is performed later, on past data only).

    Args:
        path: local CSV path (fallback when kagglehub is unavailable).
        source: dataset name: "ieee-cis" or "creditcard".
        time_column: chronological time axis column.
        target_column: target column name.

    Returns:
        (X, y) with X sorted by time and y the target.
    """
    if source not in {"ieee-cis", "creditcard"}:
        raise ValueError(f"Unsupported real dataset source: {source!r}")

    csv_path: Path | None = None
    if path is not None:
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")
    elif source == "ieee-cis":
        csv_path = _kagglehub_ieee_path()
        if csv_path is None:
            raise RuntimeError(
                "Could not locate IEEE-CIS data. Provide `dataset.path` to a local "
                "train_transaction.csv, or configure Kaggle credentials so "
                "kagglehub.competition_download('ieee-fraud-detection') works."
            )
    else:
        raise RuntimeError(
            "Credit Card Fraud data requires `dataset.path` to a local CSV file."
        )

    df = pd.read_csv(csv_path)

    if time_column not in df.columns:
        raise ValueError(f"time_column {time_column!r} not in dataset columns: {list(df.columns)}")
    if target_column not in df.columns:
        raise ValueError(f"target_column {target_column!r} not in dataset columns: {list(df.columns)}")

    # Chronological order only, no shuffling.
    df = df.sort_values(time_column, kind="mergesort").reset_index(drop=True)

    drop_cols = _drop_identifier_columns(df)
    X = df.drop(columns=[target_column] + drop_cols, errors="ignore")
    y = df[target_column].astype(np.int64)
    return X, y


def fraud_rate(y: pd.Series | np.ndarray) -> float:
    """Share of positive (fraud) samples."""
    y = np.asarray(y)
    return float(y.mean()) if y.size else 0.0
