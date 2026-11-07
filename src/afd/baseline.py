"""Baseline model without adaptation plus imbalance handling.

Trains once on the initial window (no adaptation). Imbalance handling is
config-driven and applied only through sample/class weights derived from past
data — no future information is used.

Options:
- class_weight: "none" | "balanced" | "manual"
- LightGBM scale_pos_weight (positive-class weight)
- cost-sensitive sample weights from cost_fp / cost_fn
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - LightGBM optional at import time
    LGBMClassifier = None  # type: ignore[assignment]

from sklearn.linear_model import LogisticRegression


class BaselineModel:
    """Model trained once on the initial window; no adaptation.

    ``fit`` runs on past data only. ``partial_fit`` retrains on a provided
    window and is used by the drift-triggered retraining stage (Stage 3).
    """

    def __init__(
        self,
        model_type: str = "lightgbm",
        class_weight: Literal["none", "balanced", "manual"] = "balanced",
        scale_pos_weight: float | None = None,
        cost_fp: float = 1.0,
        cost_fn: float = 10.0,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = -1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        self.model_type = model_type
        self.class_weight = class_weight
        self.scale_pos_weight = scale_pos_weight
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.kwargs = kwargs
        self._model: object | None = None
        self._feature_names: list[str] | None = None

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    def _as_array(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if self._feature_names is None:
                self._feature_names = list(X.columns)
            return X.to_numpy(dtype=float)
        return np.asarray(X, dtype=float)

    def _sample_weights(
        self,
        y: np.ndarray,
        class_weight_mode: Literal["none", "balanced", "manual"],
    ) -> np.ndarray:
        """Build one explicit weighting scheme from past labels only."""
        y = np.asarray(y)
        w = np.ones(y.shape[0], dtype=float)
        if class_weight_mode == "balanced":
            n = y.shape[0]
            n_pos = int((y == 1).sum())
            n_neg = int((y == 0).sum())
            if n_pos > 0:
                w[y == 1] = n / (2.0 * n_pos)
            if n_neg > 0:
                w[y == 0] = n / (2.0 * n_neg)
        elif class_weight_mode == "manual":
            w[y == 1] = self.cost_fn
            w[y == 0] = self.cost_fp
        if self.scale_pos_weight is not None:
            w[y == 1] *= self.scale_pos_weight
        return w

    def _build_model(self) -> object:
        if self.model_type == "lightgbm":
            if LGBMClassifier is None:  # pragma: no cover
                raise RuntimeError("LightGBM is not installed")
            params = dict(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                n_jobs=1,
                verbosity=-1,
            )
            params.update(self.kwargs)
            return LGBMClassifier(**params)
        if self.model_type == "logistic":
            return LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                **self.kwargs,
            )
        raise ValueError(f"Unknown model_type: {self.model_type!r}")

    def _positive_probability(self, proba: np.ndarray) -> np.ndarray:
        """Normalize predict_proba output to P(fraud) column."""
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.reshape(-1)

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "BaselineModel":
        """Fit on past data only (no future information).

        Skips gracefully when ``y`` contains a single class (a classifier cannot
        be trained on one class); the previously fitted model is preserved.
        """
        X_arr = self._as_array(X)
        y_arr = np.asarray(y).astype(np.int64).reshape(-1)
        if len(np.unique(y_arr)) < 2:
            # Single-class window: cannot train; keep previous model if any.
            return self
        if sample_weight is None:
            sample_weight = self._sample_weights(y_arr, self.class_weight)
        self._model = self._build_model()
        self._model.fit(X_arr, y_arr, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return P(fraud) for each row of X.

        Returns zeros (predict all legitimate) if the model has not been fitted
        yet — e.g., when all warm-up data was single-class.
        """
        if self._model is None:
            n = self._as_array(X).shape[0]
            return np.zeros(n, dtype=float)
        X_arr = self._as_array(X)
        proba = self._model.predict_proba(X_arr)
        return self._positive_probability(proba)

    def predict(self, X: pd.DataFrame | np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Hard predictions at ``threshold`` on P(fraud)."""
        return (self.predict_proba(X) >= threshold).astype(np.int64)

    def partial_fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "BaselineModel":
        """Retrain on the provided window (used for drift-triggered retraining).

        For LightGBM this refits a fresh model on ``X, y``; there is no true
        incremental update, so the window must contain enough recent labeled
        samples to learn a new boundary.
        """
        return self.fit(X, y, sample_weight=sample_weight)
