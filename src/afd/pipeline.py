"""Unified experiment pipeline assembled from the YAML config."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from afd.baseline import BaselineModel
from afd.data import SyntheticStreamConfig, SyntheticStreamGenerator, load_real_dataset
from afd.detector import ADWINDetector, DriftRetrainer
from afd.metrics import detection_delay, false_alarm_count, recovery_time
from afd.simulator import StreamSimulator


class ExperimentPipeline:
    """Assemble dataset, model, detector, AL loop, simulator and metrics."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def run(self, seed: int) -> dict[str, Any]:
        """Run one seed of the experiment; return metrics dict."""
        cfg = deepcopy(self.config)
        dataset_cfg = cfg["dataset"]
        X, y, drift_points = self._load_data(dataset_cfg, seed)

        model_cfg = dict(cfg["model"])
        model_type = model_cfg.pop("type", "lightgbm")
        model_cfg.pop("random_state", None)
        model = BaselineModel(
            model_type=model_type,
            random_state=seed,
            **model_cfg,
        )

        simulator_cfg = cfg["simulator"]
        al_cfg = cfg["active_learning"]
        strategy_context = {
            "strategy": al_cfg.get("strategy", "uncertainty"),
            "uncertainty_measure": al_cfg.get("uncertainty_measure", "margin"),
            "rare_aware": al_cfg.get("rare_aware", {}),
            "f1_threshold": cfg["evaluation"].get("f1_threshold", 0.5),
        }

        retrainer = None
        detector_cfg = cfg.get("detector", {})
        if detector_cfg.get("enabled", False):
            detector = ADWINDetector(
                delta=float(detector_cfg.get("delta", 0.002)),
                clock=int(detector_cfg.get("clock", 32)),
                max_window=detector_cfg.get("max_window"),
                input_signal=detector_cfg.get("input_signal", "error"),
            )
            retrainer = DriftRetrainer(
                detector=detector,
                retrain_window=int(detector_cfg.get("retrain_window", 5000)),
                cooldown_batches=int(detector_cfg.get("cooldown_batches", 5)),
                warmup_batches=int(detector_cfg.get("warmup_batches", simulator_cfg.get("warmup_batches", 0))),
            )

        budget_fraction = float(al_cfg.get("budget_fraction", 0.0)) if al_cfg.get("enabled", False) else 0.0
        simulator = StreamSimulator(
            model=model,
            batch_size=int(simulator_cfg["batch_size"]),
            warmup_batches=int(simulator_cfg["warmup_batches"]),
            budget_fraction=budget_fraction,
            total_budget=al_cfg.get("total_budget"),
            select_labels=None,
            strategy_context=strategy_context,
            seed=seed,
            adaptation_enabled=bool(simulator_cfg.get("adaptation_enabled", True)),
            drift_retrainer=retrainer,
        )
        simulator.run(X, y)

        frame = simulator.results_frame()
        # Batch-level rolling PR-AUC is computed from the already evaluated
        # batch PR-AUC values; warm-up rows remain NaN and are not included.
        rolling_window = int(cfg["evaluation"].get("pr_auc_window", 50))
        frame["pr_auc_rolling"] = np.nan
        if rolling_window > 0 and len(frame) > simulator.warmup_batches:
            batch_pr = frame.loc[simulator.warmup_batches:, "pr_auc"]
            frame.loc[simulator.warmup_batches:, "pr_auc_rolling"] = (
                batch_pr.rolling(rolling_window, min_periods=1).mean().to_numpy()
            )

        batch_drift_points = [
            point // int(simulator_cfg["batch_size"]) for point in drift_points
        ]
        alarm_batches = sorted({alarm.batch_index for alarm in simulator.drift_alarms})
        delays = detection_delay(alarm_batches, batch_drift_points)
        metrics = simulator.overall_metrics()
        recovery_values = [
            recovery_time(
                frame["pr_auc"].to_numpy(dtype=float),
                drift_batch=drift_batch,
                level=float(cfg["evaluation"].get("recovery_level", 0.95)),
            )
            for drift_batch in batch_drift_points
            if drift_batch < len(frame)
        ]
        metrics.update(
            {
                "seed": seed,
                "drift_points": drift_points,
                "alarm_batches": alarm_batches,
                "detection_delay_batches": delays,
                "false_alarm_count": false_alarm_count(
                    alarm_batches,
                    batch_drift_points,
                ),
                "recovery_time_batches": recovery_values,
                "retrain_batches": simulator.retrain_batches,
                "fraud_rate": float(np.mean(y)),
            }
        )
        return {
            "metrics": metrics,
            "batch_results": frame,
            "config": cfg,
        }

    @staticmethod
    def _load_data(
        dataset_cfg: dict[str, Any],
        seed: int,
    ) -> tuple[pd.DataFrame, pd.Series, list[int]]:
        source = dataset_cfg.get("source", "synthetic")
        if source == "synthetic":
            drift = dataset_cfg.get("drift", {})
            generator_cfg = SyntheticStreamConfig(
                n_samples=int(dataset_cfg.get("n_samples", 100_000)),
                n_features=int(dataset_cfg.get("n_features", 30)),
                informative_features=int(dataset_cfg.get("informative_features", 10)),
                fraud_rate=float(dataset_cfg.get("fraud_rate", 0.01)),
                drift_type=drift.get("type", "sudden"),
                t0=int(drift.get("t0", 50_000)),
                width=int(drift.get("width", 5_000)),
                interpolation=drift.get("interpolation", "linear"),
                seed=int(dataset_cfg.get("seed", 7)) + int(seed),
                target_column=dataset_cfg.get("target_column", "isFraud"),
            )
            return SyntheticStreamGenerator(generator_cfg).generate()

        X, y = load_real_dataset(
            path=dataset_cfg.get("path"),
            source=source,
            time_column=dataset_cfg.get("time_column", "TransactionDT"),
            target_column=dataset_cfg.get("target_column", "isFraud"),
        )
        numeric = X.select_dtypes(include=[np.number]).columns
        X = X.loc[:, numeric].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return X, y, []
