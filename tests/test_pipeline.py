"""Stage 5 tests for the config-driven experiment runner."""

from __future__ import annotations

from pathlib import Path

from afd.pipeline import ExperimentPipeline
from afd.run import run_sweep


def _config(tmp_path: Path) -> dict:
    return {
        "experiment": {"name": "test", "output_dir": str(tmp_path)},
        "dataset": {
            "source": "synthetic",
            "n_samples": 600,
            "n_features": 6,
            "informative_features": 3,
            "fraud_rate": 0.05,
            "target_column": "isFraud",
            "drift": {
                "type": "sudden",
                "t0": 300,
                "width": 100,
                "interpolation": "linear",
                "seed": 3,
            },
        },
        "simulator": {
            "batch_size": 100,
            "warmup_batches": 2,
            "adaptation_enabled": True,
        },
        "model": {
            "type": "logistic",
            "class_weight": "none",
            "cost_fp": 1.0,
            "cost_fn": 1.0,
        },
        "detector": {
            "enabled": True,
            "delta": 0.1,
            "clock": 1,
            "input_signal": "score",
            "retrain_window": 100,
            "cooldown_batches": 1,
            "warmup_batches": 2,
        },
        "active_learning": {
            "enabled": True,
            "budget_fraction": 0.2,
            "total_budget": 100,
            "strategy": "rare_aware",
            "uncertainty_measure": "margin",
            "rare_aware": {
                "fraud_score_share": 0.5,
                "exploration_share": 0.5,
            },
        },
        "evaluation": {
            "pr_auc_window": 3,
            "f1_threshold": 0.5,
        },
        "seeds": [1],
    }


def test_pipeline_runs_and_returns_serializable_outputs(tmp_path: Path) -> None:
    result = ExperimentPipeline(_config(tmp_path)).run(seed=1)
    assert result["metrics"]["seed"] == 1
    assert result["metrics"]["labels_spent_total"] <= 100
    assert not result["batch_results"].empty
    assert "pr_auc_rolling" in result["batch_results"].columns


def test_sweep_creates_output_directory_and_files(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["experiment"]["sweep"] = {
        "drift_types": ["sudden"],
        "budgets": [0.0],
        "strategies": ["random"],
        "variants": ["baseline"],
    }
    output_dir = tmp_path / "nested" / "sweep"

    run_sweep(config, output_dir)

    assert (output_dir / "sweep_metrics.csv").exists()
    assert (output_dir / "sweep_batch_metrics.csv").exists()
