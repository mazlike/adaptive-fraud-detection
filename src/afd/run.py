"""Command line entry point.

Usage:
    python -m afd.run --config configs/experiment.yaml

Stage 0: loads and validates the YAML config and prints a summary.
Stages 1-6: this module assembles the pipeline from ``src/afd/pipeline.py``
and runs the full experiment.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from afd.pipeline import ExperimentPipeline


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {path!r} must contain a YAML mapping")
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    """Validate that required top-level sections are present."""
    required = ["experiment", "dataset", "simulator", "model", "detector", "active_learning", "evaluation", "seeds"]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Config missing required sections: {missing}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="afd.run",
        description="Adaptive fraud detection pipeline",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run the configured drift/budget/strategy/variant factorial sweep",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    validate_config(cfg)

    print(f"[afd.run] config loaded: {args.config}")
    print(f"[afd.run] experiment: {cfg['experiment']['name']}")
    print(f"[afd.run] dataset source: {cfg['dataset']['source']}")
    print(f"[afd.run] drift: {cfg['dataset']['drift']['type']}")
    print(f"[afd.run] AL strategy: {cfg['active_learning']['strategy']}")
    print(f"[afd.run] budget fraction: {cfg['active_learning']['budget_fraction']}")
    print(f"[afd.run] detector enabled: {cfg['detector']['enabled']}")
    print(f"[afd.run] seeds: {cfg['seeds']}")
    output_dir = Path(cfg["experiment"].get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False),
        encoding="utf-8",
    )

    if args.sweep or cfg["experiment"].get("sweep", {}).get("enabled", False):
        return run_sweep(cfg, output_dir)

    pipeline = ExperimentPipeline(cfg)
    metrics_rows: list[dict[str, Any]] = []
    batch_frames: list[pd.DataFrame] = []
    for seed in cfg["seeds"]:
        result = pipeline.run(int(seed))
        metrics = dict(result["metrics"])
        metrics["drift_points"] = json.dumps(metrics["drift_points"])
        metrics["alarm_batches"] = json.dumps(metrics["alarm_batches"])
        metrics["detection_delay_batches"] = json.dumps(metrics["detection_delay_batches"])
        metrics["retrain_batches"] = json.dumps(metrics["retrain_batches"])
        metrics["recovery_time_batches"] = json.dumps(metrics.get("recovery_time_batches", []))
        metrics_rows.append(metrics)
        batch = result["batch_results"].copy()
        batch.insert(0, "seed", int(seed))
        batch_frames.append(batch)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / "metrics_by_seed.csv", index=False)
    if batch_frames:
        pd.concat(batch_frames, ignore_index=True).to_csv(
            output_dir / "batch_metrics.csv",
            index=False,
        )
    (output_dir / "metrics_by_seed.json").write_text(
        json.dumps(metrics_rows, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"[afd.run] completed {len(metrics_rows)} seed(s)")
    print(f"[afd.run] results saved to: {output_dir}")
    return 0


def run_sweep(cfg: dict[str, Any], output_dir: Path) -> int:
    """Run a reproducible factorial experiment matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep = cfg["experiment"].get("sweep", {})
    drift_types = sweep.get("drift_types", ["sudden", "gradual"])
    budgets = sweep.get("budgets", [0.01, 0.05, 0.10])
    strategies = sweep.get("strategies", ["random", "uncertainty", "rare_aware"])
    variants = sweep.get("variants", ["baseline", "adwin", "adwin_al"])
    seeds = [int(seed) for seed in cfg.get("seeds", [])]
    pipeline_rows: list[dict[str, Any]] = []
    batch_frames: list[pd.DataFrame] = []

    for drift_type in drift_types:
        for budget in budgets:
            for strategy in strategies:
                for variant in variants:
                    for seed in seeds:
                        run_cfg = deepcopy(cfg)
                        run_cfg["dataset"]["drift"]["type"] = drift_type
                        run_cfg["active_learning"]["budget_fraction"] = float(budget)
                        run_cfg["active_learning"]["strategy"] = strategy
                        if variant == "baseline":
                            run_cfg["simulator"]["adaptation_enabled"] = False
                            run_cfg["detector"]["enabled"] = False
                            run_cfg["active_learning"]["enabled"] = False
                        elif variant == "adwin":
                            run_cfg["simulator"]["adaptation_enabled"] = True
                            run_cfg["detector"]["enabled"] = True
                            run_cfg["active_learning"]["enabled"] = False
                        elif variant == "adwin_al":
                            run_cfg["simulator"]["adaptation_enabled"] = True
                            run_cfg["detector"]["enabled"] = True
                            run_cfg["active_learning"]["enabled"] = True
                        else:
                            raise ValueError(f"Unknown sweep variant: {variant!r}")

                        result = ExperimentPipeline(run_cfg).run(seed)
                        metrics = dict(result["metrics"])
                        metrics.update(
                            {
                                "drift_type": drift_type,
                                "budget_fraction": float(budget),
                                "strategy": strategy,
                                "variant": variant,
                            }
                        )
                        for key in (
                            "drift_points",
                            "alarm_batches",
                            "detection_delay_batches",
                            "retrain_batches",
                            "recovery_time_batches",
                        ):
                            metrics[key] = json.dumps(metrics.get(key, []))
                        pipeline_rows.append(metrics)
                        batch = result["batch_results"].copy()
                        batch.insert(0, "seed", seed)
                        batch.insert(1, "drift_type", drift_type)
                        batch.insert(2, "budget_fraction", float(budget))
                        batch.insert(3, "strategy", strategy)
                        batch.insert(4, "variant", variant)
                        batch_frames.append(batch)

    pd.DataFrame(pipeline_rows).to_csv(output_dir / "sweep_metrics.csv", index=False)
    if batch_frames:
        pd.concat(batch_frames, ignore_index=True).to_csv(
            output_dir / "sweep_batch_metrics.csv",
            index=False,
        )
    (output_dir / "sweep_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False),
        encoding="utf-8",
    )
    print(f"[afd.run] completed sweep with {len(pipeline_rows)} runs")
    print(f"[afd.run] sweep results saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
