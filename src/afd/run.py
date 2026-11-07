"""Command line entry point.

Usage:
    python -m afd.run --config configs/experiment.yaml

Stage 0: loads and validates the YAML config and prints a summary.
Stages 1-6: this module assembles the pipeline from ``src/afd/pipeline.py``
and runs the full experiment.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import yaml


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
    print("[afd.run] Stage 0 smoke test: config is valid. Pipeline not yet implemented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
