"""Stage 0 smoke tests: config loads and entry point runs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

CONFIG = Path("configs/experiment.yaml")


def test_config_loads() -> None:
    """The experiment config parses and contains required sections."""
    with CONFIG.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    required = [
        "experiment", "dataset", "simulator", "model", "detector",
        "active_learning", "evaluation", "seeds",
    ]
    for key in required:
        assert key in cfg, f"missing config section: {key}"
    assert cfg["dataset"]["drift"]["type"] in {"none", "sudden", "gradual"}
    assert cfg["active_learning"]["strategy"] in {"random", "uncertainty", "rare_aware"}


def test_entry_point_runs() -> None:
    """`python -m afd.run --config configs/experiment.yaml` exits 0."""
    result = subprocess.run(
        [sys.executable, "-m", "afd.run", "--config", str(CONFIG)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "config is valid" in result.stdout
