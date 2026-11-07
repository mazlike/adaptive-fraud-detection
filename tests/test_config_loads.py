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


def test_entry_point_runs(tmp_path: Path) -> None:
    """The CLI completes on a small reproducible configuration."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "afd.run",
            "--config",
            str(_small_config(tmp_path)),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "completed" in result.stdout


def _small_config(tmp_path: Path) -> Path:
    with CONFIG.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["experiment"]["output_dir"] = str(tmp_path / "results")
    cfg["dataset"]["n_samples"] = 600
    cfg["dataset"]["n_features"] = 8
    cfg["dataset"]["informative_features"] = 4
    cfg["dataset"]["drift"]["t0"] = 300
    cfg["dataset"]["drift"]["width"] = 100
    cfg["simulator"]["batch_size"] = 100
    cfg["simulator"]["warmup_batches"] = 2
    cfg["model"]["type"] = "logistic"
    cfg["seeds"] = [42]
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path
