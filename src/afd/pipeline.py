"""Unified pipeline assembled from the YAML config.

Stage 0: stub. Implemented across Stages 1-5; wired here in Stage 5.
"""

from __future__ import annotations

from typing import Any


class ExperimentPipeline:
    """Assemble dataset, model, detector, AL loop, simulator and metrics."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def run(self, seed: int) -> dict[str, Any]:
        """Run one seed of the experiment; return metrics dict."""
        raise NotImplementedError("Stage 5: implement ExperimentPipeline.run")
