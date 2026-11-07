"""Drift detection with ADWIN (river) and drift-triggered retraining.

Stage 0: stubs. Implemented in Stage 3.

- ADWIN is fed a per-sample signal: model error (0/1) or model score.
- On alarm: retrain on the most recent `retrain_window` samples.
- Documented parameters: retrain window size, warm-up batches before arming,
  cooldown batches after an alarm (suppresses immediate re-alarms).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class Alarm:
    """A drift alarm raised by the detector."""

    batch_index: int
    sample_index: int
    signal_value: float


class ADWINDetector:
    """Thin wrapper around river.drift.ADWIN.

    Feeds per-sample error or score; raises an alarm when ADWIN detects a
    change in the signal mean.
    """

    def __init__(
        self,
        delta: float = 0.002,
        clock: int = 32,
        max_window: int | None = None,
        input_signal: Literal["error", "score"] = "error",
    ) -> None:
        self.delta = delta
        self.clock = clock
        self.max_window = max_window
        self.input_signal = input_signal
        self._adwin = None
        self.alarms: list[Alarm] = []
        self.armed = False

    def update(
        self, value: float, batch_index: int, sample_index: int
    ) -> bool:
        """Feed one sample; return True if an alarm is raised."""
        raise NotImplementedError("Stage 3: implement ADWINDetector.update")


class DriftRetrainer:
    """Orchestrates drift-triggered retraining with warm-up and cooldown."""

    def __init__(
        self,
        detector: ADWINDetector,
        retrain_window: int = 5_000,
        cooldown_batches: int = 5,
        warmup_batches: int = 10,
    ) -> None:
        self.detector = detector
        self.retrain_window = retrain_window
        self.cooldown_batches = cooldown_batches
        self.warmup_batches = warmup_batches
        self._cooldown_counter = 0

    def should_retrain(self, batch_index: int) -> bool:
        """True if a retrain should happen now (alarm outside cooldown, past warm-up)."""
        raise NotImplementedError("Stage 3: implement DriftRetrainer.should_retrain")
