"""Drift detection with ADWIN (river) and drift-triggered retraining.

- ADWIN is fed a per-sample signal: model error (0/1) or model score.
- On alarm: retrain on the most recent `retrain_window` samples.
- Documented parameters: retrain window size, warm-up batches before arming,
  cooldown batches after an alarm (suppresses immediate re-alarms).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from river.drift import ADWIN


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
        self._adwin = ADWIN(delta=delta, clock=clock)
        self.alarms: list[Alarm] = []
        self.armed = False
        self._last_alarm_batch: int | None = None

    def arm(self) -> None:
        """Enable alarms after the detector warm-up period."""
        self.armed = True

    def update(
        self, value: float, batch_index: int, sample_index: int
    ) -> bool:
        """Feed one sample; return True if an alarm is raised."""
        if self.input_signal == "error" and value not in (0.0, 1.0):
            raise ValueError("error signal values must be exactly 0.0 or 1.0")
        self._adwin.update(float(value))
        if not self.armed or not self._adwin.drift_detected:
            return False
        if self._last_alarm_batch == batch_index:
            return False
        self._last_alarm_batch = batch_index
        self.alarms.append(
            Alarm(
                batch_index=batch_index,
                sample_index=sample_index,
                signal_value=float(value),
            )
        )
        return True


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
        self.retrain_batches: list[int] = []

    def should_retrain(self, batch_index: int) -> bool:
        """True if a retrain should happen now (alarm outside cooldown, past warm-up)."""
        if batch_index < self.warmup_batches:
            return False
        self.detector.arm()
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return False
        if not self.detector.alarms:
            return False
        latest = self.detector.alarms[-1]
        if latest.batch_index != batch_index:
            return False
        self.retrain_batches.append(batch_index)
        self._cooldown_counter = self.cooldown_batches
        return True
