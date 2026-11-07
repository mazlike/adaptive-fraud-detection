"""Stage 3 tests for ADWIN and drift-triggered retraining."""

from __future__ import annotations

import numpy as np
import pandas as pd

from afd.detector import ADWINDetector, Alarm, DriftRetrainer
from afd.simulator import StreamSimulator


def test_adwin_detects_an_abrupt_error_change() -> None:
    detector = ADWINDetector(delta=0.1, clock=1, input_signal="error")
    for sample_index in range(300):
        if sample_index == 150:
            detector.arm()
        value = 0.0 if sample_index < 150 else 1.0
        detector.update(value, sample_index // 10, sample_index)

    assert detector.alarms
    assert detector.alarms[0].sample_index >= 150


def test_detector_rejects_non_binary_error_signal() -> None:
    detector = ADWINDetector(input_signal="error")
    detector.arm()

    try:
        detector.update(0.5, batch_index=1, sample_index=1)
    except ValueError as exc:
        assert "0.0 or 1.0" in str(exc)
    else:
        raise AssertionError("non-binary error signal was accepted")


def test_retrainer_applies_warmup_and_cooldown() -> None:
    detector = ADWINDetector(delta=0.1, clock=1)
    retrainer = DriftRetrainer(
        detector=detector,
        cooldown_batches=2,
        warmup_batches=3,
    )

    detector.alarms.append(Alarm(batch_index=2, sample_index=20, signal_value=1.0))
    assert not retrainer.should_retrain(2)

    detector.alarms.append(Alarm(batch_index=3, sample_index=30, signal_value=1.0))
    assert retrainer.should_retrain(3)
    assert retrainer.retrain_batches == [3]

    detector.alarms.append(Alarm(batch_index=4, sample_index=40, signal_value=1.0))
    assert not retrainer.should_retrain(4)
    detector.alarms.append(Alarm(batch_index=5, sample_index=50, signal_value=1.0))
    assert not retrainer.should_retrain(5)
    detector.alarms.append(Alarm(batch_index=6, sample_index=60, signal_value=1.0))
    assert retrainer.should_retrain(6)


def test_retrainer_ignores_alarm_from_another_batch() -> None:
    detector = ADWINDetector()
    retrainer = DriftRetrainer(detector=detector, warmup_batches=0)
    detector.alarms.append(Alarm(batch_index=4, sample_index=40, signal_value=1.0))

    assert not retrainer.should_retrain(5)


def test_score_detector_observes_unlabeled_samples() -> None:
    """Score monitoring must not depend on the active-label budget."""

    class ScoreModel:
        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            values = X["score"].to_numpy(dtype=float)
            return values

    X = pd.DataFrame({"score": np.r_[np.zeros(150), np.ones(150)]})
    y = pd.Series(np.zeros(300, dtype=np.int64))
    detector = ADWINDetector(delta=0.1, clock=1, input_signal="score")
    retrainer = DriftRetrainer(detector, warmup_batches=0)
    simulator = StreamSimulator(
        model=ScoreModel(),
        batch_size=10,
        warmup_batches=0,
        budget_fraction=0.0,
        select_labels=None,
        drift_retrainer=retrainer,
    )

    simulator.run(X, y)

    assert simulator.drift_alarms
    assert simulator.labels_spent_total == 0
