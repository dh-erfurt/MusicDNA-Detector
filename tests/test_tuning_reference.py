from __future__ import annotations

import numpy as np
import pytest

from musicdna_detector import AudioBuffer, PitchTrack
from musicdna_detector.mapping import estimate_tuning_reference_hz
from musicdna_detector.models import FloatArray


def _track(frequencies: list[float], *, dt: float = 0.02) -> PitchTrack:
    size = len(frequencies)
    return PitchTrack(
        np.arange(size, dtype=np.float64) * dt,
        np.asarray(frequencies, dtype=np.float64),
        np.full(size, 0.95, dtype=np.float64),
        np.zeros(size, dtype=np.float64),
    )


class _StubEstimator:
    def __init__(self, track: PitchTrack) -> None:
        self._track = track

    def estimate(self, audio: AudioBuffer) -> PitchTrack:
        assert audio.sample_rate == 8_000
        return self._track


def test_estimate_tuning_reference_hz_uses_robust_centers_and_fallback() -> None:
    sharp = estimate_tuning_reference_hz(np.array([442.0] * 5, dtype=np.float64))
    assert sharp == pytest.approx(442.0, rel=1e-6)

    sparse = estimate_tuning_reference_hz(np.array([442.0, 442.0, np.nan], dtype=np.float64))
    assert sparse == pytest.approx(440.0)

    in_tune = estimate_tuning_reference_hz(np.array([440.0] * 5, dtype=np.float64))
    assert in_tune == pytest.approx(440.0, rel=1e-6)


def _detuned_frames(
    offset_cents: float, *, count: int = 400, jitter_cents: float = 5.0
) -> FloatArray:
    """Frames spread over a realistic vocal range, all detuned by the same offset."""
    generator = np.random.default_rng(0)
    midi = generator.integers(55, 70, size=count).astype(np.float64)
    midi += offset_cents / 100.0 + generator.normal(0.0, jitter_cents / 100.0, size=count)
    return np.asarray(440.0 * 2.0 ** ((midi - 69.0) / 12.0), dtype=np.float64)


@pytest.mark.parametrize("offset_cents", [0.0, 25.0, 40.0, 45.0, 49.0, -45.0, -49.0])
def test_estimate_tuning_reference_stays_accurate_near_the_wrap(offset_cents: float) -> None:
    """Cent residuals are circular: a linear median biases as the offset nears +-50."""
    estimated = estimate_tuning_reference_hz(_detuned_frames(offset_cents))
    estimated_cents = 1200.0 * np.log2(estimated / 440.0)
    assert estimated_cents == pytest.approx(offset_cents, abs=1.0)


def test_estimate_tuning_reference_falls_back_on_directionless_residuals() -> None:
    """Uniform residuals carry no tuning; the MIDI grid must not be shifted at random."""
    generator = np.random.default_rng(1)
    midi = generator.uniform(55.0, 70.0, size=2_000)
    noise = np.asarray(440.0 * 2.0 ** ((midi - 69.0) / 12.0), dtype=np.float64)
    assert estimate_tuning_reference_hz(noise, fallback_hz=443.0) == 443.0


def test_estimate_tuning_reference_rejects_invalid_concentration() -> None:
    with pytest.raises(ValueError, match="min_concentration"):
        estimate_tuning_reference_hz(np.full(8, 442.0), min_concentration=1.5)


def test_estimate_tuning_reference_requires_confident_voiced_duration() -> None:
    frequencies = np.full(6, 442.0, dtype=np.float64)
    times = np.arange(6, dtype=np.float64) * 0.02

    tuned = estimate_tuning_reference_hz(
        frequencies,
        times_seconds=times,
        voiced_probabilities=np.full(6, 0.95, dtype=np.float64),
        min_confidence=0.8,
        min_voiced_duration_seconds=0.10,
    )
    too_short = estimate_tuning_reference_hz(
        frequencies,
        fallback_hz=438.0,
        times_seconds=times,
        voiced_probabilities=np.full(6, 0.95, dtype=np.float64),
        min_confidence=0.8,
        min_voiced_duration_seconds=0.14,
    )
    low_confidence = estimate_tuning_reference_hz(
        frequencies,
        fallback_hz=438.0,
        times_seconds=times,
        voiced_probabilities=np.full(6, 0.79, dtype=np.float64),
        min_confidence=0.8,
        min_voiced_duration_seconds=0.10,
    )

    assert tuned == pytest.approx(442.0, rel=1e-6)
    assert too_short == 438.0
    assert low_confidence == 438.0
