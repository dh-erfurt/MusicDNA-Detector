"""Tests for the supported pYIN pitch path."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from musicdna_detector import (
    AudioBuffer,
    PitchConfig,
    PyinPitchEstimator,
    build_pitch_estimator,
)


def test_pitch_config_validates_backend_fields() -> None:
    with pytest.raises(ValueError, match="estimator"):
        PitchConfig(estimator="unsupported")  # type: ignore[arg-type]


def test_factory_builds_the_supported_pyin_estimator() -> None:
    assert isinstance(build_pitch_estimator(), PyinPitchEstimator)
    assert isinstance(build_pitch_estimator(PitchConfig()), PyinPitchEstimator)


def test_librosa_backend_receives_the_configured_pitch_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import librosa

    received: dict[str, float] = {}

    def fake_pyin(
        samples: NDArray[np.float64], **kwargs: float
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]:
        received.update(kwargs)
        frame = np.array([220.0], dtype=np.float64)
        return frame, np.array([True]), np.array([0.9], dtype=np.float64)

    monkeypatch.setattr(librosa, "pyin", fake_pyin)
    config = PitchConfig(
        pyin_backend="librosa",
        pitch_resolution=0.25,
        frame_length=512,
        hop_length=128,
    )

    PyinPitchEstimator(config).estimate(AudioBuffer(np.ones(512), 8_000))

    assert received["resolution"] == 0.25


def test_librosa_backend_rejects_native_only_voicing_normalization() -> None:
    with pytest.raises(ValueError, match=r"grid_independent_voicing.*native"):
        PitchConfig(pyin_backend="librosa", grid_independent_voicing=True)


def test_level_window_sharpens_the_onset_the_level_track_reports() -> None:
    """A long RMS window leaks energy backwards past the onset it is meant to locate.

    Onset snapping and the HMM attack rewards read this track, so its time
    resolution bounds how precisely a note start can be placed. The default 2048
    samples span 92.9 ms at 22.05 kHz and report audibility well before the sound.
    """
    sample_rate = 22_050
    onset_seconds = 0.5
    samples = np.zeros(sample_rate, dtype=np.float64)
    tone = np.arange(sample_rate - int(sample_rate * onset_seconds), dtype=np.float64)
    samples[int(sample_rate * onset_seconds) :] = 0.5 * np.sin(2 * np.pi * 200 * tone / sample_rate)

    def first_audible_second(window_seconds: float | None) -> float:
        config = PitchConfig(fmin_hz=100.0, fmax_hz=400.0, level_window_seconds=window_seconds)
        track = PyinPitchEstimator(config).estimate(AudioBuffer(samples, sample_rate))
        audible = np.flatnonzero(np.asarray(track.levels_dbfs) >= -60.0)
        return float(track.times_seconds[audible[0]])

    wide = first_audible_second(None)
    narrow = first_audible_second(0.023)

    # The wide window opens well before the onset; the narrow one lands on it.
    assert wide < onset_seconds - 0.02
    assert abs(narrow - onset_seconds) < 0.02
    assert narrow > wide


def test_level_frame_length_is_backend_independent_in_seconds() -> None:
    from musicdna_detector.pitch import level_frame_length

    # The same duration resolves to the corresponding count at any sample rate.
    assert level_frame_length(0.02, 22_050, 2_048) == 441
    assert level_frame_length(0.02, 16_000, 640) == 320
    # None keeps the caller's established fallback.
    assert level_frame_length(None, 22_050, 2_048) == 2_048
    assert level_frame_length(None, 16_000, 640) == 640
