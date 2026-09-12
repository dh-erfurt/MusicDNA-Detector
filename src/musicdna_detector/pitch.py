"""pYIN pitch estimation behind the ``PitchEstimator`` protocol.

``PyinPitchEstimator`` runs on the in-house DSP core;
``PitchConfig(pyin_backend="librosa")`` switches to librosa's implementation for
comparison, with bit-identical results either way.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from . import dsp
from .models import AudioBuffer, FloatArray, PitchConfig, PitchTrack
from .protocols import PitchEstimator


def level_frame_length(
    window_seconds: float | None, sample_rate: int, fallback_samples: int
) -> int:
    """Resolve the RMS analysis window to samples at the backend's working rate.

    Expressing it in seconds keeps the level-track resolution independent of the
    input sample rate, so thresholds like ``silence_threshold_dbfs`` and
    ``energy_onset_rise_dbfs`` retain their meaning after resampling.
    """
    if window_seconds is None:
        return fallback_samples
    return max(1, round(sample_rate * window_seconds))


def _run_pyin(
    audio: AudioBuffer, config: PitchConfig
) -> tuple[FloatArray, NDArray[np.bool_], FloatArray]:
    """Run pYIN through the configured implementation.

    The in-house core is the runtime path; ``pyin_backend="librosa"`` exists so the two
    can be compared in the same process, which is what the equality tests do. librosa is
    imported here rather than at module scope because it is an optional reference
    dependency -- the native runtime path does not need to carry it.
    """
    samples = np.asarray(audio.samples, dtype=np.float64)
    if config.pyin_backend == "native":
        result = dsp.pyin_detailed(
            samples,
            sr=audio.sample_rate,
            fmin=config.fmin_hz,
            fmax=config.fmax_hz,
            frame_length=config.frame_length,
            hop_length=config.hop_length,
            beam=config.pyin_beam,
            resolution=config.pitch_resolution,
            grid_independent_voicing=config.grid_independent_voicing,
        )
        return result.f0, result.voiced_flag, result.voiced_prob
    try:
        import librosa
    except ImportError as error:
        raise ImportError(
            "pyin_backend='librosa' needs the optional reference dependency. Install "
            "musicdna-detector[librosa], or use "
            "the default pyin_backend='native', which produces identical output."
        ) from error
    frequencies, voiced, probabilities = librosa.pyin(
        samples,
        sr=audio.sample_rate,
        fmin=config.fmin_hz,
        fmax=config.fmax_hz,
        frame_length=config.frame_length,
        hop_length=config.hop_length,
        resolution=config.pitch_resolution,
    )
    return frequencies, voiced, probabilities


class PyinPitchEstimator:
    def __init__(self, config: PitchConfig | None = None) -> None:
        self.config = config or PitchConfig()

    def estimate(self, audio: AudioBuffer) -> PitchTrack:
        config = self.config
        if audio.samples.size == 0:
            empty = np.array([], dtype=np.float64)
            return PitchTrack(empty, empty, empty, empty)
        frequencies, _voiced, probabilities = _run_pyin(audio, config)
        frequencies = frequencies.astype(np.float64, copy=False)
        probabilities = np.nan_to_num(probabilities, nan=0.0).astype(np.float64, copy=False)
        times = dsp.frame_times(len(frequencies), audio.sample_rate, config.hop_length)
        rms = dsp.rms(
            np.asarray(audio.samples, dtype=np.float64),
            level_frame_length(config.level_window_seconds, audio.sample_rate, config.frame_length),
            config.hop_length,
        )
        with np.errstate(divide="ignore"):
            levels = 20.0 * np.log10(rms)
        if len(levels) != len(times):
            levels = np.resize(levels, len(times))
        return PitchTrack(times, frequencies, probabilities, levels)


def build_pitch_estimator(config: PitchConfig | None = None) -> PitchEstimator:
    """Return the supported pYIN F0 estimator."""
    return PyinPitchEstimator(config)
