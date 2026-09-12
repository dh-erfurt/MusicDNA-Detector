"""Pure frequency-to-note mapping utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

_PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


@dataclass(frozen=True, slots=True)
class NoteMapping:
    midi_pitch: int
    pitch_name: str
    cents: float


def round_half_up(value: float) -> int:
    """Round ties toward positive infinity (the conventional pitch mapping rule)."""
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    return math.floor(value + 0.5)


def frequency_to_midi_float(frequency_hz: float, tuning_reference_hz: float = 440.0) -> float:
    if not math.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("frequency_hz must be finite and positive")
    if not math.isfinite(tuning_reference_hz) or tuning_reference_hz <= 0:
        raise ValueError("tuning_reference_hz must be finite and positive")
    return 69.0 + 12.0 * math.log2(frequency_hz / tuning_reference_hz)


def midi_to_pitch_name(midi_pitch: int) -> str:
    return f"{_PITCH_CLASSES[midi_pitch % 12]}{midi_pitch // 12 - 1}"


def frequency_to_note(frequency_hz: float, tuning_reference_hz: float = 440.0) -> NoteMapping:
    midi_float = frequency_to_midi_float(frequency_hz, tuning_reference_hz)
    midi_pitch = round_half_up(midi_float)
    return NoteMapping(
        midi_pitch=midi_pitch,
        pitch_name=midi_to_pitch_name(midi_pitch),
        cents=100.0 * (midi_float - midi_pitch),
    )


def _circular_mean_cents(residuals: FloatArray) -> tuple[float, float]:
    """Return the circular mean of cent residuals and its concentration in [0, 1].

    Residuals live on a circle of period 100 cents: +49 and -49 are two cents apart,
    not 98. A linear statistic such as the median cannot see that, so it degrades as
    the true offset approaches the +-50-cent wrap. Mapping the residuals to unit vectors
    and averaging those makes the estimate exact right up to the wrap. The resultant
    length doubles as a confidence:
    it approaches 1 for a well-defined tuning and 0 for directionless input.
    """
    angles = 2.0 * math.pi * residuals / 100.0
    resultant = np.mean(np.exp(1j * angles))
    concentration = float(np.abs(resultant))
    offset = 100.0 * float(np.angle(resultant)) / (2.0 * math.pi)
    return offset, concentration


def estimate_tuning_reference_hz(
    frequencies_hz: ArrayLike,
    *,
    fallback_hz: float = 440.0,
    min_voiced_frames: int = 5,
    times_seconds: ArrayLike | None = None,
    voiced_probabilities: ArrayLike | None = None,
    min_confidence: float = 0.0,
    min_voiced_duration_seconds: float = 0.0,
    min_concentration: float = 0.05,
) -> float:
    """Estimate a clip-level tuning reference from voiced pitch frames.

    The estimator is intentionally conservative: it snaps each finite pitch frame to the
    nearest equal-tempered MIDI note under the fallback reference, collects the cents
    residuals, and takes their circular mean as a robust global offset. Optional frame
    times and voicing probabilities let callers require a minimum amount of confident
    voiced evidence in seconds. Residuals that point in no particular direction --
    unpitched or noise-dominated input -- yield a resultant below ``min_concentration``
    and fall back rather than shifting the whole MIDI grid arbitrarily. If the input is
    too sparse or invalid, the fallback reference is returned unchanged.

    Offsets beyond +-50 cents remain out of reach by construction: a clip sung a
    semitone flat is indistinguishable from one sung in tune, for any estimator.
    """
    if not math.isfinite(fallback_hz) or fallback_hz <= 0:
        raise ValueError("fallback_hz must be finite and positive")
    if min_voiced_frames < 1:
        raise ValueError("min_voiced_frames must be >= 1")
    if not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be in [0, 1]")
    if not math.isfinite(min_voiced_duration_seconds) or min_voiced_duration_seconds < 0:
        raise ValueError("min_voiced_duration_seconds must be finite and non-negative")
    if not math.isfinite(min_concentration) or not 0 <= min_concentration <= 1:
        raise ValueError("min_concentration must be in [0, 1]")

    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if frequencies.ndim != 1:
        raise ValueError("frequencies_hz must be one-dimensional")
    mask = np.isfinite(frequencies) & (frequencies > 0)

    if voiced_probabilities is not None:
        probabilities = np.asarray(voiced_probabilities, dtype=np.float64)
        if probabilities.ndim != 1 or probabilities.shape != frequencies.shape:
            raise ValueError("voiced_probabilities must match frequencies_hz")
        mask &= np.isfinite(probabilities) & (probabilities >= min_confidence)
    elif min_confidence > 0:
        raise ValueError("voiced_probabilities are required when min_confidence > 0")

    if min_voiced_duration_seconds > 0:
        if times_seconds is None:
            raise ValueError("times_seconds are required when min_voiced_duration_seconds > 0")
        times = np.asarray(times_seconds, dtype=np.float64)
        if times.ndim != 1 or times.shape != frequencies.shape:
            raise ValueError("times_seconds must match frequencies_hz")
        if np.any(~np.isfinite(times)) or np.any(times < 0) or np.any(np.diff(times) < 0):
            raise ValueError("times_seconds must be finite, non-negative, and ordered")
        positive_hops = np.diff(times)
        positive_hops = positive_hops[positive_hops > 0]
        if positive_hops.size == 0:
            return fallback_hz
        typical_hop = float(np.median(positive_hops))
        frame_support = np.minimum(
            np.diff(np.append(times, times[-1] + typical_hop)),
            1.5 * typical_hop,
        )
        voiced_duration = float(np.sum(frame_support[mask]))
        if not math.isfinite(voiced_duration) or voiced_duration < min_voiced_duration_seconds:
            return fallback_hz

    voiced = frequencies[mask]
    if voiced.size < min_voiced_frames:
        return fallback_hz

    cents_residuals: list[float] = []
    for frequency in voiced:
        midi_float = frequency_to_midi_float(float(frequency), fallback_hz)
        midi_pitch = round_half_up(midi_float)
        cents = 100.0 * (midi_float - midi_pitch)
        if math.isfinite(cents):
            cents_residuals.append(cents)

    if len(cents_residuals) < min_voiced_frames:
        return fallback_hz

    offset_cents, concentration = _circular_mean_cents(
        np.asarray(cents_residuals, dtype=np.float64)
    )
    if not math.isfinite(offset_cents) or concentration < min_concentration:
        return fallback_hz

    estimated = fallback_hz * math.pow(2.0, offset_cents / 1200.0)
    return estimated if math.isfinite(estimated) and estimated > 0 else fallback_hz
