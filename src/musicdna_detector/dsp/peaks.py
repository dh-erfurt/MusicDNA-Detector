"""Onset peak picking.

Port of ``librosa.util.peak_pick`` and the parameter defaults ``librosa.onset.onset_detect``
supplies. The picker is inherently sequential -- accepting a peak skips the next ``wait``
frames -- so this stays a loop. It runs over the onset envelope, one value per frame, and
is nowhere near the cost of the spectral work that produces it.
"""

# Portions derived and adapted from librosa 0.11.0 (ISC).
# See THIRD_PARTY_NOTICES.txt for upstream copyright and license.
# MusicDNA-specific modifications are licensed under MIT.

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def peak_pick(
    values: FloatArray,
    *,
    pre_max: int,
    post_max: int,
    pre_avg: int,
    post_avg: int,
    delta: float,
    wait: int,
) -> IntArray:
    """Indices that are a local maximum and sufficiently above the local mean."""
    if pre_max < 0 or pre_avg < 0 or delta < 0 or wait < 0:
        raise ValueError("pre_max, pre_avg, delta and wait must be non-negative")
    if post_max <= 0 or post_avg <= 0:
        raise ValueError("post_max and post_avg must be positive")

    length = values.shape[0]
    peaks = np.zeros(length, dtype=bool)
    if length == 0:
        return np.flatnonzero(peaks).astype(np.int64)

    # librosa special-cases the first frame: it has no history to compare against.
    peaks[0] = values[0] >= np.max(values[: min(post_max, length)])
    peaks[0] &= values[0] >= np.mean(values[: min(post_avg, length)]) + delta

    index = wait + 1 if peaks[0] else 1
    while index < length:
        window_max = np.max(values[max(0, index - pre_max) : min(index + post_max, length)])
        if values[index] != window_max:
            index += 1
            continue
        window_mean = np.mean(values[max(0, index - pre_avg) : min(index + post_avg, length)])
        if values[index] < window_mean + delta:
            index += 1
            continue
        peaks[index] = True
        index += wait + 1

    return np.flatnonzero(peaks).astype(np.int64)


def onset_detect(onset_envelope: FloatArray, sr: float, hop_length: int) -> IntArray:
    """``librosa.onset.onset_detect`` for a precomputed envelope, in frame units.

    The thresholds are librosa's own defaults, expressed in milliseconds of context:
    30 ms of lookback for the local maximum, 100 ms for the local mean, 30 ms of
    refractory period after an accepted peak.
    """
    envelope = np.asarray(onset_envelope, dtype=np.float64)
    envelope = envelope - np.min(envelope, keepdims=True, axis=-1)
    envelope = envelope / (np.max(envelope, keepdims=True, axis=-1) + np.finfo(envelope.dtype).tiny)

    if not envelope.any() or not np.all(np.isfinite(envelope)):
        return np.array([], dtype=np.int64)

    return peak_pick(
        envelope,
        pre_max=_ceil_int(0.03 * sr // hop_length),
        post_max=_ceil_int(0.00 * sr // hop_length + 1),
        pre_avg=_ceil_int(0.10 * sr // hop_length),
        post_avg=_ceil_int(0.10 * sr // hop_length + 1),
        delta=0.07,
        wait=_ceil_int(0.03 * sr // hop_length),
    )


def _ceil_int(value: float) -> int:
    """``librosa.util.valid_int(..., cast=np.ceil)``."""
    return math.ceil(value)
