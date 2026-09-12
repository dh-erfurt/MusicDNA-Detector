"""YIN front end: framing, the cumulative mean normalized difference, interpolation.

Ported from ``librosa.core.pitch`` for the mono float64 case this detector actually
uses. The order of floating-point operations is copied deliberately, including where
it looks clumsy: a re-association like ``a + b + c`` to ``a + (b + c)`` changes the
last bits and would break the equality gate for no musical reason.
"""

# Portions derived and adapted from librosa 0.11.0 (ISC).
# See THIRD_PARTY_NOTICES.txt for upstream copyright and license.
# MusicDNA-specific modifications are licensed under MIT.

from __future__ import annotations

import numpy as np
import scipy.fft
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def frame_signal(y: FloatArray, frame_length: int, hop_length: int) -> FloatArray:
    """Frame a mono signal into ``(frame_length, n_frames)`` without copying."""
    if y.ndim != 1:
        raise ValueError("frame_signal expects a mono signal")
    if y.shape[0] < frame_length:
        raise ValueError(f"signal is shorter ({y.shape[0]}) than frame_length ({frame_length})")
    n_frames = 1 + (y.shape[0] - frame_length) // hop_length
    strides = (y.strides[0], hop_length * y.strides[0])
    return np.lib.stride_tricks.as_strided(y, shape=(frame_length, n_frames), strides=strides)


def _autocorrelate(y_frames: FloatArray, max_size: int) -> FloatArray:
    """Autocorrelation along the sample axis, matching ``librosa.autocorrelate``."""
    n_pad = scipy.fft.next_fast_len(2 * y_frames.shape[0] - 1, real=True)
    spectrum = scipy.fft.rfft(y_frames, n=n_pad, axis=0)
    # librosa's util.abs2 on complex input is real**2 + imag**2, not np.square.
    powspec = np.square(spectrum.real) + np.square(spectrum.imag)
    autocorr = np.asarray(scipy.fft.irfft(powspec, n=n_pad, axis=0), dtype=np.float64)
    return autocorr[: min(max_size, autocorr.shape[0])]


def cumulative_mean_normalized_difference(
    y_frames: FloatArray, min_period: int, max_period: int
) -> FloatArray:
    """Equation 8 of the YIN paper, in librosa's operation order."""
    acf_frames = _autocorrelate(y_frames, max_period + 1)

    yin_frames = np.square(y_frames)
    np.cumsum(yin_frames, out=yin_frames, axis=0)

    stop = max_period + 1
    yin_frames[0, :] = 0
    yin_frames[1:stop, :] = (
        2 * (acf_frames[0:1, :] - acf_frames[1:stop, :]) - yin_frames[: stop - 1, :]
    )

    yin_numerator = yin_frames[min_period : max_period + 1, :]
    k_range = np.arange(1, stop).reshape(-1, 1)
    cumulative_mean = np.cumsum(yin_frames[1:stop, :], axis=0) / k_range
    yin_denominator = cumulative_mean[min_period - 1 : max_period, :]
    tiny = np.finfo(yin_denominator.dtype).tiny
    return np.asarray(yin_numerator / (yin_denominator + tiny), dtype=np.float64)


def parabolic_interpolation(x: FloatArray) -> FloatArray:
    """Per-bin parabola optimum along axis 0; 0 where the optimum leaves ``[n-1, n+1]``.

    Vectorised form of librosa's numba stencil. Edge bins are 0, matching the stencil's
    untouched borders.
    """
    shifts = np.zeros_like(x)
    if x.shape[0] < 3:
        return shifts
    upper = x[2:]
    lower = x[:-2]
    centre = x[1:-1]
    a = upper + lower - 2 * centre
    b = (upper - lower) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        interior = -b / a
    interior[np.abs(b) >= np.abs(a)] = 0
    shifts[1:-1] = interior
    return shifts
