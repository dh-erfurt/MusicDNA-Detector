"""Spectral front end: STFT, mel filterbank, onset strength, RMS level, frame times.

These functions preserve the operation order of the librosa reference implementation,
and the parity tests compare their results with ``array_equal``.

Two details here are easy to get wrong and change the numbers. librosa computes the RMS
level in **float32** (``util.abs2`` defaults to it) and builds the mel weights in float32
as well, while everything around them is float64. Both are reproduced rather than
cleaned up: the level track feeds ``silence_threshold_dbfs`` and the onset snapping, so
"fixing" the precision would silently move calibrated thresholds.
"""

# Portions derived and adapted from librosa 0.11.0 (ISC).
# See THIRD_PARTY_NOTICES.txt for upstream copyright and license.
# MusicDNA-specific modifications are licensed under MIT.

from __future__ import annotations

import numpy as np
import scipy.fft
import scipy.signal
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

MEL_BANDS = 128
N_FFT = 2048
AMIN = 1e-10
TOP_DB = 80.0


def frame_times(n_frames: int, sr: float, hop_length: int) -> FloatArray:
    """``librosa.times_like``: the centre time of each frame, in seconds."""
    return np.asarray(np.arange(n_frames) * hop_length / sr, dtype=np.float64)


def _frame(y: FloatArray, frame_length: int, hop_length: int) -> FloatArray:
    n_frames = 1 + (y.shape[0] - frame_length) // hop_length
    strides = (y.strides[0], hop_length * y.strides[0])
    return np.lib.stride_tricks.as_strided(y, shape=(frame_length, n_frames), strides=strides)


def rms(y: FloatArray, frame_length: int, hop_length: int) -> NDArray[np.float32]:
    """``librosa.feature.rms``, including its float32 accumulation."""
    padded = np.pad(y, (frame_length // 2, frame_length // 2), mode="constant")
    frames = _frame(padded, frame_length, hop_length)
    power = np.mean(np.square(frames, dtype=np.float32), axis=-2, keepdims=True)
    levels: NDArray[np.float32] = np.sqrt(power, dtype=np.float32)[0]
    return levels


def stft(y: FloatArray, n_fft: int, hop_length: int) -> ComplexArray:
    """``librosa.stft`` for centred mono input with a periodic Hann window.

    librosa splits head, middle and tail to avoid copying the whole signal; the frames
    that come out are the same samples either way, so this pads once and frames once.
    """
    window = np.asarray(scipy.signal.get_window("hann", n_fft, fftbins=True), dtype=np.float64)
    padded = np.pad(y, (n_fft // 2, n_fft // 2), mode="constant")
    frames = _frame(padded, n_fft, hop_length)
    return np.asarray(scipy.fft.rfft(window[:, None] * frames, axis=0), dtype=np.complex128)


# Slaney mel scale constants: linear below 1 kHz, logarithmic above.
_F_SP = 200.0 / 3
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP
_LOG_STEP = np.log(6.4) / 27.0


def _hz_to_mel(frequency: float) -> float:
    if frequency >= _MIN_LOG_HZ:
        return float(_MIN_LOG_MEL + np.log(frequency / _MIN_LOG_HZ) / _LOG_STEP)
    return float(frequency / _F_SP)


def _mel_to_hz(mels: FloatArray) -> FloatArray:
    freqs = _F_SP * np.asarray(mels, dtype=np.float64)
    log_region = mels >= _MIN_LOG_MEL
    freqs[log_region] = _MIN_LOG_HZ * np.exp(_LOG_STEP * (mels[log_region] - _MIN_LOG_MEL))
    return freqs


def mel_filterbank(sr: float, n_fft: int, n_mels: int = MEL_BANDS) -> NDArray[np.float32]:
    """``librosa.filters.mel`` with the Slaney norm, ``fmin=0``, ``fmax=sr/2``."""
    fmax = float(sr) / 2
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=np.float32)
    fft_freqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)
    # The dtype is explicit because np.linspace's inferred one differs between
    # numpy versions, and _mel_to_hz writes into the array it is given.
    mel_edges = _mel_to_hz(
        np.linspace(_hz_to_mel(0.0), _hz_to_mel(fmax), n_mels + 2, dtype=np.float64)
    )

    fdiff = np.diff(mel_edges)
    ramps = np.subtract.outer(mel_edges, fft_freqs)

    for band in range(n_mels):
        lower = -ramps[band] / fdiff[band]
        upper = ramps[band + 2] / fdiff[band + 1]
        weights[band] = np.maximum(0, np.minimum(lower, upper))

    # Slaney normalisation: approximately constant energy per band.
    enorm = 2.0 / (mel_edges[2 : n_mels + 2] - mel_edges[:n_mels])
    weights *= enorm[:, np.newaxis]
    return weights


def melspectrogram(y: FloatArray, sr: float, n_fft: int, hop_length: int) -> FloatArray:
    power = np.abs(stft(y, n_fft, hop_length)) ** 2.0
    basis = mel_filterbank(sr, n_fft)
    return np.asarray(np.einsum("...ft,mf->...mt", power, basis, optimize=True))


def power_to_db(spectrogram: FloatArray) -> FloatArray:
    """``librosa.power_to_db`` with librosa's defaults (ref 1.0, amin 1e-10, 80 dB floor)."""
    log_spec = 10.0 * np.log10(np.maximum(AMIN, spectrogram))
    log_spec -= 10.0 * np.log10(np.maximum(AMIN, 1.0))
    return np.asarray(np.maximum(log_spec, log_spec.max() - TOP_DB), dtype=np.float64)


def onset_strength(y: FloatArray, sr: float, hop_length: int, n_fft: int = N_FFT) -> FloatArray:
    """``librosa.onset.onset_strength`` with librosa's defaults.

    Spectral flux over a mel spectrogram in dB: positive part of the frame-to-frame
    difference, averaged across bands, then shifted to undo the framing lag.
    """
    spectrogram = power_to_db(np.abs(melspectrogram(y, sr, n_fft, hop_length)))
    spectrogram = np.atleast_2d(spectrogram)

    flux = spectrogram[..., 1:] - spectrogram[..., :-1]
    flux = np.maximum(0.0, flux)
    envelope = np.mean(flux, axis=-2)

    # Counteract the framing shift: one frame for the difference, n_fft/(2*hop) for the
    # centred STFT.
    pad_width = 1 + n_fft // (2 * hop_length)
    envelope = np.pad(envelope, (pad_width, 0), mode="constant")
    return np.asarray(envelope[: spectrogram.shape[-1]], dtype=np.float64)


def normalized_onset_strength(
    y: FloatArray, sr: float, hop_length: int, n_fft: int = N_FFT
) -> FloatArray:
    """Return the finite peak-normalised onset envelope used by both consumers."""
    envelope = onset_strength(y, sr, hop_length, n_fft)
    if envelope.size == 0 or not np.any(np.isfinite(envelope)):
        return np.array([], dtype=np.float64)
    envelope = np.nan_to_num(envelope, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(envelope))
    if peak <= 0:
        return np.array([], dtype=np.float64)
    return np.asarray(envelope / peak, dtype=np.float64)
