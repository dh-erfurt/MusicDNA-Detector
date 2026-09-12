"""Parity tests for STFT, mel, onset strength, RMS, and peak picking.

librosa is the reference, every assertion is exact equality, and
that includes dtypes. Two of these functions deliberately compute in float32 because
librosa does, and an ``allclose`` comparison would let that difference through unnoticed.
"""

from __future__ import annotations

import librosa
import librosa.filters as librosa_filters
import numpy as np
import pytest
from numpy.typing import NDArray

from musicdna_detector.dsp import peaks, spectral

SAMPLE_RATE = 22_050
HOP_LENGTH = 256
N_FFT = 2_048
FRAME_LENGTH = 2_048


def _signal(kind: str, seconds: float = 1.5) -> NDArray[np.float64]:
    rng = np.random.default_rng(20260727)
    n = int(SAMPLE_RATE * seconds)
    times = np.arange(n) / SAMPLE_RATE
    if kind == "tone":
        return np.sin(2 * np.pi * 220.0 * times)
    if kind == "staccato":
        # Repeated attacks: the case onset strength and peak picking exist for.
        envelope = (np.sin(2 * np.pi * 4.0 * times) > 0.4).astype(np.float64)
        return np.sin(2 * np.pi * 330.0 * times) * envelope
    if kind == "noise":
        return rng.standard_normal(n) * 0.05
    if kind == "silence":
        return np.zeros(n)
    if kind == "mixed":
        parts = [
            np.sin(2 * np.pi * 196.0 * times[: n // 3]),
            np.zeros(n // 3),
            np.sin(2 * np.pi * 392.0 * times[: n - 2 * (n // 3)]) * 0.4,
        ]
        return np.concatenate(parts) + rng.standard_normal(n) * 0.002
    raise ValueError(kind)


ALL_SIGNALS = ["tone", "staccato", "noise", "silence", "mixed"]


@pytest.mark.parametrize("kind", ALL_SIGNALS)
@pytest.mark.parametrize("frame_length", [2_048, 512])
def test_rms_matches_librosa(kind: str, frame_length: int) -> None:
    signal = _signal(kind)
    expected = librosa.feature.rms(y=signal, frame_length=frame_length, hop_length=HOP_LENGTH)[0]
    actual = spectral.rms(signal, frame_length, HOP_LENGTH)
    assert expected.dtype == actual.dtype, "librosa accumulates the level in float32"
    assert np.array_equal(expected, actual)


@pytest.mark.parametrize("kind", ALL_SIGNALS)
def test_stft_matches_librosa(kind: str) -> None:
    signal = _signal(kind)
    expected = librosa.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH)
    assert np.array_equal(expected, spectral.stft(signal, N_FFT, HOP_LENGTH))


@pytest.mark.parametrize("sr", [8_000, 16_000, 22_050, 44_100])
def test_mel_filterbank_matches_librosa(sr: int) -> None:
    expected = librosa_filters.mel(sr=sr, n_fft=N_FFT, fmax=0.5 * sr)
    actual = spectral.mel_filterbank(sr, N_FFT)
    assert expected.dtype == actual.dtype
    assert np.array_equal(expected, actual)


@pytest.mark.parametrize("kind", ALL_SIGNALS)
def test_melspectrogram_matches_librosa(kind: str) -> None:
    signal = _signal(kind)
    expected = librosa.feature.melspectrogram(
        y=signal, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=0.5 * SAMPLE_RATE
    )
    actual = spectral.melspectrogram(signal, SAMPLE_RATE, N_FFT, HOP_LENGTH)
    assert np.array_equal(expected, actual)


@pytest.mark.parametrize("kind", ALL_SIGNALS)
def test_power_to_db_matches_librosa(kind: str) -> None:
    spectrogram = spectral.melspectrogram(_signal(kind), SAMPLE_RATE, N_FFT, HOP_LENGTH)
    expected = librosa.power_to_db(np.abs(spectrogram))
    assert np.array_equal(expected, spectral.power_to_db(np.abs(spectrogram)))


@pytest.mark.parametrize("kind", ALL_SIGNALS)
@pytest.mark.parametrize("hop_length", [256, 512])
def test_onset_strength_matches_librosa(kind: str, hop_length: int) -> None:
    signal = _signal(kind)
    expected = librosa.onset.onset_strength(y=signal, sr=SAMPLE_RATE, hop_length=hop_length)
    actual = spectral.onset_strength(signal, SAMPLE_RATE, hop_length)
    assert np.array_equal(expected, actual)


def test_normalized_onset_strength_preserves_the_existing_normalization() -> None:
    raw = spectral.onset_strength(_signal("mixed"), SAMPLE_RATE, HOP_LENGTH)
    finite = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    expected = finite / np.max(finite)

    actual = spectral.normalized_onset_strength(_signal("mixed"), SAMPLE_RATE, HOP_LENGTH)

    assert np.array_equal(expected, actual)


@pytest.mark.parametrize("kind", ALL_SIGNALS)
def test_onset_detect_matches_librosa(kind: str) -> None:
    envelope = spectral.onset_strength(_signal(kind), SAMPLE_RATE, HOP_LENGTH)
    peak = float(np.max(envelope))
    if peak <= 0:
        pytest.skip("no onset energy in this signal")
    normalized = envelope / peak
    expected = librosa.onset.onset_detect(
        onset_envelope=normalized,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        units="frames",
        backtrack=False,
    )
    actual = peaks.onset_detect(normalized, SAMPLE_RATE, HOP_LENGTH)
    assert np.array_equal(expected.astype(np.int64), actual)


def test_onset_detect_on_a_flat_envelope_returns_nothing() -> None:
    assert peaks.onset_detect(np.zeros(64), SAMPLE_RATE, HOP_LENGTH).size == 0


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_peak_pick_matches_librosa(seed: int) -> None:
    rng = np.random.default_rng(seed)
    values = np.abs(rng.standard_normal(600))
    kwargs = dict(pre_max=3, post_max=1, pre_avg=9, post_avg=10, delta=0.5, wait=3)
    expected = librosa.util.peak_pick(values, **kwargs)  # type: ignore[arg-type]
    assert np.array_equal(expected.astype(np.int64), peaks.peak_pick(values, **kwargs))


def test_peak_pick_rejects_a_non_positive_lookahead() -> None:
    with pytest.raises(ValueError, match="post_max"):
        peaks.peak_pick(
            np.zeros(10), pre_max=1, post_max=0, pre_avg=1, post_avg=1, delta=0.1, wait=1
        )


@pytest.mark.parametrize("n_frames", [0, 1, 7])
def test_frame_times_match_librosa(n_frames: int) -> None:
    reference = np.zeros(n_frames)
    expected = librosa.times_like(reference, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    assert np.array_equal(expected, spectral.frame_times(n_frames, SAMPLE_RATE, HOP_LENGTH))
