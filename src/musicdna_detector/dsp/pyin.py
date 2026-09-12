"""Probabilistic YIN for mono audio, contract-compatible with ``librosa.pyin``.

Only the parameters this detector uses are exposed; the librosa defaults for the rest
are inlined so the numeric result stays comparable. Multi-channel input, ``center=False``
and alternative pad modes are deliberately absent -- the loader always hands us a
centred mono float64 buffer, and unused generality is what made the original slow to
read and impossible to specialise.
"""

# Portions derived and adapted from librosa 0.11.0 (ISC).
# See THIRD_PARTY_NOTICES.txt for upstream copyright and license.
# MusicDNA-specific modifications are licensed under MIT.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .observation import observation_probabilities, threshold_prior
from .viterbi import build_transition, decode

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class PyinResult:
    """What pyin knows, rather than what it is usually asked for.

    ``f0``, ``voiced_flag`` and ``voiced_prob`` are the librosa contract. The two
    candidate arrays are the part the decode throws away: for every frame, the pitch bins
    that carried probability mass before the Viterbi picked one of them. They are shaped
    ``(n_frames, n_candidates)``, strongest first, padded with NaN and 0 where a frame had
    fewer candidates than asked for.

    This matters because YIN's difference function troughs at multiples of the true
    period, so the sub-harmonic an octave down is not a random competitor -- it is the
    systematic one. A decode that collapses each frame to a single value cannot tell a
    later stage that the second-best answer was exactly an octave away.
    """

    f0: FloatArray
    voiced_flag: BoolArray
    voiced_prob: FloatArray
    candidate_frequencies_hz: FloatArray | None = None
    candidate_probabilities: FloatArray | None = None


# librosa.pyin defaults, fixed here because the equality gate compares against them.
N_THRESHOLDS = 100
BETA_PARAMETERS = (2.0, 18.0)
BOLTZMANN_PARAMETER = 2.0
RESOLUTION = 0.1
MAX_TRANSITION_RATE = 35.92
SWITCH_PROB = 0.01
NO_TROUGH_PROB = 0.01


def _top_candidates(
    observation_probs: FloatArray, freqs: FloatArray, n_candidates: int
) -> tuple[FloatArray, FloatArray]:
    """The ``n_candidates`` most probable pitch bins per frame, strongest first."""
    voiced = observation_probs[: freqs.size]
    n_frames = voiced.shape[1]
    take = min(n_candidates, voiced.shape[0])
    partitioned = np.argpartition(-voiced, take - 1, axis=0)[:take]
    probabilities = np.take_along_axis(voiced, partitioned, axis=0)
    order = np.argsort(-probabilities, axis=0, kind="stable")
    bins = np.take_along_axis(partitioned, order, axis=0).T
    probabilities = np.take_along_axis(probabilities, order, axis=0).T

    frequencies = freqs[bins]
    empty = probabilities <= 0.0
    frequencies[empty] = np.nan
    probabilities[empty] = 0.0

    if take < n_candidates:  # pragma: no cover - only for absurdly small pitch ranges
        pad = n_candidates - take
        frequencies = np.hstack([frequencies, np.full((n_frames, pad), np.nan)])
        probabilities = np.hstack([probabilities, np.zeros((n_frames, pad))])
    return frequencies, probabilities


def pyin_detailed(
    y: FloatArray,
    *,
    sr: float,
    fmin: float,
    fmax: float,
    frame_length: int = 2048,
    hop_length: int | None = None,
    resolution: float = RESOLUTION,
    beam: float | None = None,
    n_candidates: int = 0,
    grid_independent_voicing: bool = False,
) -> PyinResult:
    """Run pYIN and optionally keep the per-frame candidate distribution.

    ``n_candidates=0`` is the plain decode and costs nothing extra. Any other value adds
    one partition per frame over the pitch bins and changes no decision anywhere: the
    candidates ride alongside the decoded track for later stages to use or ignore.
    """
    if fmin <= 0 or fmax <= fmin:
        raise ValueError("pyin needs 0 < fmin < fmax")
    if y.ndim != 1:
        raise ValueError("this pyin implementation is mono only")
    if hop_length is None:
        hop_length = frame_length // 4

    from .yin import (
        cumulative_mean_normalized_difference,
        frame_signal,
        parabolic_interpolation,
    )

    padding = frame_length // 2
    padded = np.pad(np.asarray(y, dtype=np.float64), (padding, padding), mode="constant")
    y_frames = frame_signal(padded, frame_length, hop_length)

    min_period = int(np.floor(sr / fmax))
    max_period = min(int(np.ceil(sr / fmin)), frame_length - 1)

    yin_frames = cumulative_mean_normalized_difference(y_frames, min_period, max_period)
    parabolic_shifts = parabolic_interpolation(yin_frames)

    thresholds, beta_probs = threshold_prior(N_THRESHOLDS, BETA_PARAMETERS)

    n_bins_per_semitone = int(np.ceil(1.0 / resolution))
    n_pitch_bins = int(np.floor(12 * n_bins_per_semitone * np.log2(fmax / fmin))) + 1

    reference_bins = None
    if grid_independent_voicing:
        # The bin count the default resolution would have produced for this range, so
        # the voiced/unvoiced balance stays where every threshold in the pipeline was
        # calibrated, whatever `resolution` is set to.
        reference_default = int(np.ceil(1.0 / RESOLUTION))
        reference_bins = int(np.floor(12 * reference_default * np.log2(fmax / fmin))) + 1

    observation_probs, voiced_prob = observation_probabilities(
        yin_frames,
        parabolic_shifts,
        sr=sr,
        thresholds=thresholds,
        boltzmann_parameter=BOLTZMANN_PARAMETER,
        beta_probs=beta_probs,
        no_trough_prob=NO_TROUGH_PROB,
        min_period=min_period,
        fmin=fmin,
        n_pitch_bins=n_pitch_bins,
        n_bins_per_semitone=n_bins_per_semitone,
        unvoiced_reference_bins=reference_bins,
    )

    max_semitones_per_frame = round(MAX_TRANSITION_RATE * 12 * hop_length / sr)
    transition_width = max_semitones_per_frame * n_bins_per_semitone + 1
    transition = build_transition(n_pitch_bins, transition_width, SWITCH_PROB)

    p_init = np.ones(2 * n_pitch_bins) / (2 * n_pitch_bins)
    states = decode(observation_probs, transition, p_init, beam=beam)

    freqs = fmin * 2 ** (np.arange(n_pitch_bins) / (12 * n_bins_per_semitone))
    f0 = freqs[states % n_pitch_bins]
    voiced_flag = states < n_pitch_bins
    f0[~voiced_flag] = np.nan

    candidate_frequencies = candidate_probabilities = None
    if n_candidates > 0:
        candidate_frequencies, candidate_probabilities = _top_candidates(
            observation_probs, freqs, n_candidates
        )

    return PyinResult(f0, voiced_flag, voiced_prob, candidate_frequencies, candidate_probabilities)


def pyin(
    y: FloatArray,
    *,
    sr: float,
    fmin: float,
    fmax: float,
    frame_length: int = 2048,
    hop_length: int | None = None,
    resolution: float = RESOLUTION,
    beam: float | None = None,
) -> tuple[FloatArray, BoolArray, FloatArray]:
    """Return ``(f0, voiced_flag, voiced_prob)``, the librosa-compatible contract.

    ``f0`` is NaN on unvoiced frames, as in librosa's default ``fill_na``. ``beam``
    prunes the decode; ``None`` keeps it exact.
    """
    result = pyin_detailed(
        y,
        sr=sr,
        fmin=fmin,
        fmax=fmax,
        frame_length=frame_length,
        hop_length=hop_length,
        resolution=resolution,
        beam=beam,
    )
    return result.f0, result.voiced_flag, result.voiced_prob
