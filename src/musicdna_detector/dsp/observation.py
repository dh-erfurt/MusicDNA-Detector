"""Per-frame observation probabilities over the pitch grid.

This follows the behavior implemented by librosa 0.11.0 rather than attempting to
reconstruct the paper alone: a beta prior over YIN thresholds, a Boltzmann prior over
the troughs below each threshold, and a fixed share of probability handed to the global
minimum when no trough clears the threshold.

The detailed pYIN API can retain the multi-candidate distribution assembled here. The
standard estimator collapses it to one decoded frequency before note segmentation.

librosa does all of this one frame at a time. Most of it does not need to be: trough
detection, the threshold comparison, the running trough rank and the Boltzmann prior are
elementwise or per-column, so they are computed for many frames per call here. One step
resists batching and is deliberately left per frame -- see ``_frame_probabilities``.
"""

# Portions derived and adapted from librosa 0.11.0 (ISC).
# See THIRD_PARTY_NOTICES.txt for upstream copyright and license.
# MusicDNA-specific modifications are licensed under MIT.

from __future__ import annotations

import numpy as np
import scipy.stats
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.intp]

# How many frames share one batched call. This is a cache-sized throughput choice, not
# a numerical parameter: changing it must not change the returned probabilities.
MAX_TROUGHS_PER_BLOCK = 256


def threshold_prior(
    n_thresholds: int, beta_parameters: tuple[float, float]
) -> tuple[FloatArray, FloatArray]:
    """Return the YIN threshold grid and the beta prior mass per threshold bin."""
    thresholds = np.linspace(0, 1, n_thresholds + 1)
    beta_cdf = scipy.stats.beta.cdf(thresholds, beta_parameters[0], beta_parameters[1])
    return (
        np.asarray(thresholds, dtype=np.float64),
        np.asarray(np.diff(beta_cdf), dtype=np.float64),
    )


def _troughs(yin_frames: FloatArray) -> BoolArray:
    """Local minima per frame, with librosa's edge rules, for every frame at once."""
    is_trough = np.zeros(yin_frames.shape, dtype=bool)
    if yin_frames.shape[0] < 2:
        return is_trough
    is_trough[1:-1] = (yin_frames[1:-1] < yin_frames[:-2]) & (yin_frames[1:-1] <= yin_frames[2:])
    is_trough[-1] = yin_frames[-1] < yin_frames[-2]
    # pyin overrides the first bin: localmin leaves it False, pyin compares it against
    # its right neighbour instead.
    is_trough[0] = yin_frames[0] < yin_frames[1]
    return is_trough


def _beta_prefix_sums(beta_probs: FloatArray) -> FloatArray:
    """``[sum(beta_probs[:n]) for n in 0..len]``, each one an independent ``np.sum``.

    A cumulative sum would be cheaper and would *not* give these numbers: ``np.sum``
    reduces pairwise, ``np.cumsum`` accumulates sequentially, and the two disagree in the
    last bits. This value lands in the probabilities, so the cheap version is wrong here.
    """
    return np.array([np.sum(beta_probs[:n]) for n in range(beta_probs.size + 1)], dtype=np.float64)


def _frame_probabilities(
    prior_block: FloatArray,
    below_block: BoolArray,
    heights_block: FloatArray,
    beta_probs: FloatArray,
    beta_prefix: FloatArray,
    no_trough_prob: float,
) -> FloatArray:
    """Collapse one frame's trough priors onto its troughs.

    The matrix-vector product stays per frame because changing the reduction shape can
    change floating-point rounding. Preserving this operation order keeps parity with
    the reference implementation.
    """
    probabilities: FloatArray = prior_block.dot(beta_probs)
    global_min = int(np.argmin(heights_block))
    n_below_min = below_block.shape[1] - int(np.count_nonzero(below_block[global_min]))
    probabilities[global_min] += no_trough_prob * beta_prefix[n_below_min]
    return probabilities


def _blocks(counts: NDArray[np.int_]) -> list[tuple[int, int]]:
    """Frame ranges whose combined trough count stays inside the block budget."""
    blocks: list[tuple[int, int]] = []
    start = 0
    running = 0
    for frame, count in enumerate(counts):
        if running and running + int(count) > MAX_TROUGHS_PER_BLOCK:
            blocks.append((start, frame))
            start = frame
            running = 0
        running += int(count)
    if start < counts.size:
        blocks.append((start, int(counts.size)))
    return blocks


def observation_probabilities(
    yin_frames: FloatArray,
    parabolic_shifts: FloatArray,
    *,
    sr: float,
    thresholds: FloatArray,
    boltzmann_parameter: float,
    beta_probs: FloatArray,
    no_trough_prob: float,
    min_period: int,
    fmin: float,
    n_pitch_bins: int,
    n_bins_per_semitone: int,
    unvoiced_reference_bins: int | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Return ``(observation_probs, voiced_prob)`` for a mono clip.

    ``observation_probs`` has shape ``(2 * n_pitch_bins, n_frames)``: the voiced bins
    first, then the unvoiced copy. ``voiced_prob`` has shape ``(n_frames,)``.

    ``unvoiced_reference_bins`` fixes what the unvoiced mass is divided by. librosa
    divides by ``n_pitch_bins``, which makes every unvoiced state more likely as the
    pitch grid gets coarser -- the unvoiced hypothesis should not depend on how finely
    pitch was discretised, and that coupling is what stops the grid from being a pure
    throughput knob. ``None`` reproduces librosa.
    """
    yin_probs = np.zeros_like(yin_frames)

    is_trough = _troughs(yin_frames)
    counts = np.count_nonzero(is_trough, axis=0)
    # Transposed so the non-zeros arrive grouped by frame and ascending in period within
    # a frame -- the order librosa's per-frame loop produces.
    period_of_trough = np.nonzero(is_trough.T)[1]
    frame_of_trough = np.repeat(np.arange(counts.size), counts)
    heights = yin_frames[period_of_trough, frame_of_trough]
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    beta_prefix = _beta_prefix_sums(beta_probs)

    for first_frame, last_frame in _blocks(counts):
        block_starts = starts[first_frame:last_frame]
        block_counts = counts[first_frame:last_frame]
        low = int(block_starts[0])
        high = int(block_starts[-1] + block_counts[-1])
        if high == low:
            continue

        block_heights = heights[low:high]
        below = np.less.outer(block_heights, thresholds[1:])

        # Rank of each trough among the troughs below the same threshold, restarted at
        # every frame boundary: one cumulative sum minus each frame's own base.
        ranks = np.zeros((below.shape[0] + 1, below.shape[1]), dtype=np.int64)
        np.cumsum(below, axis=0, out=ranks[1:])
        local_starts = block_starts - low
        base = ranks[local_starts]
        totals = ranks[local_starts + block_counts] - base
        rows_of_frame = np.repeat(np.arange(block_counts.size), block_counts)

        positions = ranks[1:] - base[rows_of_frame] - 1
        prior = scipy.stats.boltzmann.pmf(positions, boltzmann_parameter, totals[rows_of_frame])
        prior[~below] = 0

        for local_frame, (start, count) in enumerate(zip(local_starts, block_counts, strict=True)):
            if count == 0:
                continue
            stop = int(start) + int(count)
            probabilities = _frame_probabilities(
                prior[start:stop],
                below[start:stop],
                block_heights[start:stop],
                beta_probs,
                beta_prefix,
                no_trough_prob,
            )
            yin_probs[period_of_trough[low + start : low + stop], first_frame + local_frame] = (
                probabilities
            )

    yin_period, frame_index = np.nonzero(yin_probs)

    integer_periods = min_period + yin_period
    period_candidates = integer_periods + parabolic_shifts[yin_period, frame_index]
    f0_candidates = sr / period_candidates

    bin_position = 12 * n_bins_per_semitone * np.log2(f0_candidates / fmin)
    bin_index = np.clip(np.round(bin_position), 0, n_pitch_bins).astype(int)

    observation_probs = np.zeros((2 * n_pitch_bins, yin_frames.shape[1]))
    observation_probs[bin_index, frame_index] = yin_probs[yin_period, frame_index]

    voiced_prob = np.clip(np.sum(observation_probs[:n_pitch_bins, :], axis=0, keepdims=True), 0, 1)
    divisor = n_pitch_bins if unvoiced_reference_bins is None else unvoiced_reference_bins
    observation_probs[n_pitch_bins:, :] = (1 - voiced_prob) / divisor

    return observation_probs, voiced_prob[0]
