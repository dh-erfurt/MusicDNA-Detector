"""Banded Viterbi decode for the pYIN pitch grid.

pYIN's transition matrix is ``kron(2x2 voicing switch, banded local matrix)``: a state
may only move a bounded number of pitch bins per frame. librosa builds that structure
densely; this implementation stores and walks the active band.

This walks the band. The saving is arithmetic that was never going to change the answer,
so the result is bit-identical rather than approximate -- including the tie-break rule,
which matters more than it sounds: in unvoiced stretches every bin carries the same
probability ``(1 - voiced_prob) / n_pitch_bins`` by construction, so ties are the norm
there, not an edge case. librosa resolves them by ``np.argmax`` over the whole source
row, i.e. lowest state index wins, and so does this.

Out-of-band transitions are not strictly impossible: librosa takes ``log(0 + tiny)``, not
``-inf``, so a sufficiently large value gap could in principle make one win. That cannot
be assumed away, so every frame is guarded against it and falls back to a dense decode
for that frame if the guard trips.
"""

# Portions derived and adapted from librosa 0.11.0 (ISC).
# See THIRD_PARTY_NOTICES.txt for upstream copyright and license.
# MusicDNA-specific modifications are licensed under MIT.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]
BoolArray = NDArray[np.bool_]

TINY = float(np.finfo(np.float64).tiny)
LOG_TINY = float(np.log(TINY))


def _pad_center(data: FloatArray, size: int) -> FloatArray:
    """librosa.util.pad_center for the 1-D case."""
    n = data.shape[0]
    lpad = int((size - n) // 2)
    return np.pad(data, (lpad, int(size - n - lpad)), mode="constant")


def transition_local(n_states: int, width: int) -> FloatArray:
    """Row-normalised triangular band, matching ``librosa.sequence.transition_local``.

    ``wrap=False``, so rows near the edges are truncated and then renormalised -- which
    is why the band is not Toeplitz and has to be stored per target bin.
    """
    transition = np.zeros((n_states, n_states), dtype=np.float64)
    window = scipy.signal.get_window("triangle", width, fftbins=False)
    for i in range(n_states):
        trans_row = _pad_center(np.asarray(window, dtype=np.float64), n_states)
        trans_row = np.roll(trans_row, n_states // 2 + i + 1)
        trans_row[min(n_states, i + width // 2 + 1) :] = 0
        trans_row[: max(0, i - width // 2)] = 0
        transition[i] = trans_row
    transition /= transition.sum(axis=1, keepdims=True)
    return transition


def transition_loop(n_states: int, prob: float) -> FloatArray:
    """``librosa.sequence.transition_loop`` for a scalar self-loop probability."""
    transition = np.empty((n_states, n_states), dtype=np.float64)
    for i in range(n_states):
        transition[i] = (1.0 - prob) / (n_states - 1)
        transition[i, i] = prob
    return transition


@dataclass(frozen=True, slots=True)
class BandedTransition:
    """Banded storage of ``log(kron(t_switch, t_local) + tiny)``.

    ``log_band[source_block, target_block, target_bin, k]`` is the log weight of moving
    from source bin ``target_bin - half + k`` in ``source_block`` into ``target_bin`` in
    ``target_block``. The log is taken over the *product*, exactly as librosa does over
    the assembled matrix: taking ``log(a) + log(b)`` instead differs in the last bit and
    that is enough to flip a tie.
    """

    n_pitch_bins: int
    width: int
    log_band: FloatArray
    source_index: IntArray
    valid: BoolArray
    t_local: FloatArray
    t_switch: FloatArray

    @property
    def n_states(self) -> int:
        return 2 * self.n_pitch_bins


def build_transition(
    n_pitch_bins: int, transition_width: int, switch_prob: float
) -> BandedTransition:
    t_local = transition_local(n_pitch_bins, transition_width)
    t_switch = transition_loop(2, 1 - switch_prob)

    half = transition_width // 2
    targets = np.arange(n_pitch_bins)[:, None]
    offsets = np.arange(transition_width)[None, :]
    source_index = targets - half + offsets
    valid = (source_index >= 0) & (source_index < n_pitch_bins)
    clipped = np.clip(source_index, 0, n_pitch_bins - 1)

    # local[j, k] = t_local[source, j]: the column, because transitions are read into j
    local = t_local[clipped, targets]
    log_band = np.empty((2, 2, n_pitch_bins, transition_width), dtype=np.float64)
    for source_block in range(2):
        for target_block in range(2):
            log_band[source_block, target_block] = np.log(
                t_switch[source_block, target_block] * local + TINY
            )

    return BandedTransition(
        n_pitch_bins=n_pitch_bins,
        width=transition_width,
        log_band=log_band,
        source_index=clipped,
        valid=valid,
        t_local=t_local,
        t_switch=t_switch,
    )


def _dense_log_transition(transition: BandedTransition) -> FloatArray:
    """Full ``log(kron(t_switch, t_local) + tiny)``, only built if a guard trips."""
    dense = np.log(np.kron(transition.t_switch, transition.t_local) + TINY)
    return np.asarray(dense, dtype=np.float64)


def _active_window(previous: FloatArray, n_bins: int, beam: float, half: int) -> tuple[int, int]:
    """Bin range a path can still reach, given how far behind a state may fall.

    States more than ``beam`` nats below the frame's best are treated as dead. Their
    surviving neighbours can move at most ``half`` bins per frame, so the reachable
    target range is the hull of the live bins widened by the band's half-width. The hull
    is taken over both voicing blocks together, because a path may switch between them.
    """
    threshold = float(previous.max()) - beam
    live = np.nonzero(previous >= threshold)[0] % n_bins
    if live.size == 0:  # pragma: no cover - previous is never all -inf in practice
        return 0, n_bins
    low = max(0, int(live.min()) - half)
    high = min(n_bins, int(live.max()) + half + 1)
    return low, high


def decode(
    observation_probs: FloatArray,
    transition: BandedTransition,
    p_init: FloatArray,
    *,
    beam: float | None = None,
) -> NDArray[np.uint16]:
    """Maximum-likelihood state path.

    With ``beam=None`` this is identical to ``librosa.sequence.viterbi``. A finite
    ``beam`` restricts each frame to the pitch bins a path can still plausibly reach,
    which is an approximation because it can remove states from consideration. No finite
    beam width is guaranteed to preserve the exact path for every input.
    """
    n_states, n_steps = observation_probs.shape
    if n_states != transition.n_states:
        raise ValueError("observation probabilities do not match the transition size")
    if beam is not None and beam <= 0:
        raise ValueError("beam must be positive, or None for an exact decode")

    log_prob = np.log(observation_probs + TINY).T
    log_p_init = np.log(p_init + TINY)

    n_bins = transition.n_pitch_bins
    half = transition.width // 2
    clipped = transition.source_index
    invalid = ~transition.valid

    value = np.empty((n_steps, n_states), dtype=np.float64)
    pointer = np.zeros((n_steps, n_states), dtype=np.uint16)
    value[0] = log_prob[0] + log_p_init

    dense_log_transition: FloatArray | None = None

    for step in range(1, n_steps):
        previous = value[step - 1]
        if beam is None:
            low, high = 0, n_bins
        else:
            low, high = _active_window(previous, n_bins, beam, half)
            value[step] = -np.inf

        window = slice(low, high)
        rows = np.arange(high - low)
        window_index = clipped[window]
        window_invalid = invalid[window]
        gathered = [
            previous[block * n_bins : (block + 1) * n_bins][window_index] for block in range(2)
        ]

        best_overall = np.inf
        for target_block in range(2):
            candidates = []
            for source_block in range(2):
                block = (
                    gathered[source_block] + transition.log_band[source_block, target_block, window]
                )
                block[window_invalid] = -np.inf
                offset = np.argmax(block, axis=1)
                candidates.append(
                    (block[rows, offset], window_index[rows, offset] + source_block * n_bins)
                )
            (value_a, pointer_a), (value_b, pointer_b) = candidates
            # Ties go to the lower state index, and block 0 holds the lower indices.
            prefer_a = value_a >= value_b
            best = np.where(prefer_a, value_a, value_b)
            target = slice(target_block * n_bins + low, target_block * n_bins + high)
            value[step, target] = log_prob[step, target] + best
            pointer[step, target] = np.where(prefer_a, pointer_a, pointer_b)
            best_overall = min(best_overall, float(best.min()))

        # Could an out-of-band source have won? Only if the value gap exceeds log(tiny).
        # Under a beam the decode is already approximate, so the exact fallback would be
        # answering a question this mode does not claim to answer.
        if beam is None and float(previous.max()) + LOG_TINY >= best_overall:
            if dense_log_transition is None:
                dense_log_transition = _dense_log_transition(transition)
            transitions_out = previous + dense_log_transition.T
            pointer[step] = np.argmax(transitions_out, axis=1)
            value[step] = log_prob[step] + transitions_out[np.arange(n_states), pointer[step]]

    states = np.zeros(n_steps, dtype=np.uint16)
    states[-1] = int(np.argmax(value[-1]))
    for step in range(n_steps - 2, -1, -1):
        states[step] = pointer[step + 1, states[step + 1]]
    return states
