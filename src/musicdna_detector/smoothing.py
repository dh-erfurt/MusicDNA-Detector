"""Total-variation denoising of a pitch track, before notes are decoded.

The note decoder over-segments because the pitch track wobbles: a frame or two of
deviation inside a held note can buy a state change, and the decoder has no way to
tell a wobble from an attack. Penalties cannot reliably distinguish that wobble from a
genuine attack, so this module operates on the pitch signal before decoding.

Total variation is the other answer, and it is the one the query-by-humming
literature reaches for (Incorporating Total Variation Regularization in the design
of an intelligent Query by Humming system, arXiv:2302.04577). Minimising

    0.5 * ||x - y||^2  +  weight * sum |x[i+1] - x[i]|

produces a **piecewise-constant** signal: it is free to keep a real step, because one
step costs the same however large it is, and it flattens small oscillations, because
many little steps cost more than none. That is exactly the shape of a sung melody,
and it is a prior about the *signal* rather than a penalty on the decoder.

Chambolle's projection algorithm is used rather than a direct solver: at a few
hundred frames the cost is irrelevant and the iteration is short enough to check by
eye. `weight = 0` returns the input untouched, which is the shipped behaviour.

Unvoiced frames are held out entirely -- they carry no pitch, and letting them take
part would smear the ends of notes into silence.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def total_variation_denoise(
    values: FloatArray, weight: float, *, iterations: int = 200
) -> FloatArray:
    """Minimise 0.5*||x-y||^2 + weight*TV(x) for a one-dimensional signal."""

    if weight <= 0 or values.size < 3:
        return values.astype(np.float64, copy=True)
    observed = values.astype(np.float64, copy=False)

    # With D the forward difference, the primal is min_x 0.5||x-y||^2 + w*||Dx||_1,
    # whose dual is min_p 0.5||y - D^T p||^2 subject to |p| <= w, and x = y - D^T p.
    # Projected gradient on p is then three lines, and the step is bounded by
    # 1/||D D^T|| = 1/4.
    def transpose(field: FloatArray) -> FloatArray:
        out = np.zeros_like(observed)
        out[:-1] -= field
        out[1:] += field
        return out

    # Annotated rather than inferred: `np.zeros` fixes the shape to one dimension
    # on Python 3.10's numpy stubs, and `np.clip` below returns the general form.
    dual: FloatArray = np.zeros(observed.size - 1, dtype=np.float64)
    step = 0.24
    for _ in range(iterations):
        primal = observed - transpose(dual)
        dual = np.clip(dual + step * np.diff(primal), -weight, weight)
    return observed - transpose(dual)


def denoise_voiced_runs(values: FloatArray, voiced: NDArray[np.bool_], weight: float) -> FloatArray:
    """Apply the denoiser inside each contiguous voiced run, leaving gaps alone.

    A rest is not a pitch, so smoothing across one would pull the last frames of a
    note toward whatever follows the silence.
    """

    if weight <= 0:
        return values.astype(np.float64, copy=True)
    out = values.astype(np.float64, copy=True)
    start: int | None = None
    for index in range(len(values) + 1):
        inside = index < len(values) and bool(voiced[index])
        if inside and start is None:
            start = index
        elif not inside and start is not None:
            if index - start >= 3:
                out[start:index] = total_variation_denoise(out[start:index], weight)
            start = None
    return out


def median_filter_voiced(values: FloatArray, voiced: NDArray[np.bool_], width: int) -> FloatArray:
    """A running median inside each voiced run -- the classical robust alternative.

    TV keeps steps because a step costs the same however large it is; a median keeps
    them because a majority of the window still belongs to one side. The median cannot
    invent a value the window does not contain, while TV has no fixed window length.
    """

    if width <= 1:
        return values.astype(np.float64, copy=True)
    half = width // 2
    out = values.astype(np.float64, copy=True)
    start: int | None = None
    for index in range(len(values) + 1):
        inside = index < len(values) and bool(voiced[index])
        if inside and start is None:
            start = index
        elif not inside and start is not None:
            run = out[start:index]
            if run.size >= width:
                padded = np.pad(run, half, mode="edge")
                windows = np.lib.stride_tricks.sliding_window_view(padded, width)
                out[start:index] = np.median(windows, axis=1)
            start = None
    return out
