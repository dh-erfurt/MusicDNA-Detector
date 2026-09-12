"""Specialized DSP operations used by the detector's native pitch path.

The implementation targets centered mono float64 input and preserves the numerical
behavior used from librosa 0.11.0 while avoiding a runtime dependency on librosa. The
optional librosa backend remains the reference used by parity tests. Source and license
provenance are recorded in ``THIRD_PARTY_NOTICES.txt``.
"""

from __future__ import annotations

from .peaks import onset_detect, peak_pick
from .pyin import PyinResult, pyin, pyin_detailed
from .spectral import (
    frame_times,
    melspectrogram,
    normalized_onset_strength,
    onset_strength,
    power_to_db,
    rms,
    stft,
)

__all__ = [
    "PyinResult",
    "frame_times",
    "melspectrogram",
    "normalized_onset_strength",
    "onset_detect",
    "onset_strength",
    "peak_pick",
    "power_to_db",
    "pyin",
    "pyin_detailed",
    "rms",
    "stft",
]
