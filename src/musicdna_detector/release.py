"""Place a note's end at its acoustic release instead of the decoder's last frame.

The rule uses a level threshold relative to each note's own plateau. A relative
threshold avoids imposing one fixed release duration on every recording and leaves
notes whose level does not decay enough at their decoded ending.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from .models import FloatArray, NoteEvent, PitchTrack, SegmentationConfig

# The reference level for "this note was sounding". It is deliberately fixed so the
# release rule remains a stable relative measurement rather than another profile knob.
_PLATEAU_PERCENTILE = 75.0


def trim_note_offsets(
    track: PitchTrack,
    events: tuple[NoteEvent, ...],
    config: SegmentationConfig,
) -> tuple[NoteEvent, ...]:
    """Move each note end to the frame where its level has decayed past the threshold.

    Endings only ever move earlier, never later, and never below the configured minimum
    duration. A note whose level never falls that far -- a legato note that ends because
    the pitch moves on, not because the sound stops -- keeps the decoded ending.
    """
    drop = config.offset_release_drop_dbfs
    if drop <= 0 or not events:
        return events
    times = track.times_seconds
    levels = cast(FloatArray, track.levels_dbfs)
    if len(times) == 0 or len(levels) != len(times):
        return events

    trimmed: list[NoteEvent] = []
    for event in events:
        lo = int(np.searchsorted(times, event.start_seconds, side="left"))
        hi = max(lo, int(np.searchsorted(times, event.end_seconds, side="right")) - 1)
        window = levels[lo : hi + 1]
        finite = window[np.isfinite(window)]
        end = event.end_seconds
        if finite.size >= 2:
            threshold = float(np.percentile(finite, _PLATEAU_PERCENTILE)) - drop
            earliest = event.start_seconds + config.minimum_duration_seconds
            for index in range(lo, hi + 1):
                if times[index] < earliest:
                    continue
                level = float(levels[index])
                if np.isfinite(level) and level <= threshold:
                    end = float(times[index])
                    break
        end = min(
            event.end_seconds, max(end, event.start_seconds + config.minimum_duration_seconds)
        )
        if end >= event.end_seconds:
            trimmed.append(event)
            continue
        trimmed.append(
            NoteEvent(
                event.start_seconds,
                end,
                event.frequency_hz,
                event.confidence,
                tuning_reference_hz=config.tuning_reference_hz,
            )
        )
    return tuple(trimmed)
