"""Post-segmentation note consolidation for weak local boundaries."""

from __future__ import annotations

import math
from typing import cast

import numpy as np

from .models import FloatArray, NoteEvent, PitchTrack, SegmentationConfig


def consolidate_note_events(
    track: PitchTrack,
    events: tuple[NoteEvent, ...],
    config: SegmentationConfig,
) -> tuple[NoteEvent, ...]:
    """Merge adjacent same-pitch fragments when the boundary has weak evidence.

    The segmenter should remain recall-oriented: it may emit candidate boundaries from
    pitch or energy evidence. This pass validates only the safest negative case: two
    neighboring events with nearly the same pitch, no real pause, and no energy valley at
    the boundary are more likely vibrato/jitter fragments than separate notes.
    """
    if not config.consolidate_notes or len(events) < 2:
        return events

    ordered = list(sorted(events, key=lambda event: event.start_seconds))
    changed = True
    while changed:
        changed = False
        merged: list[NoteEvent] = []
        index = 0
        while index < len(ordered):
            current = ordered[index]
            if index + 1 >= len(ordered):
                merged.append(current)
                index += 1
                continue
            nxt = ordered[index + 1]
            if _should_merge(track, current, nxt, config):
                merged.append(_merge_events(track, current, nxt, config))
                index += 2
                changed = True
            else:
                merged.append(current)
                index += 1
        ordered = merged
    return tuple(ordered)


def _should_merge(
    track: PitchTrack,
    left: NoteEvent,
    right: NoteEvent,
    config: SegmentationConfig,
) -> bool:
    gap = right.start_seconds - left.end_seconds
    if gap < -1e-9 or gap > config.consolidation_max_gap_seconds:
        return False
    interval = abs(12.0 * math.log2(right.frequency_hz / left.frequency_hz))
    if interval > config.consolidation_same_pitch_semitones:
        return False
    boundary = 0.5 * (left.end_seconds + right.start_seconds)
    if _has_spectral_flux_rearticulation(track, boundary, config):
        return False
    return not _has_energy_rearticulation(track, boundary, config)


def _has_energy_rearticulation(
    track: PitchTrack,
    boundary: float,
    config: SegmentationConfig,
) -> bool:
    levels = cast(FloatArray, track.levels_dbfs)
    if len(levels) == 0 or not np.any(np.isfinite(levels)):
        return False
    times = track.times_seconds
    frame = int(np.searchsorted(times, boundary, side="left"))
    window = config.energy_onset_window_frames
    lo = max(0, frame - window)
    hi = min(len(levels), frame + window + 1)
    if hi - lo < 3:
        return False
    local = levels[lo:hi]
    finite = local[np.isfinite(local)]
    if finite.size < 3:
        return False
    center = float(np.min(finite))
    left = levels[lo : max(lo, frame)]
    right = levels[min(len(levels), frame + 1) : hi]
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if left.size == 0 or right.size == 0:
        return False
    dip_before = float(np.max(left)) - center
    rise_after = float(np.max(right)) - center
    return (
        dip_before >= 0.5 * config.consolidation_min_rearticulation_dbfs
        and rise_after >= config.consolidation_min_rearticulation_dbfs
    )


def _has_spectral_flux_rearticulation(
    track: PitchTrack,
    boundary: float,
    config: SegmentationConfig,
) -> bool:
    if config.onset_feature not in ("spectral_flux", "energy_and_spectral_flux"):
        return False
    if track.onset_strengths is None or len(track.onset_strengths) == 0:
        return False
    strengths = track.onset_strengths
    times = track.times_seconds
    frame = int(np.searchsorted(times, boundary, side="left"))
    window = config.spectral_flux_onset_window_frames
    lo = max(0, frame - window)
    hi = min(len(strengths), frame + window + 1)
    if hi <= lo:
        return False
    local = strengths[lo:hi]
    finite = local[np.isfinite(local)]
    if finite.size == 0:
        return False
    return float(np.max(finite)) >= config.spectral_flux_onset_threshold


def _merge_events(
    track: PitchTrack,
    left: NoteEvent,
    right: NoteEvent,
    config: SegmentationConfig,
) -> NoteEvent:
    start = left.start_seconds
    end = right.end_seconds
    inside = np.flatnonzero(
        (track.times_seconds >= start)
        & (track.times_seconds < end)
        & np.isfinite(track.frequencies_hz)
    )
    if inside.size:
        # Follow the profile's aggregation rule so a merged note is summarized in the
        # same way as every other note. This keeps consolidation a structural pass
        # rather than a second, implicit pitch-aggregation policy.
        frequencies = track.frequencies_hz[inside]
        if config.note_pitch_aggregate == "cents_mean":
            frequency = float(2.0 ** np.mean(np.log2(frequencies)))
        else:
            frequency = float(np.median(frequencies))
        confidence = float(np.mean(track.voiced_probabilities[inside]))
    else:
        left_weight = left.duration_seconds
        right_weight = right.duration_seconds
        frequency = float(
            np.average(
                [left.frequency_hz, right.frequency_hz],
                weights=[left_weight, right_weight],
            )
        )
        confidence = float(
            np.average([left.confidence, right.confidence], weights=[left_weight, right_weight])
        )
    return NoteEvent(
        start,
        end,
        frequency,
        min(1.0, max(0.0, confidence)),
        tuning_reference_hz=config.tuning_reference_hz,
    )
