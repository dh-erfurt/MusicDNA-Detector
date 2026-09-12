from __future__ import annotations

import numpy as np
import pytest

from musicdna_detector.consolidation import consolidate_note_events
from musicdna_detector.models import NoteEvent, PitchTrack, SegmentationConfig


def _track(levels: list[float], onset_strengths: list[float] | None = None) -> PitchTrack:
    times = np.arange(len(levels), dtype=float) * 0.02
    return PitchTrack(
        times,
        np.full(len(levels), 440.0),
        np.full(len(levels), 0.95),
        np.array(levels, dtype=float),
        None if onset_strengths is None else np.array(onset_strengths, dtype=float),
    )


def test_consolidation_merges_adjacent_same_pitch_without_rearticulation() -> None:
    track = _track([-10.0] * 10)
    events = (
        NoteEvent(0.0, 0.1, 440.0, 0.9),
        NoteEvent(0.1, 0.2, 442.0, 0.9),
    )

    merged = consolidate_note_events(track, events, SegmentationConfig())

    assert len(merged) == 1
    assert merged[0].start_seconds == 0.0
    assert merged[0].end_seconds == 0.2
    assert merged[0].pitch_name == "A4"


def test_consolidation_keeps_same_pitch_repeat_with_energy_valley() -> None:
    track = _track([-10.0, -10.0, -10.0, -30.0, -10.0, -10.0, -10.0])
    events = (
        NoteEvent(0.0, 0.08, 440.0, 0.9),
        NoteEvent(0.08, 0.14, 440.0, 0.9),
    )

    merged = consolidate_note_events(track, events, SegmentationConfig())

    assert len(merged) == 2


def test_consolidation_keeps_same_pitch_repeat_with_spectral_flux_rearticulation() -> None:
    track = _track([-10.0] * 10, [0.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0])
    events = (
        NoteEvent(0.0, 0.08, 440.0, 0.9),
        NoteEvent(0.08, 0.20, 440.0, 0.9),
    )

    merged = consolidate_note_events(
        track,
        events,
        SegmentationConfig(onset_feature="spectral_flux", spectral_flux_onset_threshold=0.5),
    )

    assert len(merged) == 2


def test_consolidation_keeps_real_pitch_change() -> None:
    track = _track([-10.0] * 10)
    events = (
        NoteEvent(0.0, 0.1, 440.0, 0.9),
        NoteEvent(0.1, 0.2, 493.88, 0.9),
    )

    merged = consolidate_note_events(track, events, SegmentationConfig())

    assert [event.pitch_name for event in merged] == ["A4", "B4"]


def test_merge_follows_the_configured_pitch_aggregation() -> None:
    """A merged note must use the profile's pitch rule, not always the median.

    The frames below are deliberately skewed: the log mean and the median of the same
    frames differ, so a merge that ignored `note_pitch_aggregate` would be visible here.
    """
    times = np.arange(10, dtype=float) * 0.02
    frequencies = np.array([440.0, 440.0, 440.0, 440.0, 440.0, 441.0, 441.0, 460.0, 460.0, 460.0])
    track = PitchTrack(
        times,
        frequencies,
        np.full(10, 0.95),
        np.full(10, -10.0),
    )
    events = (
        NoteEvent(0.0, 0.1, 440.0, 0.9),
        NoteEvent(0.1, 0.2, 441.0, 0.9),
    )

    median = consolidate_note_events(track, events, SegmentationConfig())
    log_mean = consolidate_note_events(
        track, events, SegmentationConfig(note_pitch_aggregate="cents_mean")
    )

    assert len(median) == 1
    assert len(log_mean) == 1
    # The merged note spans 0.0 to 0.2 s and every frame time is below that.
    assert median[0].frequency_hz == pytest.approx(float(np.median(frequencies)))
    assert log_mean[0].frequency_hz == pytest.approx(float(2.0 ** np.mean(np.log2(frequencies))))
    assert median[0].frequency_hz != pytest.approx(log_mean[0].frequency_hz)
