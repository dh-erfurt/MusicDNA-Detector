from __future__ import annotations

import numpy as np
import pytest

from musicdna_detector.models import NoteEvent, PitchTrack, SegmentationConfig
from musicdna_detector.release import trim_note_offsets


def _track(levels: list[float], hop_seconds: float = 0.02) -> PitchTrack:
    times = np.arange(len(levels), dtype=float) * hop_seconds
    return PitchTrack(
        times,
        np.full(len(levels), 440.0),
        np.full(len(levels), 0.95),
        np.array(levels, dtype=float),
    )


def _config(**overrides: object) -> SegmentationConfig:
    return SegmentationConfig(offset_release_drop_dbfs=8.0, **overrides)  # type: ignore[arg-type]


def test_disabled_by_default() -> None:
    track = _track([-10.0] * 5 + [-40.0] * 5)
    events = (NoteEvent(0.0, 0.2, 440.0, 0.9),)

    assert trim_note_offsets(track, events, SegmentationConfig()) == events


def test_end_moves_to_the_decay() -> None:
    # Six frames at plateau, then the level collapses at 0.12 s.
    track = _track([-10.0] * 6 + [-30.0] * 4)
    events = (NoteEvent(0.0, 0.2, 440.0, 0.9),)

    trimmed = trim_note_offsets(track, events, _config())

    assert trimmed[0].end_seconds == pytest.approx(0.12)
    assert trimmed[0].start_seconds == pytest.approx(0.0)


def test_sustained_note_keeps_its_decoded_end() -> None:
    """A note that never decays -- legato, the pitch moves on instead -- is left alone."""
    track = _track([-10.0] * 10)
    events = (NoteEvent(0.0, 0.2, 440.0, 0.9),)

    assert trim_note_offsets(track, events, _config()) == events


def test_never_shortens_below_the_minimum_duration() -> None:
    # The level falls one frame in, which alone would leave a 20 ms note.
    track = _track([-10.0, -40.0] + [-40.0] * 8)
    events = (NoteEvent(0.0, 0.2, 440.0, 0.9),)

    trimmed = trim_note_offsets(track, events, _config(minimum_duration_seconds=0.05))

    assert trimmed[0].duration_seconds >= 0.05


def test_endings_never_move_later() -> None:
    track = _track([-40.0] * 4 + [-10.0] * 6)
    events = (NoteEvent(0.0, 0.06, 440.0, 0.9),)

    trimmed = trim_note_offsets(track, events, _config())

    assert trimmed[0].end_seconds <= events[0].end_seconds


def test_pitch_and_confidence_survive_the_trim() -> None:
    track = _track([-10.0] * 6 + [-30.0] * 4)
    events = (NoteEvent(0.0, 0.2, 441.0, 0.42),)

    trimmed = trim_note_offsets(track, events, _config())

    assert trimmed[0].frequency_hz == pytest.approx(441.0)
    assert trimmed[0].confidence == pytest.approx(0.42)
    assert trimmed[0].pitch_name == events[0].pitch_name


def test_silent_levels_are_ignored() -> None:
    track = _track([-10.0, -np.inf, -10.0, -np.inf, -10.0, -30.0, -30.0, -30.0])
    events = (NoteEvent(0.0, 0.16, 440.0, 0.9),)

    trimmed = trim_note_offsets(track, events, _config())

    assert trimmed[0].end_seconds == pytest.approx(0.10)


def test_negative_drop_is_rejected() -> None:
    with pytest.raises(ValueError, match="offset_release_drop_dbfs"):
        SegmentationConfig(offset_release_drop_dbfs=-1.0)
