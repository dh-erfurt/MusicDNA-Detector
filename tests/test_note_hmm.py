"""Tests for the attack-state HMM note decoder."""

from __future__ import annotations

import math

import numpy as np
import pytest

from musicdna_detector import (
    AnalysisConfig,
    NoteHmmSegmenter,
    PitchTrack,
    SegmentationConfig,
    build_segmenter,
)


def _track(
    frequencies: list[float],
    *,
    levels: list[float] | None = None,
    onset_strengths: list[float] | None = None,
    step: float = 0.02,
) -> PitchTrack:
    count = len(frequencies)
    times = np.arange(count, dtype=np.float64) * step
    array = np.array(frequencies, dtype=np.float64)
    probabilities = np.where(np.isfinite(array), 0.95, 0.05)
    level_array = np.full(count, -20.0) if levels is None else np.array(levels, dtype=np.float64)
    onset_array = None if onset_strengths is None else np.array(onset_strengths)
    return PitchTrack(times, array, probabilities, level_array, onset_array)


def test_factory_builds_note_hmm() -> None:
    segmenter = build_segmenter(SegmentationConfig(segmenter="note_hmm"))
    assert isinstance(segmenter, NoteHmmSegmenter)


def test_decodes_two_pitch_notes_with_gap() -> None:
    frequencies = [440.0] * 20 + [float("nan")] * 5 + [523.25] * 20
    segmenter = NoteHmmSegmenter(SegmentationConfig(minimum_duration_seconds=0.05))
    events = segmenter.segment(_track(frequencies), audio_duration=0.9)
    assert len(events) == 2
    assert events[0].pitch_name == "A4"
    assert events[1].pitch_name == "C5"
    assert events[0].end_seconds <= events[1].start_seconds


def test_rearticulation_splits_same_pitch_on_onset_evidence() -> None:
    # 40 voiced A4 frames with a strong energy dip + rise and a spectral-flux
    # spike at frame 20: the attack re-entry should split the run in two.
    frequencies = [440.0] * 40
    levels = [-20.0] * 17 + [-38.0, -38.0, -38.0] + [-14.0] * 20
    onsets = [0.0] * 20 + [4.0] + [0.0] * 19
    config = SegmentationConfig(
        segmenter="note_hmm",
        minimum_duration_seconds=0.05,
        note_hmm_rearticulation_penalty=0.3,
        onset_feature="energy_and_spectral_flux",
    )
    events = NoteHmmSegmenter(config).segment(
        _track(frequencies, levels=levels, onset_strengths=onsets), audio_duration=0.9
    )
    assert len(events) == 2
    assert all(event.pitch_name == "A4" for event in events)
    assert events[1].start_seconds == pytest.approx(0.40, abs=0.03)

    # Without the onset evidence and with the default penalty, the run stays one note.
    plain = NoteHmmSegmenter(
        SegmentationConfig(segmenter="note_hmm", minimum_duration_seconds=0.05)
    ).segment(_track(frequencies), audio_duration=0.9)
    assert len(plain) == 1


def test_empty_and_unvoiced_tracks_produce_no_events() -> None:
    segmenter = NoteHmmSegmenter(SegmentationConfig())
    empty = PitchTrack(
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
    )
    assert segmenter.segment(empty, audio_duration=0.0) == ()
    silent = _track([float("nan")] * 10)
    assert segmenter.segment(silent, audio_duration=0.2) == ()


def test_note_hmm_config_validation() -> None:
    with pytest.raises(ValueError, match="note_hmm"):
        SegmentationConfig(note_hmm_attack_sigma_factor=0.5)
    with pytest.raises(ValueError, match="note_hmm"):
        SegmentationConfig(note_hmm_rearticulation_penalty=-1.0)


def test_note_hmm_runs_inside_full_pipeline(tmp_path) -> None:
    from dataclasses import replace

    import soundfile as sf

    from musicdna_detector import analyze_audio

    sample_rate = 16_000
    tone = 0.35 * np.sin(2 * np.pi * 440.0 * np.arange(sample_rate) / sample_rate)
    path = tmp_path / "tone.wav"
    sf.write(path, tone, sample_rate)
    config = AnalysisConfig()
    config = replace(config, segmentation=replace(config.segmentation, segmenter="note_hmm"))
    result = analyze_audio(path, config)
    assert len(result.events) >= 1
    assert result.events[0].pitch_name == "A4"


def test_onset_snap_moves_start_to_evidence_peak() -> None:
    # Voicing starts at frame 12, but the energy onset is at frame 8: with
    # snapping enabled the note start moves back to the evidence peak.
    frequencies = [float("nan")] * 12 + [440.0] * 20
    levels = [-55.0] * 8 + [-18.0] * 24
    config = SegmentationConfig(
        segmenter="note_hmm",
        minimum_duration_seconds=0.05,
        note_hmm_onset_snap_seconds=0.12,
    )
    events = NoteHmmSegmenter(config).segment(
        _track(frequencies, levels=levels), audio_duration=0.7
    )
    assert len(events) == 1
    assert events[0].start_seconds == pytest.approx(0.16, abs=0.021)

    # Disabled snap keeps the voicing-aligned boundary.
    plain_config = SegmentationConfig(segmenter="note_hmm", minimum_duration_seconds=0.05)
    plain = NoteHmmSegmenter(plain_config).segment(
        _track(frequencies, levels=levels), audio_duration=0.7
    )
    assert len(plain) == 1
    assert plain[0].start_seconds >= 0.22


def test_note_pitch_frames_sustain_ignores_the_transient_attack() -> None:
    """The decoder treats ATTACK as transient pitch; 'sustain' keeps that judgement.

    The note is deliberately short. A median over a long note shrugs off a few
    scooped attack frames all by itself -- that is what a median is for. The attack
    only moves the estimate when it is a large share of the note, which is the
    normal case for a hummed query, not the exception.
    """
    from musicdna_detector.models import PitchTrack, SegmentationConfig
    from musicdna_detector.note_hmm import NoteHmmSegmenter

    # A short note whose first frames scoop up from a semitone below into a stable A4.
    scoop = [415.3, 421.0, 428.0, 434.0, 437.0]
    body = [440.0] * 5
    frequencies = np.array(scoop + body, dtype=np.float64)
    size = len(frequencies)
    track = PitchTrack(
        np.arange(size, dtype=np.float64) * 0.01,
        frequencies,
        np.full(size, 0.95, dtype=np.float64),
        np.full(size, -12.0, dtype=np.float64),
    )

    def pitch(mode: str) -> float:
        config = SegmentationConfig(
            segmenter="note_hmm",
            minimum_duration_seconds=0.02,
            note_pitch_frames=mode,  # type: ignore[arg-type]
        )
        events = NoteHmmSegmenter(config).segment(track, audio_duration=size * 0.01)
        assert events, f"no events for mode {mode}"
        return events[0].frequency_hz

    assert pitch("sustain") == pytest.approx(440.0, abs=1e-9)
    assert pitch("all") < pitch("sustain")


def test_note_pitch_frames_rejects_an_unknown_mode() -> None:
    from musicdna_detector.models import SegmentationConfig

    with pytest.raises(ValueError, match="note_pitch_frames"):
        SegmentationConfig(note_pitch_frames="attack")  # type: ignore[arg-type]


def test_cents_mean_returns_the_centre_of_a_modulated_tone() -> None:
    """A vibrato's pitch is its centre; the median returns an arbitrary sweep point.

    The frames sweep asymmetrically around A4 -- more time is spent on the flat side,
    as a singer sagging into a note does. The median follows that time bias, the log
    mean stays near the centre the listener names.
    """
    from musicdna_detector.models import PitchTrack, SegmentationConfig
    from musicdna_detector.note_hmm import NoteHmmSegmenter

    centre = 440.0
    cents = [-60.0, -55.0, -50.0, -45.0, -40.0, -20.0, 10.0, 40.0, 55.0, 65.0]
    frequencies = np.array([centre * 2.0 ** (value / 1200.0) for value in cents])
    size = len(frequencies)
    track = PitchTrack(
        np.arange(size, dtype=np.float64) * 0.01,
        frequencies,
        np.full(size, 0.95, dtype=np.float64),
        np.full(size, -12.0, dtype=np.float64),
    )

    def pitch(aggregate: str) -> float:
        config = SegmentationConfig(
            segmenter="note_hmm",
            minimum_duration_seconds=0.02,
            note_pitch_aggregate=aggregate,  # type: ignore[arg-type]
        )
        events = NoteHmmSegmenter(config).segment(track, audio_duration=size * 0.01)
        assert events, f"no events for {aggregate}"
        return events[0].frequency_hz

    def error_cents(frequency: float) -> float:
        return abs(1200.0 * math.log2(frequency / centre))

    assert error_cents(pitch("cents_mean")) == pytest.approx(abs(sum(cents) / len(cents)), abs=1e-6)
    assert error_cents(pitch("cents_mean")) < error_cents(pitch("median"))


def test_note_pitch_aggregate_rejects_an_unknown_mode() -> None:
    from musicdna_detector.models import SegmentationConfig

    with pytest.raises(ValueError, match="note_pitch_aggregate"):
        SegmentationConfig(note_pitch_aggregate="mean")  # type: ignore[arg-type]


def test_vibrato_stays_one_note_and_a_real_step_does_not() -> None:
    """Ported from the removed hysteresis tests: these two are the domain's oldest
    segmentation properties, and note_hmm is now the only decoder that has to hold
    them. A semitone of wobble is a singer, not a note boundary."""
    config = SegmentationConfig(minimum_duration_seconds=0.02)
    vibrato = NoteHmmSegmenter(config).segment(
        _track([438, 442, 438, 442, 438, 442, 438, 442]), audio_duration=0.16
    )
    step = NoteHmmSegmenter(config).segment(
        _track([440, 440, 440, 440, 523.25, 523.25, 523.25, 523.25]), audio_duration=0.16
    )
    assert len(vibrato) == 1
    assert [event.pitch_name for event in step] == ["A4", "C5"]


def test_gradual_glissando_is_not_fragmented() -> None:
    events = NoteHmmSegmenter(SegmentationConfig(minimum_duration_seconds=0.02)).segment(
        _track([440, 445, 450, 455, 460, 465, 470, 475]), audio_duration=0.16
    )
    assert len(events) == 1


def test_event_end_is_clamped_to_audio_duration() -> None:
    events = NoteHmmSegmenter(SegmentationConfig(minimum_duration_seconds=0.02)).segment(
        _track([440, 440, 440, 440]), audio_duration=0.05
    )
    assert events
    assert max(event.end_seconds for event in events) <= 0.05
