"""Tests for optional raw-onset coverage-gap recovery."""

from __future__ import annotations

import numpy as np
import pytest

from musicdna_detector import (
    AnalysisConfig,
    AudioBuffer,
    NoteEvent,
    PitchTrack,
    SegmentationConfig,
    analyze,
    analyze_audio,
    recover_coverage_gaps,
)


def _onset_probe_audio(sample_rate: int = 8_000) -> AudioBuffer:
    samples = np.zeros(sample_rate, dtype=np.float64)
    for start, duration, frequency, amplitude in (
        (0.00, 0.20, 440.00, 0.30),
        (0.36, 0.08, 493.88, 0.80),
        (0.65, 0.20, 523.25, 0.30),
    ):
        offset = int(start * sample_rate)
        size = int(duration * sample_rate)
        times = np.arange(size, dtype=np.float64) / sample_rate
        envelope = np.ones(size, dtype=np.float64)
        attack = min(80, size)
        envelope[:attack] = np.linspace(0.0, 1.0, attack)
        samples[offset : offset + size] += (
            amplitude * envelope * np.sin(2.0 * np.pi * frequency * times)
        )
    return AudioBuffer(samples, sample_rate)


def _repeated_note_audio(sample_rate: int = 16_000) -> AudioBuffer:
    """Deterministic product-test signal with repeated notes and smooth pitch motion."""
    segments: list[np.ndarray] = []

    def silence(duration: float) -> None:
        segments.append(np.zeros(round(duration * sample_rate), dtype=np.float64))

    def tone(frequency: float, duration: float, *, vibrato: bool = False) -> None:
        times = np.arange(round(duration * sample_rate), dtype=np.float64) / sample_rate
        instantaneous = (
            frequency + 5.0 * np.sin(2 * np.pi * 5.5 * times)
            if vibrato
            else np.full(len(times), frequency, dtype=np.float64)
        )
        phase = 2 * np.pi * np.cumsum(instantaneous) / sample_rate
        envelope = np.minimum(1.0, np.linspace(0.0, 1.0, len(times), endpoint=True) * 20.0)
        segments.append(0.35 * envelope * np.sin(phase))

    def glissando(start_hz: float, end_hz: float, duration: float) -> None:
        instantaneous = np.linspace(start_hz, end_hz, round(duration * sample_rate), endpoint=True)
        phase = 2 * np.pi * np.cumsum(instantaneous) / sample_rate
        segments.append(0.35 * np.sin(phase))

    silence(0.20)
    tone(440.0, 0.60)
    silence(0.25)
    tone(440.0, 0.60)
    silence(0.25)
    tone(523.251, 0.60, vibrato=True)
    silence(0.25)
    glissando(587.330, 659.255, 0.60)
    return AudioBuffer(np.concatenate(segments), sample_rate)


def _coverage_gap_track() -> PitchTrack:
    times = np.arange(0.0, 1.0, 0.032, dtype=np.float64)
    frequencies = np.full_like(times, np.nan)
    probabilities = np.zeros_like(times)
    levels = np.full_like(times, -80.0)
    voiced = (times >= 0.47) & (times <= 0.56)
    frequencies[voiced] = 493.88
    probabilities[voiced] = 0.90
    levels[voiced] = -12.0
    return PitchTrack(times, frequencies, probabilities, levels)


def _raw_onset_config(**kwargs: object) -> SegmentationConfig:
    return SegmentationConfig(
        coverage_recovery="raw_onset",
        raw_onset_recovery_min_duration_seconds=0.04,
        raw_onset_recovery_max_duration_seconds=0.12,
        raw_onset_recovery_context_seconds=0.10,
        raw_onset_recovery_min_voiced_frames=2,
        raw_onset_recovery_min_onset_strength=0.25,
        raw_onset_recovery_event_margin_seconds=0.02,
        raw_onset_recovery_hop_length=256,
        **kwargs,  # type: ignore[arg-type]
    )


def test_analysis_config_threads_coverage_recovery_selection() -> None:
    assert AnalysisConfig().segmentation.coverage_recovery == "raw_onset"

    config = AnalysisConfig(coverage_recovery="raw_onset")
    assert config.segmentation.coverage_recovery == "raw_onset"
    assert config.to_dict()["segmentation"]["coverage_recovery"] == "raw_onset"

    disabled = AnalysisConfig(coverage_recovery="off")
    assert disabled.segmentation.coverage_recovery == "off"

    explicit = AnalysisConfig(
        segmentation=SegmentationConfig(
            coverage_recovery="raw_onset",
            raw_onset_recovery_min_voiced_frames=3,
        )
    )
    assert explicit.segmentation.coverage_recovery == "raw_onset"
    assert explicit.segmentation.raw_onset_recovery_min_voiced_frames == 3


def test_off_mode_preserves_exact_segmenter_events() -> None:
    class StubEstimator:
        def estimate(self, audio: AudioBuffer) -> PitchTrack:
            assert audio.sample_rate == 8_000
            return _coverage_gap_track()

    class StubSegmenter:
        def segment(self, track: PitchTrack, *, audio_duration: float) -> tuple[NoteEvent, ...]:
            assert audio_duration == pytest.approx(1.0)
            return (NoteEvent(0.0, 0.20, 440.0, 0.9), NoteEvent(0.65, 0.85, 523.25, 0.9))

    result = analyze(
        _onset_probe_audio(),
        AnalysisConfig(coverage_recovery="off"),
        estimator=StubEstimator(),
        segmenter=StubSegmenter(),
    )
    assert [
        (event.start_seconds, event.end_seconds, event.pitch_name) for event in result.events
    ] == [
        (0.0, 0.2, "A4"),
        (0.65, 0.85, "C5"),
    ]


def test_raw_onset_recovery_adds_plausible_note_in_coverage_gap() -> None:
    events = (NoteEvent(0.0, 0.20, 440.0, 0.9), NoteEvent(0.65, 0.85, 523.25, 0.9))
    recovered = recover_coverage_gaps(
        _onset_probe_audio(),
        _coverage_gap_track(),
        events,
        _raw_onset_config(),
    )
    assert [(event.start_seconds, event.end_seconds, event.pitch_name) for event in recovered] == [
        (0.0, 0.2, "A4"),
        (0.48, 0.5760000000000001, "B4"),
        (0.65, 0.85, "C5"),
    ]
    assert recovered[1].confidence > 0.5


def _split_pitch_track(low_hz: float, high_hz: float) -> PitchTrack:
    """Coverage-gap track whose voiced window straddles two pitch clusters."""
    times = np.arange(0.0, 1.0, 0.032, dtype=np.float64)
    frequencies = np.full_like(times, np.nan)
    probabilities = np.zeros_like(times)
    levels = np.full_like(times, -80.0)
    voiced = (times >= 0.47) & (times <= 0.60)
    indices = np.flatnonzero(voiced)
    half = len(indices) // 2
    frequencies[indices[:half]] = low_hz
    frequencies[indices[half:]] = high_hz
    probabilities[voiced] = 0.90
    levels[voiced] = -12.0
    return PitchTrack(times, frequencies, probabilities, levels)


def test_pitch_spread_gate_rejects_unstable_recovery_window() -> None:
    events = (NoteEvent(0.0, 0.20, 440.0, 0.9), NoteEvent(0.65, 0.85, 523.25, 0.9))
    track = _split_pitch_track(440.0, 659.26)

    ungated = recover_coverage_gaps(
        _onset_probe_audio(),
        track,
        events,
        _raw_onset_config(),
    )
    assert len(ungated) == 3

    gated = recover_coverage_gaps(
        _onset_probe_audio(),
        track,
        events,
        _raw_onset_config(raw_onset_recovery_max_pitch_spread_semitones=1.0),
    )
    assert gated == events


def test_pitch_spread_gate_keeps_stable_recovery_window() -> None:
    events = (NoteEvent(0.0, 0.20, 440.0, 0.9), NoteEvent(0.65, 0.85, 523.25, 0.9))
    recovered = recover_coverage_gaps(
        _onset_probe_audio(),
        _coverage_gap_track(),
        events,
        _raw_onset_config(raw_onset_recovery_max_pitch_spread_semitones=1.0),
    )
    assert [event.pitch_name for event in recovered] == ["A4", "B4", "C5"]


def test_modal_median_pitch_method_avoids_between_cluster_median() -> None:
    events = (NoteEvent(0.0, 0.20, 440.0, 0.9), NoteEvent(0.75, 0.95, 523.25, 0.9))
    track = _split_pitch_track(440.0, 659.26)

    plain = recover_coverage_gaps(
        _onset_probe_audio(),
        track,
        events,
        _raw_onset_config(),
    )
    modal = recover_coverage_gaps(
        _onset_probe_audio(),
        track,
        events,
        _raw_onset_config(raw_onset_recovery_pitch_method="modal_median"),
    )
    assert len(plain) == 3 and len(modal) == 3
    # The plain median of a 50/50 split lands between the clusters; the modal median
    # snaps to one of the actually observed pitches.
    assert plain[1].pitch_name not in ("A4", "E5")
    assert modal[1].pitch_name in ("A4", "E5")


def test_analysis_config_preserves_recovery_pitch_validation_fields() -> None:
    config = AnalysisConfig(
        segmentation=SegmentationConfig(
            raw_onset_recovery_max_pitch_spread_semitones=1.5,
            raw_onset_recovery_pitch_method="modal_median",
        )
    )
    assert config.segmentation.raw_onset_recovery_max_pitch_spread_semitones == 1.5
    assert config.segmentation.raw_onset_recovery_pitch_method == "modal_median"
    encoded = config.to_dict()["segmentation"]
    assert encoded["raw_onset_recovery_max_pitch_spread_semitones"] == 1.5
    assert encoded["raw_onset_recovery_pitch_method"] == "modal_median"

    default = AnalysisConfig()
    assert default.segmentation.raw_onset_recovery_max_pitch_spread_semitones is None
    assert default.segmentation.raw_onset_recovery_pitch_method == "median"
    default.to_json()


def test_raw_onset_recovery_rejects_noise_without_voiced_context() -> None:
    sample_rate = 8_000
    samples = np.zeros(sample_rate, dtype=np.float64)
    samples[int(0.35 * sample_rate)] = 1.0
    times = np.arange(0.0, 1.0, 0.032, dtype=np.float64)
    track = PitchTrack(
        times,
        np.full_like(times, np.nan),
        np.zeros_like(times),
        np.full_like(times, -10.0),
    )
    recovered = recover_coverage_gaps(
        AudioBuffer(samples, sample_rate),
        track,
        (),
        _raw_onset_config(),
    )
    assert recovered == ()


def test_raw_onset_recovery_does_not_fragment_golden_case() -> None:
    audio = _repeated_note_audio()
    result = analyze_audio(
        audio.samples,
        AnalysisConfig(coverage_recovery="raw_onset"),
        sample_rate=audio.sample_rate,
    )
    assert [event.pitch_name for event in result.events] == ["A4", "A4", "C5", "D#5"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"coverage_recovery": "spectral"},
        {"raw_onset_recovery_min_duration_seconds": -0.01},
        {"raw_onset_recovery_max_duration_seconds": 0.01},
        {"raw_onset_recovery_context_seconds": -0.01},
        {"raw_onset_recovery_min_voiced_frames": 0},
        {"raw_onset_recovery_min_onset_strength": -0.01},
        {"raw_onset_recovery_event_margin_seconds": -0.01},
        {"raw_onset_recovery_hop_length": 0},
        {"raw_onset_recovery_max_pitch_spread_semitones": 0.0},
        {"raw_onset_recovery_max_pitch_spread_semitones": -1.0},
        {"raw_onset_recovery_max_pitch_spread_semitones": float("nan")},
        {"raw_onset_recovery_max_pitch_spread_semitones": float("inf")},
        {"raw_onset_recovery_pitch_method": "mode"},
    ],
)
def test_segmentation_config_rejects_invalid_recovery_parameters(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        SegmentationConfig(**kwargs)  # type: ignore[arg-type]
