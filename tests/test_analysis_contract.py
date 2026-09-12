import importlib.util
import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from musicdna_detector import (
    ANALYSIS_RESULT_SCHEMA_VERSION,
    AnalysisConfig,
    AnalysisResult,
    AudioSource,
    BackendInfo,
    DecoderBackend,
    Diagnostics,
    NoteEvent,
    PitchEvent,
    PitchTrack,
    SegmentationConfig,
    analyze_audio,
    frequency_to_note,
    round_half_up,
)


def test_product_package_excludes_benchmark_runtime() -> None:
    import musicdna_detector

    for name in ("benchmark", "metrics", "report"):
        assert importlib.util.find_spec(f"musicdna_detector.{name}") is None
    for name in (
        "BenchmarkReport",
        "GoldenAudioCase",
        "evaluate_notes_standard",
        "run_benchmark_report",
    ):
        assert not hasattr(musicdna_detector, name)


def test_note_event_is_canonical_and_pitch_event_compatible() -> None:
    event = NoteEvent(0.1, 0.6, 440.0, 0.9)
    assert isinstance(event, PitchEvent)
    assert event.duration_seconds == pytest.approx(0.5)
    assert event.duration_ms == pytest.approx(500.0)
    assert (event.midi_pitch, event.pitch_name, event.cents) == (69, "A4", 0.0)
    with pytest.raises(FrozenInstanceError):
        event.pitch_name = "A#4"  # type: ignore[misc]
    with pytest.raises(ValueError, match="midi_pitch"):
        NoteEvent(0.1, 0.6, 440.0, 0.9, midi_pitch=70)
    with pytest.raises(ValueError, match="cents"):
        NoteEvent(0.1, 0.6, 440.0, 0.9, cents=float("nan"))


def test_mapping_uses_tuning_and_round_half_up() -> None:
    assert round_half_up(1.5) == 2
    assert round_half_up(-1.5) == -1
    assert frequency_to_note(442.0, 442.0).pitch_name == "A4"
    halfway = 440.0 * 2 ** (0.5 / 12)
    assert frequency_to_note(halfway).midi_pitch == 70
    for invalid in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            frequency_to_note(invalid)


def test_config_accepts_flat_and_legacy_nested_contract() -> None:
    config = AnalysisConfig(
        sample_rate=16_000,
        fmin_hz=80.0,
        fmax_hz=1_000.0,
        min_note_duration_seconds=0.12,
        silence_threshold_dbfs=-48.0,
        decoder_policy="native_then_ffmpeg",
    )
    assert config.sample_rate == 16_000
    assert config.pitch.fmin_hz == 80.0
    assert config.segmentation.minimum_duration_seconds == 0.12
    assert config.loader.decoder_policy == "native_then_ffmpeg"


def test_nested_config_keeps_silence_on_both_stages() -> None:
    # This field belonged to the removed hysteresis segmenter and is not part of the
    # current note-HMM configuration.
    config = AnalysisConfig(
        segmentation=SegmentationConfig(silence_threshold_dbfs=-42.0),
    )
    assert config.silence_threshold_dbfs == -42.0
    assert config.segmentation.silence_threshold_dbfs == -42.0


def test_result_json_has_null_for_unpitched_and_no_path() -> None:
    track = PitchTrack(
        np.array([0.0, 0.1]),
        np.array([np.nan, 440.0]),
        np.array([0.0, 0.9]),
        np.array([-np.inf, -12.0]),
    )
    result = AnalysisResult(
        (),
        track,
        0.2,
        8_000,
        AnalysisConfig(),
        AudioSource("path", "wav"),
        BackendInfo(DecoderBackend.NATIVE),
        Diagnostics(),
    )
    encoded = result.to_json()
    assert "NaN" not in encoded and "Infinity" not in encoded
    assert "secret" not in encoded
    decoded = json.loads(encoded)
    assert (
        result.schema_version
        == decoded["schema_version"]
        == ANALYSIS_RESULT_SCHEMA_VERSION
        == "musicdna-analysis-v0"
    )
    assert "path" not in decoded["source"]
    assert decoded["pitch_track"]["frequencies_hz"][0] is None
    assert decoded["pitch_track"]["onset_strengths"] is None
    assert set(decoded["source"]) == {
        "kind",
        "format",
        "original_sample_rate",
        "analyzed_sample_rate",
        "channels",
        "duration_s",
        "resampled",
    }
    assert decoded["backend"]["decoder_policy"] == "native_only"


def test_analysis_serialization_keeps_pitch_event_alias_contract() -> None:
    event = PitchEvent(0.0, 0.2, 440.0, 0.8)
    result = AnalysisResult(
        (event,),
        PitchTrack(
            np.array([0.0]),
            np.array([440.0]),
            np.array([0.8]),
            onset_strengths=np.array([0.35]),
        ),
        0.2,
        8_000,
        AnalysisConfig(minimum_duration_seconds=0.05),
    )
    decoded = json.loads(result.to_json())
    assert decoded["schema_version"] == ANALYSIS_RESULT_SCHEMA_VERSION
    assert set(decoded["events"][0]) == {
        "start_seconds",
        "end_seconds",
        "duration_seconds",
        "duration_ms",
        "frequency_hz",
        "midi_pitch",
        "pitch_name",
        "confidence",
        "cents",
    }
    assert decoded["events"][0]["duration_ms"] == pytest.approx(200.0)
    assert decoded["pitch_track"]["onset_strengths"] == pytest.approx([0.35])
    assert decoded["config"]["segmentation"]["minimum_duration_seconds"] == 0.05


def test_ndarray_requires_explicit_sample_rate() -> None:
    samples = np.sin(2 * np.pi * 440 * np.arange(4_000) / 8_000)
    with pytest.raises(TypeError, match="sample_rate is required"):
        analyze_audio(samples)
    result = analyze_audio(samples, sample_rate=8_000)
    assert result.source.kind == "array"
    with pytest.raises(ValueError, match="one-dimensional"):
        analyze_audio(np.column_stack((samples, samples)), sample_rate=8_000)
