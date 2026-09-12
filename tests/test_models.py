import json
import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from musicdna_detector import (
    AnalysisConfig,
    AnalysisResult,
    AudioBuffer,
    LoaderConfig,
    PitchConfig,
    PitchEvent,
    PitchTrack,
    SegmentationConfig,
    SourceMetadata,
)


def test_audio_buffer_is_defensively_immutable() -> None:
    samples = np.array([0.0, 0.5])
    audio = AudioBuffer(samples, 2)
    samples[0] = 1.0
    assert audio.samples.tolist() == [0.0, 0.5]
    with pytest.raises(ValueError):
        audio.samples[0] = 1.0
    with pytest.raises(ValueError):
        audio.samples.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        audio.sample_rate = 4  # type: ignore[misc]


def test_result_has_strict_json_output() -> None:
    track = PitchTrack(np.array([0.0]), np.array([440.0]), np.array([0.9]))
    result = AnalysisResult(
        (PitchEvent(0.0, 0.1, 440.0, 0.9),), track, 0.1, 22_050, AnalysisConfig()
    )
    decoded = json.loads(result.to_json())
    assert decoded["events"][0]["frequency_hz"] == 440.0
    assert decoded["config"]["loader"]["target_sample_rate"] == 22_050
    assert json.loads(AnalysisConfig().to_json())["pitch"]["hop_length"] == 256


def test_analysis_config_exposes_decoder_policy() -> None:
    assert AnalysisConfig().decoder_policy == "native_only"
    assert AnalysisConfig(decoder_policy="native_then_ffmpeg").decoder_policy == (
        "native_then_ffmpeg"
    )


def test_backend_metadata_defaults_describe_current_runtime() -> None:
    from musicdna_detector import BackendMetadata, DecoderBackend

    metadata = BackendMetadata(DecoderBackend.MEMORY)
    assert metadata.pitch_estimator == "PyinPitchEstimator"
    assert metadata.segmenter == "NoteHmmSegmenter"


@pytest.mark.parametrize("sample_rate", [0, -1])
def test_audio_buffer_rejects_invalid_rate(sample_rate: int) -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        AudioBuffer(np.zeros(1), sample_rate)


@pytest.mark.parametrize(
    ("config_type", "kwargs"),
    [
        (PitchConfig, {"fmin_hz": float("nan")}),
        (SegmentationConfig, {"maximum_gap_seconds": float("nan")}),
    ],
)
def test_configs_reject_non_finite_values(
    config_type: type[object], kwargs: dict[str, float]
) -> None:
    with pytest.raises(ValueError, match="finite"):
        config_type(**kwargs)


@pytest.mark.parametrize(
    ("config_type", "kwargs", "message"),
    [
        (PitchConfig, {"smoothing_tv_weight": float("nan")}, "smoothing_tv_weight"),
        (PitchConfig, {"smoothing_tv_weight": float("inf")}, "smoothing_tv_weight"),
        (PitchConfig, {"smoothing_tv_weight": -0.1}, "smoothing_tv_weight"),
        (PitchConfig, {"smoothing_median_frames": -1}, "smoothing_median_frames"),
        (PitchConfig, {"smoothing_median_frames": 2}, "smoothing_median_frames"),
        (PitchConfig, {"smoothing_median_frames": 3.0}, "smoothing_median_frames"),
        (
            SegmentationConfig,
            {"note_hmm_observation_trust": float("nan")},
            "note_hmm_observation_trust",
        ),
        (
            SegmentationConfig,
            {"note_hmm_observation_trust": -0.1},
            "note_hmm_observation_trust",
        ),
        (
            SegmentationConfig,
            {"note_hmm_states_per_semitone": 0},
            "note_hmm_states_per_semitone",
        ),
        (
            SegmentationConfig,
            {"note_hmm_states_per_semitone": 1.5},
            "note_hmm_states_per_semitone",
        ),
        (
            SegmentationConfig,
            {"state_change_distance_sigma": float("inf")},
            "state_change_distance_sigma",
        ),
        (
            SegmentationConfig,
            {"state_change_distance_sigma": 0.0},
            "state_change_distance_sigma",
        ),
    ],
)
def test_decoder_measurement_knobs_reject_invalid_values(
    config_type: type[object], kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        config_type(**kwargs)


def test_note_hmm_states_per_semitone_is_capped_at_pyins_measured_grid() -> None:
    """Three matches pYIN's nPPS; higher values only inflate quadratic HMM matrices."""
    assert SegmentationConfig(note_hmm_states_per_semitone=3).note_hmm_states_per_semitone == 3
    with pytest.raises(ValueError, match=r"note_hmm_states_per_semitone.*\[1, 3\]"):
        SegmentationConfig(note_hmm_states_per_semitone=4)


def test_result_defensively_freezes_events_and_checks_audio_bounds() -> None:
    track = PitchTrack(np.array([0.0]), np.array([440.0]), np.array([0.9]))
    event = PitchEvent(0.0, 0.1, 440.0, 0.9)
    events = [event]
    result = AnalysisResult(events, track, 0.1, 22_050, AnalysisConfig())  # type: ignore[arg-type]
    events.clear()
    assert result.events == (event,)
    with pytest.raises(ValueError, match="audio duration"):
        AnalysisResult((event,), track, 0.09, 22_050, AnalysisConfig())


def test_zero_minimum_duration_is_refused_at_the_config_not_at_release() -> None:
    """A zero minimum used to be accepted and then crash trim_note_offsets.

    The segmenter can emit a note of length zero under that setting, and NoteEvent
    rejects it three stages later with an error that names neither the knob nor the
    configuration that caused it.
    """
    with pytest.raises(ValueError, match="minimum_duration_seconds must be positive"):
        SegmentationConfig(minimum_duration_seconds=0.0)
    with pytest.raises(ValueError, match="minimum_duration_seconds must be positive"):
        SegmentationConfig(minimum_duration_seconds=-0.01)


@pytest.mark.parametrize(
    "field_name",
    [
        "fmin_hz",
        "fmax_hz",
        "level_window_seconds",
        "silence_threshold_dbfs",
        "pyin_beam",
        "smoothing_tv_weight",
        "pitch_resolution",
    ],
)
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_pitch_config_rejects_every_non_finite_float(field_name: str, invalid: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        PitchConfig(**{field_name: invalid})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field_name",
    [
        "offset_probability",
        "minimum_duration_seconds",
        "maximum_gap_seconds",
        "silence_threshold_dbfs",
        "tuning_reference_hz",
        "spectral_flux_onset_threshold",
        "raw_onset_recovery_min_duration_seconds",
        "raw_onset_recovery_max_duration_seconds",
        "raw_onset_recovery_context_seconds",
        "raw_onset_recovery_min_onset_strength",
        "raw_onset_recovery_event_margin_seconds",
        "raw_onset_recovery_max_pitch_spread_semitones",
        "consolidation_max_gap_seconds",
        "consolidation_same_pitch_semitones",
        "consolidation_min_rearticulation_dbfs",
        "note_hmm_observation_trust",
        "state_pitch_sigma_semitones",
        "state_change_distance_sigma",
        "state_change_penalty",
        "state_silence_transition_penalty",
        "state_voiced_silence_penalty",
        "state_unvoiced_note_penalty",
        "state_onset_reward",
        "note_hmm_attack_sigma_factor",
        "note_hmm_attack_unvoiced_penalty",
        "note_hmm_attack_self_cost",
        "note_hmm_rearticulation_penalty",
        "note_hmm_onset_snap_seconds",
        "offset_release_drop_dbfs",
    ],
)
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_segmentation_config_rejects_every_non_finite_float(
    field_name: str, invalid: float
) -> None:
    with pytest.raises(ValueError, match="finite"):
        SegmentationConfig(**{field_name: invalid})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("config_type", "field_name"),
    [
        (LoaderConfig, "target_sample_rate"),
        (PitchConfig, "frame_length"),
        (PitchConfig, "hop_length"),
        (SegmentationConfig, "energy_onset_window_frames"),
        (SegmentationConfig, "spectral_flux_onset_window_frames"),
        (SegmentationConfig, "spectral_flux_hop_length"),
        (SegmentationConfig, "raw_onset_recovery_min_voiced_frames"),
        (SegmentationConfig, "raw_onset_recovery_hop_length"),
        (SegmentationConfig, "state_onset_window_frames"),
    ],
)
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_config_integer_fields_reject_non_finite_surrogates(
    config_type: type[object], field_name: str, invalid: float
) -> None:
    with pytest.raises(ValueError):
        config_type(**{field_name: invalid})  # type: ignore[call-arg]


def test_pitch_resolution_must_be_positive() -> None:
    with pytest.raises(ValueError, match="pitch_resolution"):
        PitchConfig(pitch_resolution=0.0)
    with pytest.raises(ValueError, match="pitch_resolution"):
        PitchConfig(pitch_resolution=-0.1)


def test_pitch_track_supports_only_domain_meaningful_non_finite_values() -> None:
    track = PitchTrack(
        np.array([0.0, 0.1]),
        np.array([np.nan, 440.0]),
        np.array([0.0, 0.9]),
        np.array([-np.inf, -12.0]),
    )
    assert np.isnan(track.frequencies_hz[0])
    assert np.isneginf(track.levels_dbfs[0])

    for levels in (np.array([np.nan]), np.array([np.inf])):
        with pytest.raises(ValueError, match="levels_dbfs"):
            PitchTrack(np.array([0.0]), np.array([440.0]), np.array([0.9]), levels)


@pytest.mark.parametrize("sample_rate", [float("nan"), float("inf"), float("-inf")])
def test_result_and_source_reject_non_finite_sample_rates(sample_rate: float) -> None:
    track = PitchTrack(np.array([0.0]), np.array([440.0]), np.array([0.9]))
    with pytest.raises(ValueError, match="sample_rate"):
        AnalysisResult((), track, 0.1, sample_rate, AnalysisConfig())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sample rates"):
        SourceMetadata("buffer", original_sample_rate=sample_rate)  # type: ignore[arg-type]


def test_result_rejects_unordered_and_overlapping_events() -> None:
    track = PitchTrack(np.array([0.0]), np.array([440.0]), np.array([0.9]))
    first = PitchEvent(0.0, 0.2, 440.0, 0.9)
    overlapping = PitchEvent(0.1, 0.3, 442.0, 0.8)
    earlier = PitchEvent(0.3, 0.4, 444.0, 0.8)

    with pytest.raises(ValueError, match="ordered and non-overlapping"):
        AnalysisResult((first, overlapping), track, 0.4, 22_050, AnalysisConfig())
    with pytest.raises(ValueError, match="ordered and non-overlapping"):
        AnalysisResult((earlier, first), track, 0.4, 22_050, AnalysisConfig())

    adjacent = PitchEvent(0.2, 0.3, 442.0, 0.8)
    assert AnalysisResult((first, adjacent), track, 0.3, 22_050, AnalysisConfig()).events == (
        first,
        adjacent,
    )

    nearly_adjacent = PitchEvent(math.nextafter(0.2, 0.0), 0.3, 442.0, 0.8)
    assert AnalysisResult(
        (first, nearly_adjacent), track, 0.3, 22_050, AnalysisConfig()
    ).events == (first, nearly_adjacent)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_remaining_serialized_numeric_fields_reject_non_finite_values(
    invalid: float,
) -> None:
    with pytest.raises(ValueError, match="ffmpeg_timeout_seconds"):
        LoaderConfig(ffmpeg_timeout_seconds=invalid)
    with pytest.raises(ValueError, match="normalization_target_dbfs"):
        AnalysisConfig(normalization_target_dbfs=invalid)
    with pytest.raises(ValueError, match="normalization_reference_percentile"):
        AnalysisConfig(normalization_reference_percentile=invalid)
    with pytest.raises(ValueError, match="duration_s"):
        SourceMetadata("buffer", duration_s=invalid)

    track = PitchTrack(np.array([0.0]), np.array([440.0]), np.array([0.9]))
    with pytest.raises(ValueError, match="duration_seconds"):
        AnalysisResult((), track, invalid, 22_050, AnalysisConfig())


@pytest.mark.parametrize("field_name", ["original_sample_rate", "analyzed_sample_rate", "channels"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_source_integer_fields_reject_non_finite_surrogates(
    field_name: str, invalid: float
) -> None:
    with pytest.raises(ValueError):
        SourceMetadata("buffer", **{field_name: invalid})  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid", [True, np.bool_(True)])
def test_serialized_numeric_fields_reject_boolean_values(invalid: object) -> None:
    with pytest.raises(ValueError, match="ffmpeg_timeout_seconds"):
        LoaderConfig(ffmpeg_timeout_seconds=invalid)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="pitch numeric fields"):
        PitchConfig(smoothing_tv_weight=invalid)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="segmentation numeric fields"):
        SegmentationConfig(maximum_gap_seconds=invalid)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="normalization numeric fields"):
        AnalysisConfig(normalization_target_dbfs=invalid)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="duration_s"):
        SourceMetadata("buffer", duration_s=invalid)  # type: ignore[arg-type]

    track = PitchTrack(np.array([0.0]), np.array([440.0]), np.array([0.9]))
    with pytest.raises(ValueError, match="duration_seconds"):
        AnalysisResult((), track, invalid, 22_050, AnalysisConfig())  # type: ignore[arg-type]


def test_source_metadata_rejects_invalid_serialized_literals() -> None:
    with pytest.raises(ValueError, match="source kind"):
        SourceMetadata("invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="source format"):
        SourceMetadata("buffer", format=123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="resampled"):
        SourceMetadata("buffer", resampled=1)  # type: ignore[arg-type]
