import json
from dataclasses import asdict, fields, replace

import pytest

from musicdna_detector import (
    AnalysisConfig,
    PitchConfig,
    SegmentationConfig,
    analysis_config_for_profile,
)


def test_default_profile_uses_unsnapped_note_hmm_segmentation() -> None:
    """The default profile uses conservative, unsnapped note-HMM decoding."""
    config = AnalysisConfig()

    assert config.profile == "default"
    assert config.segmentation.segmenter == "note_hmm"
    assert config.segmentation.note_hmm_onset_snap_seconds == 0.0
    assert config.segmentation.note_hmm_attack_unvoiced_penalty == 0.3
    assert config.segmentation.coverage_recovery == "raw_onset"
    # Offset release is enabled only for the humming profile.
    assert config.segmentation.offset_release_drop_dbfs == 0.0
    # A higher persistence cost keeps sustained notes together through brief voicing
    # dips.
    assert config.segmentation.state_silence_transition_penalty == 2.0
    # The default profile uses a wider same-pitch merge gap but keeps a conservative
    # pitch-distance limit.
    assert config.segmentation.consolidation_max_gap_seconds == 0.25
    assert config.segmentation.consolidation_same_pitch_semitones == 0.75
    # Spectral-flux onsets are reserved for the humming profile.
    assert config.segmentation.onset_feature == "energy"
    # Bare configuration defaults must use the same supported decoder.
    assert SegmentationConfig().segmenter == "note_hmm"


def test_default_profile_matches_singing_in_the_v09_contract() -> None:
    default = AnalysisConfig().to_dict()
    singing = AnalysisConfig(profile="singing").to_dict()

    assert default.pop("profile") == "default"
    assert singing.pop("profile") == "singing"
    assert default == singing


def test_singing_profile_uses_unsnapped_note_hmm_segmentation() -> None:
    config = AnalysisConfig(profile="singing")

    assert config.profile == "singing"
    assert config.segmentation.segmenter == "note_hmm"
    assert config.segmentation.note_hmm_onset_snap_seconds == 0.0
    assert config.segmentation.note_hmm_attack_unvoiced_penalty == 0.3
    # Offset release is humming-only.
    assert config.segmentation.offset_release_drop_dbfs == 0.0
    # The singing profile also keeps sustained notes together through brief voicing
    # dips.
    assert config.segmentation.state_silence_transition_penalty == 2.0
    # The wider same-pitch merge gap is shared by the singing profile.
    assert config.segmentation.consolidation_max_gap_seconds == 0.25
    assert config.segmentation.consolidation_same_pitch_semitones == 0.75
    # Spectral-flux onsets remain disabled for singing profiles.
    assert config.segmentation.onset_feature == "energy"


def test_humming_profile_uses_note_hmm_segmentation() -> None:
    config = AnalysisConfig(profile="humming")

    assert config.profile == "humming"
    assert config.segmentation.segmenter == "note_hmm"
    assert config.segmentation.coverage_recovery == "raw_onset"


def test_humming_profile_uses_spectral_flux_to_find_legato_boundaries() -> None:
    """A note that starts without changing pitch is invisible to a pitch contour.

    Spectral flux supplies onset evidence when a same-pitch re-articulation has no
    pitch jump. Both the decoder reward and consolidation guard use this field.
    """
    config = AnalysisConfig(profile="humming")

    assert config.segmentation.onset_feature == "energy_and_spectral_flux"
    assert config.segmentation.spectral_flux_onset_threshold == 0.1
    assert config.segmentation.state_onset_reward == 0.6
    # Consolidation keeps its conservative defaults when flux is enabled.
    assert config.segmentation.consolidation_min_rearticulation_dbfs == 4.0
    assert config.segmentation.consolidation_max_gap_seconds == 0.03
    # Bare dataclass defaults are unchanged, so direct SegmentationConfig() use and
    # every other profile keep the energy-only onset feature.
    assert SegmentationConfig().onset_feature == "energy"
    assert SegmentationConfig().spectral_flux_onset_threshold == 0.3
    assert SegmentationConfig().state_onset_reward == 0.15


def test_profile_helper_returns_analysis_config() -> None:
    config = analysis_config_for_profile("humming")

    assert isinstance(config, AnalysisConfig)
    assert config.segmentation.segmenter == "note_hmm"


def test_explicit_overrides_take_precedence_over_profile_defaults() -> None:
    config = AnalysisConfig(
        profile="humming",
        coverage_recovery="off",
        min_note_duration_seconds=0.08,
    )

    assert config.profile == "humming"
    assert config.segmentation.segmenter == "note_hmm"
    assert config.segmentation.coverage_recovery == "off"
    assert config.segmentation.minimum_duration_seconds == 0.08


def test_profile_serializes_with_effective_config() -> None:
    payload = json.loads(AnalysisConfig(profile="singing").to_json())

    assert payload["profile"] == "singing"
    assert payload["segmentation"]["segmenter"] == "note_hmm"
    assert payload["segmentation"]["note_hmm_onset_snap_seconds"] == 0.0


def test_analysis_config_rebuild_preserves_all_known_config_fields() -> None:
    # Adding a configuration field requires updating this inventory and the
    # non-default fixture below, preventing another silent forwarding loss.
    assert {field.name for field in fields(PitchConfig)} == {
        "fmin_hz",
        "fmax_hz",
        "frame_length",
        "hop_length",
        "silence_threshold_dbfs",
        "smoothing_median_frames",
        "smoothing_tv_weight",
        "estimator",
        "pyin_backend",
        "pyin_beam",
        "pitch_resolution",
        "grid_independent_voicing",
        "level_window_seconds",
    }
    assert {field.name for field in fields(SegmentationConfig)} == {
        "offset_probability",
        "minimum_duration_seconds",
        "maximum_gap_seconds",
        "silence_threshold_dbfs",
        "tuning_reference_hz",
        "segmenter",
        "energy_onset_window_frames",
        "onset_feature",
        "spectral_flux_onset_threshold",
        "spectral_flux_onset_window_frames",
        "spectral_flux_hop_length",
        "coverage_recovery",
        "raw_onset_recovery_min_duration_seconds",
        "raw_onset_recovery_max_duration_seconds",
        "raw_onset_recovery_context_seconds",
        "raw_onset_recovery_min_voiced_frames",
        "raw_onset_recovery_min_onset_strength",
        "raw_onset_recovery_event_margin_seconds",
        "raw_onset_recovery_hop_length",
        "raw_onset_recovery_max_pitch_spread_semitones",
        "raw_onset_recovery_pitch_method",
        "consolidate_notes",
        "consolidation_max_gap_seconds",
        "consolidation_same_pitch_semitones",
        "consolidation_min_rearticulation_dbfs",
        "state_pitch_sigma_semitones",
        "state_change_distance_sigma",
        "state_change_penalty",
        "state_silence_transition_penalty",
        "state_voiced_silence_penalty",
        "state_unvoiced_note_penalty",
        "state_onset_reward",
        "state_onset_window_frames",
        "note_hmm_attack_sigma_factor",
        "note_hmm_observation_trust",
        "note_hmm_attack_unvoiced_penalty",
        "note_hmm_attack_self_cost",
        "note_hmm_rearticulation_penalty",
        "note_hmm_states_per_semitone",
        "note_hmm_onset_snap_seconds",
        "offset_release_drop_dbfs",
        "note_pitch_frames",
        "note_pitch_aggregate",
    }
    pitch = PitchConfig(
        fmin_hz=70.0,
        fmax_hz=1_500.0,
        frame_length=1_024,
        hop_length=128,
        silence_threshold_dbfs=-45.0,
        estimator="pyin",
        pyin_backend="native",
        pyin_beam=50.0,
        pitch_resolution=0.2,
        grid_independent_voicing=True,
        level_window_seconds=0.03,
    )
    segmentation = SegmentationConfig(
        offset_probability=0.4,
        minimum_duration_seconds=0.06,
        maximum_gap_seconds=0.02,
        silence_threshold_dbfs=-45.0,
        tuning_reference_hz=442.0,
        segmenter="note_hmm",
        energy_onset_window_frames=7,
        onset_feature="energy_and_spectral_flux",
        spectral_flux_onset_threshold=0.2,
        spectral_flux_onset_window_frames=5,
        spectral_flux_hop_length=128,
        raw_onset_recovery_min_duration_seconds=0.03,
        raw_onset_recovery_max_duration_seconds=0.14,
        raw_onset_recovery_context_seconds=0.09,
        raw_onset_recovery_min_voiced_frames=4,
        raw_onset_recovery_min_onset_strength=0.3,
        raw_onset_recovery_event_margin_seconds=0.02,
        raw_onset_recovery_hop_length=128,
        raw_onset_recovery_max_pitch_spread_semitones=0.5,
        raw_onset_recovery_pitch_method="modal_median",
        consolidate_notes=False,
        consolidation_max_gap_seconds=0.02,
        consolidation_same_pitch_semitones=0.6,
        consolidation_min_rearticulation_dbfs=3.0,
        state_pitch_sigma_semitones=0.8,
        state_change_penalty=2.5,
        state_silence_transition_penalty=0.8,
        state_voiced_silence_penalty=3.5,
        state_unvoiced_note_penalty=1.2,
        state_onset_reward=0.1,
        state_onset_window_frames=5,
        note_hmm_attack_sigma_factor=2.5,
        note_hmm_attack_unvoiced_penalty=0.4,
        note_hmm_attack_self_cost=0.2,
        note_hmm_rearticulation_penalty=3.5,
        note_hmm_onset_snap_seconds=0.03,
        offset_release_drop_dbfs=6.0,
    )
    rebuilt = replace(
        AnalysisConfig(profile="humming", pitch=pitch, segmentation=segmentation),
        profile="humming",
    )

    assert rebuilt.pitch == pitch
    assert rebuilt.segmentation == segmentation
    payload = json.loads(rebuilt.to_json())
    assert payload["pitch"] == asdict(pitch)
    assert payload["segmentation"] == asdict(segmentation)


def test_analysis_config_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="profile must be"):
        AnalysisConfig(profile="spectral")  # type: ignore[arg-type]


def test_humming_profile_uses_humming_specific_defaults() -> None:
    """The humming profile enables its query-specific pitch and release settings."""
    humming = AnalysisConfig(profile="humming")
    assert humming.segmentation.note_hmm_onset_snap_seconds == pytest.approx(0.04)
    # Humming uses a log-domain mean over pitched frames.
    assert humming.segmentation.note_pitch_aggregate == "cents_mean"
    # Endings are trimmed to the acoustic release.
    assert humming.segmentation.offset_release_drop_dbfs == pytest.approx(8.0)
    # Humming uses a lower persistence penalty than the singing profiles.
    assert humming.segmentation.state_silence_transition_penalty == 1.0
    # The wider merge gap is reserved for the singing profiles.
    assert humming.segmentation.consolidation_max_gap_seconds == 0.03
    assert humming.segmentation.consolidation_same_pitch_semitones == 0.75
    assert humming.segmentation.note_pitch_frames == "all"
    # The shorter level window supplies sharper onset evidence for humming.
    assert humming.pitch.level_window_seconds == pytest.approx(0.023)
    # Pitch distance and re-articulation thresholds stay conservative.
    assert humming.segmentation.consolidation_same_pitch_semitones == pytest.approx(0.75)
    assert humming.segmentation.consolidation_min_rearticulation_dbfs == pytest.approx(4.0)

    # Singing profiles keep the backend level window and median pitch aggregation.
    for profile in ("default", "singing"):
        assert AnalysisConfig(profile=profile).pitch.level_window_seconds is None
        assert AnalysisConfig(profile=profile).segmentation.note_pitch_frames == "all"
        assert AnalysisConfig(profile=profile).segmentation.note_pitch_aggregate == "median"


def test_humming_profile_caps_the_pitch_range_at_c6() -> None:
    """Humming never reaches C6; the wider default only costs pYIN decoder time."""
    humming = AnalysisConfig(profile="humming")
    assert humming.pitch.fmax_hz == pytest.approx(1_046.502)
    assert humming.pitch.fmin_hz == pytest.approx(65.406)

    # Singing can go above C6, so the cap stays profile-scoped.
    for profile in ("default", "singing"):
        assert AnalysisConfig(profile=profile).pitch.fmax_hz == pytest.approx(2_093.005)


def test_explicit_fmax_still_overrides_the_humming_profile() -> None:
    config = AnalysisConfig(profile="humming", fmax_hz=1_500.0)
    assert config.pitch.fmax_hz == pytest.approx(1_500.0)
