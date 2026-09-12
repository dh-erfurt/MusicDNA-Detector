"""Immutable public domain models and configuration."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from itertools import pairwise
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .contract import ANALYSIS_RESULT_SCHEMA_VERSION, EVENT_ORDER_TOLERANCE_SECONDS

FloatArray = NDArray[np.float64]


def _is_int(value: object) -> bool:
    """Return whether *value* is a JSON-safe integer, excluding booleans."""
    return type(value) is int


def _is_bool(value: object) -> bool:
    """Return whether *value* is a Python or NumPy boolean scalar."""
    return isinstance(value, (bool, np.bool_))


def _reject_boolean_numbers(values: tuple[object, ...], message: str) -> None:
    if any(_is_bool(value) for value in values):
        raise ValueError(message)


def _immutable_vector(
    values: ArrayLike, name: str, *, allow_nan: bool = False, allow_infinite: bool = False
) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if (not allow_infinite and np.any(np.isinf(result))) or (
        not allow_nan and np.any(np.isnan(result))
    ):
        raise ValueError(f"{name} contains invalid non-finite values")
    return np.frombuffer(np.ascontiguousarray(result).tobytes(), dtype=np.float64)


def _immutable_level_vector(values: ArrayLike) -> FloatArray:
    """Freeze dBFS values while preserving ``-inf`` as digital silence."""
    result = _immutable_vector(values, "levels_dbfs", allow_infinite=True)
    if np.any(np.isnan(result)) or np.any(np.isposinf(result)):
        raise ValueError("levels_dbfs contains invalid non-finite values")
    return result


def _json_number(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _json_vector(values: FloatArray) -> list[float | None]:
    return [_json_number(float(value)) for value in values]


class DecoderBackend(str, Enum):
    NATIVE = "native"
    FFMPEG = "ffmpeg"
    MEMORY = "memory"


DecoderPolicy = Literal["native_only", "native_then_ffmpeg"]
# One member since 0.5. The alias stays so a future decoder is introduced by name,
# the way note_hmm was, rather than by changing what build_segmenter returns.
SegmenterName = Literal["note_hmm"]
CoverageRecoveryMode = Literal["off", "raw_onset"]
RecoveryPitchMethod = Literal["median", "modal_median"]
PitchEstimatorName = Literal["pyin"]
PyinBackendName = Literal["librosa", "native"]
OnsetFeatureMode = Literal["energy", "spectral_flux", "energy_and_spectral_flux"]
NotePitchFrameMode = Literal["all", "sustain"]
NotePitchAggregateMode = Literal["median", "cents_mean"]
AnalysisProfileName = Literal["default", "singing", "humming"]


@dataclass(frozen=True, slots=True)
class SourceMetadata:
    """Privacy-safe source metadata. It intentionally never stores a path."""

    kind: Literal["path", "buffer", "array"]
    format: str | None = None
    original_sample_rate: int = 0
    analyzed_sample_rate: int = 0
    channels: int = 1
    duration_s: float = 0.0
    resampled: bool = False

    def __post_init__(self) -> None:
        if self.kind not in ("path", "buffer", "array"):
            raise ValueError("source kind must be 'path', 'buffer', or 'array'")
        if self.format is not None and not isinstance(self.format, str):
            raise ValueError("source format must be a string or None")
        if type(self.resampled) is not bool:
            raise ValueError("source resampled must be a boolean")
        if (
            not _is_int(self.original_sample_rate)
            or not _is_int(self.analyzed_sample_rate)
            or self.original_sample_rate < 0
            or self.analyzed_sample_rate < 0
        ):
            raise ValueError("source sample rates must be non-negative")
        if not _is_int(self.channels) or self.channels <= 0:
            raise ValueError("source channels must be positive")
        if _is_bool(self.duration_s) or not math.isfinite(self.duration_s) or self.duration_s < 0:
            raise ValueError("source duration_s must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class BackendMetadata:
    decoder: DecoderBackend
    decoder_policy: DecoderPolicy = "native_only"
    pitch_estimator: str = "PyinPitchEstimator"
    segmenter: str = "NoteHmmSegmenter"
    ffmpeg_used: bool = False

    def __post_init__(self) -> None:
        if self.decoder_policy not in ("native_only", "native_then_ffmpeg"):
            raise ValueError("invalid decoder_policy")
        if self.ffmpeg_used and self.decoder != DecoderBackend.FFMPEG:
            raise ValueError("ffmpeg_used requires the ffmpeg decoder")
        if type(self.ffmpeg_used) is not bool:
            raise ValueError("ffmpeg_used must be a boolean")


@dataclass(frozen=True, slots=True)
class AnalysisWarning:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class Diagnostics:
    warnings: tuple[AnalysisWarning, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "warnings", tuple(self.warnings))


@dataclass(frozen=True, slots=True)
class AudioBuffer:
    samples: FloatArray
    sample_rate: int
    source: SourceMetadata = field(default_factory=lambda: SourceMetadata("buffer"))
    decoder_backend: DecoderBackend = DecoderBackend.MEMORY
    decoder_policy: DecoderPolicy = "native_only"

    def __post_init__(self) -> None:
        if not _is_int(self.sample_rate) or self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        object.__setattr__(self, "samples", _immutable_vector(self.samples, "samples"))
        source = self.source
        if source.original_sample_rate == 0:
            source = SourceMetadata(
                kind=source.kind,
                format=source.format,
                original_sample_rate=self.sample_rate,
                analyzed_sample_rate=self.sample_rate,
                channels=source.channels,
                duration_s=len(self.samples) / self.sample_rate,
                resampled=False,
            )
        object.__setattr__(self, "source", source)

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate


@dataclass(frozen=True, slots=True)
class PitchTrack:
    """Frame observations. NaN frequency means audible or silent but unpitched."""

    times_seconds: FloatArray
    frequencies_hz: FloatArray
    voiced_probabilities: FloatArray
    levels_dbfs: FloatArray | None = None
    onset_strengths: FloatArray | None = None

    def __post_init__(self) -> None:
        times = _immutable_vector(self.times_seconds, "times_seconds")
        legacy_frequencies = np.array(self.frequencies_hz, dtype=np.float64, copy=True)
        legacy_frequencies[legacy_frequencies == 0] = np.nan
        frequencies = _immutable_vector(legacy_frequencies, "frequencies_hz", allow_nan=True)
        probabilities = _immutable_vector(self.voiced_probabilities, "voiced_probabilities")
        levels = (
            np.zeros(len(times), dtype=np.float64)
            if self.levels_dbfs is None
            else _immutable_level_vector(self.levels_dbfs)
        )
        onset_strengths = (
            None
            if self.onset_strengths is None
            else _immutable_vector(self.onset_strengths, "onset_strengths")
        )
        for name, value in (
            ("times_seconds", times),
            ("frequencies_hz", frequencies),
            ("voiced_probabilities", probabilities),
            ("levels_dbfs", levels),
            ("onset_strengths", onset_strengths),
        ):
            object.__setattr__(self, name, value)
        size = len(times)
        vectors = (frequencies, probabilities, levels)
        if any(len(values) != size for values in vectors) or (
            onset_strengths is not None and len(onset_strengths) != size
        ):
            raise ValueError("pitch-track arrays must have equal length")
        if np.any(times < 0) or np.any(np.diff(times) < 0):
            raise ValueError("times_seconds must be non-negative and ordered")
        if np.any(frequencies[np.isfinite(frequencies)] <= 0):
            raise ValueError("pitched frequencies must be positive")
        if np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("voiced_probabilities must be in [0, 1]")
        if onset_strengths is not None and np.any(onset_strengths < 0):
            raise ValueError("onset_strengths must be non-negative")


@dataclass(frozen=True, slots=True, init=False)
class NoteEvent:
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    duration_ms: float
    frequency_hz: float
    midi_pitch: int
    pitch_name: str
    confidence: float
    cents: float

    def __init__(
        self,
        start_seconds: float,
        end_seconds: float,
        frequency_hz: float,
        confidence: float,
        *,
        duration_seconds: float | None = None,
        midi_pitch: int | None = None,
        pitch_name: str | None = None,
        cents: float | None = None,
        tuning_reference_hz: float = 440.0,
    ) -> None:
        # Local import keeps mapping.py independent of the domain models.
        from .mapping import frequency_to_note

        _reject_boolean_numbers(
            (
                start_seconds,
                end_seconds,
                frequency_hz,
                confidence,
                duration_seconds,
                midi_pitch,
                cents,
                tuning_reference_hz,
            ),
            "note-event numeric values must not be boolean",
        )
        mapped = frequency_to_note(frequency_hz, tuning_reference_hz)
        duration = end_seconds - start_seconds
        values = (start_seconds, end_seconds, duration, frequency_hz, confidence)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("note-event values must be finite")
        if start_seconds < 0 or duration <= 0:
            raise ValueError("a note event must have positive duration")
        if frequency_hz <= 0 or not 0 <= confidence <= 1:
            raise ValueError("frequency must be positive and confidence must be in [0, 1]")
        if not math.isfinite(duration * 1000.0):
            raise ValueError("note-event duration_ms must be finite")
        if duration_seconds is not None and not math.isclose(duration_seconds, duration):
            raise ValueError("duration_seconds must equal end_seconds - start_seconds")
        if midi_pitch is not None and (not _is_int(midi_pitch) or midi_pitch != mapped.midi_pitch):
            raise ValueError("midi_pitch must match frequency_hz and tuning_reference_hz")
        if pitch_name is not None and pitch_name != mapped.pitch_name:
            raise ValueError("pitch_name must match frequency_hz and tuning_reference_hz")
        if cents is not None and (
            not math.isfinite(cents) or not math.isclose(cents, mapped.cents)
        ):
            raise ValueError("cents must match frequency_hz and tuning_reference_hz")
        object.__setattr__(self, "start_seconds", float(start_seconds))
        object.__setattr__(self, "end_seconds", float(end_seconds))
        object.__setattr__(self, "duration_seconds", float(duration))
        object.__setattr__(self, "duration_ms", float(duration) * 1000.0)
        object.__setattr__(self, "frequency_hz", float(frequency_hz))
        object.__setattr__(
            self, "midi_pitch", mapped.midi_pitch if midi_pitch is None else midi_pitch
        )
        object.__setattr__(
            self, "pitch_name", mapped.pitch_name if pitch_name is None else pitch_name
        )
        object.__setattr__(self, "confidence", float(confidence))
        object.__setattr__(self, "cents", mapped.cents if cents is None else float(cents))


# Source-compatible name retained for v2 callers and serialized consumers.
PitchEvent = NoteEvent


@dataclass(frozen=True, slots=True)
class LoaderConfig:
    target_sample_rate: int | None = 22_050
    decoder_policy: DecoderPolicy = "native_only"
    ffmpeg_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.target_sample_rate is not None and (
            not _is_int(self.target_sample_rate) or self.target_sample_rate <= 0
        ):
            raise ValueError("target_sample_rate must be positive or None")
        if self.decoder_policy not in ("native_only", "native_then_ffmpeg"):
            raise ValueError("decoder_policy must be native_only or native_then_ffmpeg")
        if (
            _is_bool(self.ffmpeg_timeout_seconds)
            or not math.isfinite(self.ffmpeg_timeout_seconds)
            or self.ffmpeg_timeout_seconds <= 0
        ):
            raise ValueError("ffmpeg_timeout_seconds must be positive")


@dataclass(frozen=True, slots=True)
class PitchConfig:
    fmin_hz: float = 65.406
    fmax_hz: float = 2_093.005
    frame_length: int = 2_048
    hop_length: int = 256
    # RMS level analysis window in seconds. It is independent of the input sample
    # rate; None selects the backend default. The level track supports silence,
    # onset, attack, and re-articulation decisions.
    level_window_seconds: float | None = None
    silence_threshold_dbfs: float = -60.0
    # F0 estimator name remains explicit so the serialized contract identifies the
    # supported pitch path and can introduce any future estimator by name.
    estimator: PitchEstimatorName = "pyin"
    # Which pYIN implementation runs behind estimator="pyin". The native backend is
    # the default specialized implementation; librosa is an optional reference
    # backend used for equivalence checks and is not a default runtime dependency.
    pyin_backend: PyinBackendName = "native"
    # Beam width for the pYIN decode, in nats below the frame's best state. None keeps
    # the exact decode and is the default; a finite value is an approximation and can
    # change the result, so it remains opt-in.
    pyin_beam: float | None = None
    # Total-variation weight applied to the pitch track, in semitones, before notes
    # are decoded. Zero is the shipped behavior: no smoothing.
    #
    smoothing_tv_weight: float = 0.0
    # Width of a running median over the voiced pitch, in frames. Zero or one is
    # off. This provides a robust local alternative to total-variation smoothing.
    smoothing_median_frames: int = 0
    # Width of a pYIN pitch bin in semitones. Both supported backends honor it.
    pitch_resolution: float = 0.1
    # Normalize unvoiced observation mass against the default grid rather than the
    # selected grid, so pitch resolution does not change the voicing prior. Disabled
    # by default for compatibility with the shipped profile; native backend only.
    grid_independent_voicing: bool = False

    def __post_init__(self) -> None:
        _reject_boolean_numbers(
            (
                self.fmin_hz,
                self.fmax_hz,
                self.level_window_seconds,
                self.silence_threshold_dbfs,
                self.pyin_beam,
                self.smoothing_tv_weight,
                self.smoothing_median_frames,
                self.pitch_resolution,
            ),
            "pitch numeric fields must not be boolean",
        )
        values = (self.fmin_hz, self.fmax_hz, self.silence_threshold_dbfs)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("pitch thresholds must be finite")
        if self.fmin_hz <= 0 or self.fmax_hz <= self.fmin_hz:
            raise ValueError("pitch bounds must satisfy 0 < fmin_hz < fmax_hz")
        if (
            not _is_int(self.frame_length)
            or not _is_int(self.hop_length)
            or self.frame_length <= 0
            or self.hop_length <= 0
        ):
            raise ValueError("frame_length and hop_length must be positive")
        if self.estimator != "pyin":
            raise ValueError("estimator must be 'pyin'")
        if self.pyin_backend not in ("librosa", "native"):
            raise ValueError("pyin_backend must be 'librosa' or 'native'")
        if self.pyin_beam is not None and (
            not math.isfinite(self.pyin_beam) or self.pyin_beam <= 0
        ):
            raise ValueError("pyin_beam must be finite and positive (None = exact decode)")
        if self.pyin_beam is not None and self.pyin_backend != "native":
            raise ValueError("pyin_beam only applies to the native pyin backend")
        if not math.isfinite(self.smoothing_tv_weight) or self.smoothing_tv_weight < 0:
            raise ValueError("smoothing_tv_weight must be finite and non-negative")
        if (
            type(self.smoothing_median_frames) is not int
            or self.smoothing_median_frames < 0
            or (self.smoothing_median_frames > 1 and self.smoothing_median_frames % 2 == 0)
        ):
            raise ValueError("smoothing_median_frames must be 0, 1, or a positive odd integer")
        if not math.isfinite(self.pitch_resolution) or self.pitch_resolution <= 0:
            raise ValueError("pitch_resolution must be finite and positive")
        if self.level_window_seconds is not None and (
            not math.isfinite(self.level_window_seconds) or self.level_window_seconds <= 0
        ):
            raise ValueError(
                "level_window_seconds must be finite and positive (None = backend default)"
            )
        if type(self.grid_independent_voicing) is not bool:
            raise ValueError("grid_independent_voicing must be a boolean")
        if self.grid_independent_voicing and self.pyin_backend != "native":
            raise ValueError("grid_independent_voicing only applies to the native pyin backend")


@dataclass(frozen=True, slots=True)
class SegmentationConfig:
    offset_probability: float = 0.45
    minimum_duration_seconds: float = 0.05
    # Compatibility-only field from the removed hysteresis/onset/Viterbi segmenters.
    # The current note-HMM path intentionally does not read it; consolidation has its
    # separate `consolidation_max_gap_seconds` control. It remains serialized because
    # removing a field from the closed musicdna-analysis-v0 contract would require a
    # new contract name and schema.
    maximum_gap_seconds: float = 0.04
    silence_threshold_dbfs: float = -60.0
    tuning_reference_hz: float = 440.0
    # The v0 contract still carries this field for compatibility, while all supported
    # profiles use the note-HMM decoder.
    segmenter: SegmenterName = "note_hmm"
    # Onset evidence for the note-HMM decoder and consolidation.
    energy_onset_window_frames: int = 8
    onset_feature: OnsetFeatureMode = "energy"
    spectral_flux_onset_threshold: float = 0.30
    spectral_flux_onset_window_frames: int = 4
    spectral_flux_hop_length: int = 256
    # Post-segmentation coverage recovery, kept separate from decoder tuning.
    coverage_recovery: CoverageRecoveryMode = "raw_onset"
    raw_onset_recovery_min_duration_seconds: float = 0.04
    raw_onset_recovery_max_duration_seconds: float = 0.12
    raw_onset_recovery_context_seconds: float = 0.08
    raw_onset_recovery_min_voiced_frames: int = 3
    raw_onset_recovery_min_onset_strength: float = 0.20
    raw_onset_recovery_event_margin_seconds: float = 0.01
    raw_onset_recovery_hop_length: int = 256
    # Pitch validation for recovered notes: reject a recovery window whose voiced-frame
    # pitch spread (semitone standard deviation) exceeds the gate, and optionally estimate
    # the note pitch from the modal semitone bin instead of the plain median. None
    # disables the gate, which keeps pre-gate behavior (and stays JSON-serializable).
    raw_onset_recovery_max_pitch_spread_semitones: float | None = None
    raw_onset_recovery_pitch_method: RecoveryPitchMethod = "median"
    # Conservative post-pass for v2: merge adjacent same-pitch fragments only when the
    # boundary lacks a real pause or energy re-articulation.
    consolidate_notes: bool = True
    consolidation_max_gap_seconds: float = 0.03
    consolidation_same_pitch_semitones: float = 0.75
    consolidation_min_rearticulation_dbfs: float = 4.0
    # Global state model: emission costs are negative log probabilities, so this
    # parameter acts as an observation temperature.
    note_hmm_observation_trust: float = 1.0
    # Number of decoder states per semitone. Larger values increase transition-matrix
    # cost and are therefore opt-in.
    note_hmm_states_per_semitone: int = 1
    state_pitch_sigma_semitones: float = 0.75
    # Optional distance-dependent transition cost. None keeps a flat change penalty;
    # a finite value adds a Gaussian-style distance cost.
    state_change_distance_sigma: float | None = None
    state_change_penalty: float = 3.0
    state_silence_transition_penalty: float = 1.0
    state_voiced_silence_penalty: float = 4.0
    state_unvoiced_note_penalty: float = 1.5
    state_onset_reward: float = 0.15
    state_onset_window_frames: int = 4
    # note_hmm decoder (attack-state extension of the Viterbi decoder): the
    # attack phase tolerates transient pitch (sigma factor) and breathy starts
    # (lower unvoiced penalty); re-articulation lets a sustained note re-enter
    # its own attack state so repeated same-pitch notes can split.
    note_hmm_attack_sigma_factor: float = 2.0
    note_hmm_attack_unvoiced_penalty: float = 1.0
    note_hmm_attack_self_cost: float = 0.1
    note_hmm_rearticulation_penalty: float = 4.0
    # Move each decoded note start to the strongest onset evidence within the
    # configured window. Zero disables snapping.
    note_hmm_onset_snap_seconds: float = 0.0
    # End each note at the first frame whose level has fallen this far below the note's
    # plateau. Zero disables the post-pass.
    offset_release_drop_dbfs: float = 0.0
    # Which decoded frames define a note's pitch (note_hmm only). The decoder already
    # models ATTACK as a transient-pitch phase and widens its sigma accordingly, yet
    # "all" then feeds those same frames into the pitch median unweighted. "sustain"
    # keeps the decoder's own judgement by estimating pitch from SUSTAIN frames only,
    # falling back to all pitched frames for notes decoded without one.
    #
    # ``sustain`` is retained for compatibility; supported profiles currently use
    # ``all``.
    note_pitch_frames: NotePitchFrameMode = "all"
    # How the selected frames become one pitch. "median" is robust to octave outliers
    # but on a modulated tone it returns an arbitrary point of the sweep; "cents_mean"
    # averages in the log-frequency domain and therefore returns the centre of a vibrato
    # or glide, which is the pitch a listener names.
    #
    # Log-frequency averaging is useful for continuous pitch motion; median remains
    # the default because it is more robust to outliers.
    note_pitch_aggregate: NotePitchAggregateMode = "median"

    def __post_init__(self) -> None:
        values = (
            self.offset_probability,
            self.minimum_duration_seconds,
            self.maximum_gap_seconds,
            self.silence_threshold_dbfs,
            self.tuning_reference_hz,
            self.spectral_flux_onset_threshold,
            self.raw_onset_recovery_min_duration_seconds,
            self.raw_onset_recovery_max_duration_seconds,
            self.raw_onset_recovery_context_seconds,
            self.raw_onset_recovery_min_onset_strength,
            self.raw_onset_recovery_event_margin_seconds,
            self.consolidation_max_gap_seconds,
            self.consolidation_same_pitch_semitones,
            self.consolidation_min_rearticulation_dbfs,
            self.state_pitch_sigma_semitones,
            self.state_change_penalty,
            self.state_silence_transition_penalty,
            self.state_voiced_silence_penalty,
            self.state_unvoiced_note_penalty,
            self.state_onset_reward,
            self.note_hmm_attack_sigma_factor,
            self.note_hmm_attack_unvoiced_penalty,
            self.note_hmm_attack_self_cost,
            self.note_hmm_rearticulation_penalty,
            self.note_hmm_onset_snap_seconds,
            self.offset_release_drop_dbfs,
        )
        _reject_boolean_numbers(
            (
                *values,
                self.raw_onset_recovery_max_pitch_spread_semitones,
                self.state_change_distance_sigma,
                self.note_hmm_observation_trust,
                self.energy_onset_window_frames,
                self.spectral_flux_onset_window_frames,
                self.spectral_flux_hop_length,
                self.raw_onset_recovery_min_voiced_frames,
                self.raw_onset_recovery_hop_length,
                self.note_hmm_states_per_semitone,
                self.state_onset_window_frames,
            ),
            "segmentation numeric fields must not be boolean",
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("segmentation thresholds must be finite")
        if not 0 <= self.offset_probability <= 1:
            raise ValueError("offset_probability must be in [0, 1]")
        if self.maximum_gap_seconds < 0:
            raise ValueError("duration and gap thresholds must be non-negative")
        # Zero was accepted and then crashed the release stage: the segmenter can emit a
        # note of length zero, which NoteEvent rejects with "must have positive duration".
        # A minimum of zero is not a way to switch the filter off, so refuse it here
        # rather than fail three stages later.
        if self.minimum_duration_seconds <= 0:
            raise ValueError("minimum_duration_seconds must be positive")
        if self.tuning_reference_hz <= 0:
            raise ValueError("tuning_reference_hz must be positive")
        if self.segmenter != "note_hmm":
            raise ValueError("segmenter must be 'note_hmm'")
        if self.coverage_recovery not in ("off", "raw_onset"):
            raise ValueError("coverage_recovery must be 'off' or 'raw_onset'")
        if self.onset_feature not in ("energy", "spectral_flux", "energy_and_spectral_flux"):
            raise ValueError(
                "onset_feature must be 'energy', 'spectral_flux', or 'energy_and_spectral_flux'"
            )
        if self.spectral_flux_onset_threshold < 0:
            raise ValueError("spectral_flux_onset_threshold must be >= 0")
        if not _is_int(self.energy_onset_window_frames) or self.energy_onset_window_frames < 1:
            raise ValueError("energy_onset_window_frames must be >= 1")
        if (
            not _is_int(self.spectral_flux_onset_window_frames)
            or not _is_int(self.spectral_flux_hop_length)
            or self.spectral_flux_onset_window_frames < 1
            or self.spectral_flux_hop_length < 1
        ):
            raise ValueError("spectral-flux onset frames and hop length must be >= 1")
        if (
            self.raw_onset_recovery_min_duration_seconds < 0
            or self.raw_onset_recovery_context_seconds < 0
            or self.raw_onset_recovery_min_onset_strength < 0
            or self.raw_onset_recovery_event_margin_seconds < 0
        ):
            raise ValueError("raw-onset recovery thresholds must be non-negative")
        if (
            self.raw_onset_recovery_max_duration_seconds
            < self.raw_onset_recovery_min_duration_seconds
        ):
            raise ValueError("raw-onset recovery max duration must be >= min duration")
        if (
            not _is_int(self.raw_onset_recovery_min_voiced_frames)
            or self.raw_onset_recovery_min_voiced_frames < 1
        ):
            raise ValueError("raw-onset recovery min voiced frames must be >= 1")
        if (
            not _is_int(self.raw_onset_recovery_hop_length)
            or self.raw_onset_recovery_hop_length < 1
        ):
            raise ValueError("raw-onset recovery hop length must be >= 1")
        if self.raw_onset_recovery_max_pitch_spread_semitones is not None and (
            not math.isfinite(self.raw_onset_recovery_max_pitch_spread_semitones)
            or self.raw_onset_recovery_max_pitch_spread_semitones <= 0
        ):
            raise ValueError(
                "raw-onset recovery max pitch spread must be finite and positive"
                " (None disables the gate)"
            )
        if self.raw_onset_recovery_pitch_method not in ("median", "modal_median"):
            raise ValueError("raw_onset_recovery_pitch_method must be 'median' or 'modal_median'")
        if (
            self.consolidation_max_gap_seconds < 0
            or self.consolidation_same_pitch_semitones < 0
            or self.consolidation_min_rearticulation_dbfs < 0
        ):
            raise ValueError("consolidation thresholds must be non-negative")
        if (
            self.state_pitch_sigma_semitones <= 0
            or self.state_change_penalty < 0
            or self.state_silence_transition_penalty < 0
            or self.state_voiced_silence_penalty < 0
            or self.state_unvoiced_note_penalty < 0
            or self.state_onset_reward < 0
        ):
            raise ValueError("state-model costs must be positive/non-negative")
        if (
            not math.isfinite(self.note_hmm_observation_trust)
            or self.note_hmm_observation_trust < 0
        ):
            raise ValueError("note_hmm_observation_trust must be finite and non-negative")
        if (
            type(self.note_hmm_states_per_semitone) is not int
            or self.note_hmm_states_per_semitone < 1
            or self.note_hmm_states_per_semitone > 3
        ):
            raise ValueError("note_hmm_states_per_semitone must be an integer in [1, 3]")
        if self.state_change_distance_sigma is not None and (
            not math.isfinite(self.state_change_distance_sigma)
            or self.state_change_distance_sigma <= 0
        ):
            raise ValueError(
                "state_change_distance_sigma must be finite and positive (None keeps flat cost)"
            )
        if (
            self.note_hmm_attack_sigma_factor < 1
            or self.note_hmm_attack_unvoiced_penalty < 0
            or self.note_hmm_attack_self_cost < 0
            or self.note_hmm_rearticulation_penalty < 0
            or self.note_hmm_onset_snap_seconds < 0
        ):
            raise ValueError("note_hmm parameters must be non-negative (sigma factor >= 1)")
        if not math.isfinite(self.offset_release_drop_dbfs) or self.offset_release_drop_dbfs < 0:
            raise ValueError(
                "offset_release_drop_dbfs must be finite and non-negative (0 disables)"
            )
        if not _is_int(self.state_onset_window_frames) or self.state_onset_window_frames < 1:
            raise ValueError("state_onset_window_frames must be >= 1")
        if self.note_pitch_frames not in ("all", "sustain"):
            raise ValueError("note_pitch_frames must be 'all' or 'sustain'")
        if self.note_pitch_aggregate not in ("median", "cents_mean"):
            raise ValueError("note_pitch_aggregate must be 'median' or 'cents_mean'")
        if type(self.consolidate_notes) is not bool:
            raise ValueError("consolidate_notes must be a boolean")


def _profile_components(
    profile: AnalysisProfileName,
) -> tuple[LoaderConfig, PitchConfig, SegmentationConfig]:
    if profile in ("default", "singing"):
        # The default profile is intentionally equivalent to singing in 0.9.
        # Keep the implementation shared so the two profiles cannot drift apart.
        return (
            LoaderConfig(),
            PitchConfig(),
            SegmentationConfig(
                segmenter="note_hmm",
                note_hmm_onset_snap_seconds=0.0,
                note_hmm_attack_unvoiced_penalty=0.3,
                state_silence_transition_penalty=2.0,
                consolidation_max_gap_seconds=0.25,
            ),
        )
    if profile == "humming":
        # Humming profile: restrict the pitch range and use shorter, more local level
        # evidence plus the recovery and release passes suited to query recordings.
        return (
            LoaderConfig(),
            PitchConfig(
                fmax_hz=1_046.502,
                level_window_seconds=0.023,
            ),
            SegmentationConfig(
                segmenter="note_hmm",
                coverage_recovery="raw_onset",
                consolidate_notes=True,
                note_hmm_onset_snap_seconds=0.04,
                state_unvoiced_note_penalty=0.2,
                offset_release_drop_dbfs=8.0,
                note_pitch_aggregate="cents_mean",
                onset_feature="energy_and_spectral_flux",
                spectral_flux_onset_threshold=0.1,
                state_onset_reward=0.6,
            ),
        )
    raise ValueError("profile must be 'default', 'singing', or 'humming'")


def analysis_config_for_profile(profile: AnalysisProfileName) -> AnalysisConfig:
    """Return an immutable analysis config for a named use-case profile."""
    return AnalysisConfig(profile=profile)


@dataclass(frozen=True, slots=True, init=False)
class AnalysisConfig:
    profile: AnalysisProfileName
    loader: LoaderConfig
    pitch: PitchConfig
    segmentation: SegmentationConfig
    normalize_audio: bool
    normalization_target_dbfs: float
    normalization_reference_percentile: float

    def __init__(
        self,
        loader: LoaderConfig | None = None,
        pitch: PitchConfig | None = None,
        segmentation: SegmentationConfig | None = None,
        *,
        sample_rate: int | None = None,
        target_sample_rate: int | None = None,
        fmin_hz: float | None = None,
        fmax_hz: float | None = None,
        minimum_duration_seconds: float | None = None,
        min_note_duration_seconds: float | None = None,
        silence_threshold_dbfs: float | None = None,
        tuning_reference_hz: float | None = None,
        decoder_policy: DecoderPolicy | None = None,
        segmenter: SegmenterName | None = None,
        coverage_recovery: CoverageRecoveryMode | None = None,
        onset_feature: OnsetFeatureMode | None = None,
        spectral_flux_onset_threshold: float | None = None,
        normalize_audio: bool = False,
        normalization_target_dbfs: float = -18.0,
        normalization_reference_percentile: float = 99.0,
        profile: AnalysisProfileName = "default",
    ) -> None:
        if type(normalize_audio) is not bool:
            raise ValueError("normalize_audio must be a boolean")
        _reject_boolean_numbers(
            (normalization_target_dbfs, normalization_reference_percentile),
            "normalization numeric fields must not be boolean",
        )
        if profile not in (
            "default",
            "singing",
            "humming",
        ):
            raise ValueError("profile must be 'default', 'singing', or 'humming'")
        if not math.isfinite(normalization_target_dbfs) or normalization_target_dbfs > 0:
            raise ValueError("normalization_target_dbfs must be finite and non-positive")
        if (
            not math.isfinite(normalization_reference_percentile)
            or not 0 < normalization_reference_percentile <= 100
        ):
            raise ValueError("normalization_reference_percentile must be in (0, 100]")
        base_loader, base_pitch, base_segment = _profile_components(profile)
        old_loader = loader or base_loader
        old_pitch = pitch or base_pitch
        old_segment = segmentation or base_segment
        rate = sample_rate if sample_rate is not None else target_sample_rate
        minimum = (
            min_note_duration_seconds
            if min_note_duration_seconds is not None
            else minimum_duration_seconds
        )
        if silence_threshold_dbfs is not None:
            silence = silence_threshold_dbfs
        elif segmentation is not None:
            silence = old_segment.silence_threshold_dbfs
        else:
            silence = old_pitch.silence_threshold_dbfs
        new_loader = LoaderConfig(
            target_sample_rate=old_loader.target_sample_rate if rate is None else rate,
            decoder_policy=old_loader.decoder_policy if decoder_policy is None else decoder_policy,
            ffmpeg_timeout_seconds=old_loader.ffmpeg_timeout_seconds,
        )
        new_pitch = replace(
            old_pitch,
            # Same rule as segmentation below: only the knobs this constructor
            # explicitly exposes are overridden, and every other pitch field survives
            # via dataclasses.replace. Enumerating them by hand silently dropped any
            # newly added field, which is the bug this mirrors the fix for.
            fmin_hz=old_pitch.fmin_hz if fmin_hz is None else fmin_hz,
            fmax_hz=old_pitch.fmax_hz if fmax_hz is None else fmax_hz,
            silence_threshold_dbfs=silence,
        )
        new_segment = replace(
            old_segment,
            # Only the knobs this constructor explicitly exposes are overridden;
            # every other segmentation field survives via dataclasses.replace so
            # newly added fields can never be silently dropped again.
            minimum_duration_seconds=(
                old_segment.minimum_duration_seconds if minimum is None else minimum
            ),
            silence_threshold_dbfs=silence,
            tuning_reference_hz=(
                old_segment.tuning_reference_hz
                if tuning_reference_hz is None
                else tuning_reference_hz
            ),
            segmenter=old_segment.segmenter if segmenter is None else segmenter,
            coverage_recovery=(
                old_segment.coverage_recovery if coverage_recovery is None else coverage_recovery
            ),
            onset_feature=(old_segment.onset_feature if onset_feature is None else onset_feature),
            spectral_flux_onset_threshold=(
                old_segment.spectral_flux_onset_threshold
                if spectral_flux_onset_threshold is None
                else spectral_flux_onset_threshold
            ),
        )
        object.__setattr__(self, "loader", new_loader)
        object.__setattr__(self, "pitch", new_pitch)
        object.__setattr__(self, "segmentation", new_segment)
        object.__setattr__(self, "normalize_audio", bool(normalize_audio))
        object.__setattr__(self, "normalization_target_dbfs", float(normalization_target_dbfs))
        object.__setattr__(
            self,
            "normalization_reference_percentile",
            float(normalization_reference_percentile),
        )
        object.__setattr__(self, "profile", profile)

    @property
    def sample_rate(self) -> int | None:
        return self.loader.target_sample_rate

    @property
    def fmin_hz(self) -> float:
        return self.pitch.fmin_hz

    @property
    def fmax_hz(self) -> float:
        return self.pitch.fmax_hz

    @property
    def min_note_duration_seconds(self) -> float:
        return self.segmentation.minimum_duration_seconds

    @property
    def silence_threshold_dbfs(self) -> float:
        return self.pitch.silence_threshold_dbfs

    @property
    def target_sample_rate(self) -> int | None:
        return self.sample_rate

    @property
    def minimum_duration_seconds(self) -> float:
        return self.min_note_duration_seconds

    @property
    def tuning_reference_hz(self) -> float:
        return self.segmentation.tuning_reference_hz

    @property
    def decoder_policy(self) -> DecoderPolicy:
        return self.loader.decoder_policy

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Complete detector analysis envelope serialized as ``musicdna-analysis-v0``."""

    events: tuple[NoteEvent, ...]
    pitch_track: PitchTrack
    duration_seconds: float
    sample_rate: int
    config: AnalysisConfig
    source: SourceMetadata = field(default_factory=lambda: SourceMetadata("buffer"))
    backend: BackendMetadata = field(default_factory=lambda: BackendMetadata(DecoderBackend.MEMORY))
    diagnostics: Diagnostics = field(default_factory=Diagnostics)
    # The public analysis contract includes event durations in seconds and milliseconds.
    schema_version: str = field(default=ANALYSIS_RESULT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        if (
            _is_bool(self.duration_seconds)
            or not math.isfinite(self.duration_seconds)
            or self.duration_seconds < 0
        ):
            raise ValueError("duration_seconds must be finite and non-negative")
        if not _is_int(self.sample_rate) or self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        events = tuple(self.events)
        if any(event.end_seconds > self.duration_seconds for event in events):
            raise ValueError("note events must not extend beyond the audio duration")
        if any(
            event.start_seconds < previous.end_seconds
            and not math.isclose(
                event.start_seconds,
                previous.end_seconds,
                rel_tol=0.0,
                abs_tol=EVENT_ORDER_TOLERANCE_SECONDS,
            )
            for previous, event in pairwise(events)
        ):
            raise ValueError("note events must be ordered and non-overlapping")
        object.__setattr__(self, "events", events)
        if self.source.original_sample_rate == 0:
            object.__setattr__(
                self,
                "source",
                SourceMetadata(
                    self.source.kind,
                    self.source.format,
                    self.sample_rate,
                    self.sample_rate,
                    self.source.channels,
                    self.duration_seconds,
                    False,
                ),
            )

    @property
    def warnings(self) -> tuple[AnalysisWarning, ...]:
        return self.diagnostics.warnings

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "events": [asdict(event) for event in self.events],
            "pitch_track": {
                "times_seconds": _json_vector(self.pitch_track.times_seconds),
                "frequencies_hz": _json_vector(self.pitch_track.frequencies_hz),
                "voiced_probabilities": _json_vector(self.pitch_track.voiced_probabilities),
                "levels_dbfs": _json_vector(cast(FloatArray, self.pitch_track.levels_dbfs)),
                "onset_strengths": (
                    None
                    if self.pitch_track.onset_strengths is None
                    else _json_vector(self.pitch_track.onset_strengths)
                ),
            },
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "config": self.config.to_dict(),
            "source": asdict(self.source),
            "backend": asdict(self.backend),
            "diagnostics": {"warnings": [asdict(warning) for warning in self.warnings]},
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)


# Migration aliases for the initial foundation API.
AudioSource = SourceMetadata
BackendInfo = BackendMetadata
AnalysisDiagnostics = Diagnostics
