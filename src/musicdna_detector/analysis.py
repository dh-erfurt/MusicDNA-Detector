"""Public orchestration APIs."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import dsp
from .audio import SoundFileLoader
from .consolidation import consolidate_note_events
from .models import (
    AnalysisConfig,
    AnalysisResult,
    AnalysisWarning,
    AudioBuffer,
    BackendMetadata,
    Diagnostics,
    PitchConfig,
    PitchTrack,
    SegmentationConfig,
    SourceMetadata,
)
from .pitch import build_pitch_estimator
from .protocols import AudioLoader, PitchEstimator, PitchSegmenter
from .recovery import recover_coverage_gaps
from .release import trim_note_offsets
from .segmentation import build_segmenter

_OCTAVE_NEIGHBOUR_MIN_CONFIDENCE = 0.80
_OCTAVE_OUTLIER_MAX_CONFIDENCE = 0.75
_OCTAVE_CONFIDENCE_MARGIN = 0.10
_OCTAVE_NEIGHBOUR_MIN_DURATION_SECONDS = 0.08
_OCTAVE_OUTLIER_MAX_DURATION_SECONDS = 0.12
_OCTAVE_OUTLIER_MAX_NEIGHBOUR_DURATION_RATIO = 0.50
_OCTAVE_TOLERANCE_SEMITONES = 0.50
_OCTAVE_CONTINUITY_IMPROVEMENT_SEMITONES = 10.0


def analyze(
    audio: AudioBuffer,
    config: AnalysisConfig | None = None,
    *,
    estimator: PitchEstimator | None = None,
    segmenter: PitchSegmenter | None = None,
) -> AnalysisResult:
    resolved = config or AnalysisConfig()
    audio = _normalize_audio(audio, resolved)
    pitch_estimator = estimator or build_pitch_estimator(resolved.pitch)
    recovery_envelope: NDArray[np.float64] | None = None
    recovery_hop = resolved.segmentation.raw_onset_recovery_hop_length
    should_prepare_onsets = (
        resolved.segmentation.coverage_recovery == "raw_onset" and audio.samples.size > 0
    )
    # The raw-onset FFT depends only on the immutable audio buffer.  Run it beside the
    # built-in pitch estimator, then reuse the result for spectral-flux attachment when
    # both consumers use the same hop.  Custom estimators stay sequential because their
    # protocol does not promise that they leave the supplied buffer untouched.
    if should_prepare_onsets and estimator is None:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="musicdna-front-end") as executor:
            onset_future = executor.submit(
                dsp.normalized_onset_strength,
                np.asarray(audio.samples, dtype=np.float64),
                audio.sample_rate,
                recovery_hop,
            )
            track = pitch_estimator.estimate(audio)
            recovery_envelope = onset_future.result()
    else:
        track = pitch_estimator.estimate(audio)
        if should_prepare_onsets:
            recovery_envelope = dsp.normalized_onset_strength(
                np.asarray(audio.samples, dtype=np.float64), audio.sample_rate, recovery_hop
            )
    track = _smooth_pitch_track(track, resolved.pitch)
    track = _attach_spectral_flux_onsets(
        audio,
        track,
        resolved.segmentation,
        normalized_envelope=recovery_envelope
        if recovery_hop == resolved.segmentation.spectral_flux_hop_length
        else None,
    )
    effective_segmentation = resolved.segmentation
    effective_config = AnalysisConfig(
        loader=resolved.loader,
        pitch=resolved.pitch,
        segmentation=effective_segmentation,
        normalize_audio=resolved.normalize_audio,
        normalization_target_dbfs=resolved.normalization_target_dbfs,
        normalization_reference_percentile=resolved.normalization_reference_percentile,
        profile=resolved.profile,
    )
    pitch_segmenter = segmenter or build_segmenter(effective_segmentation)
    events = pitch_segmenter.segment(track, audio_duration=audio.duration_seconds)
    events = recover_coverage_gaps(
        audio,
        track,
        events,
        effective_segmentation,
        raw_onset_envelope=recovery_envelope,
    )
    events = consolidate_note_events(track, events, effective_segmentation)
    # Apply this after consolidation: the release rule changes note ends, while
    # consolidation decides merges from the original inter-note gaps.
    events = trim_note_offsets(track, events, effective_segmentation)
    warnings: list[AnalysisWarning] = []
    if not events:
        warnings.append(
            AnalysisWarning("no_notes", "No note events met the configured thresholds.")
        )
    backend = BackendMetadata(
        audio.decoder_backend,
        decoder_policy=audio.decoder_policy,
        pitch_estimator=type(pitch_estimator).__name__,
        segmenter=type(pitch_segmenter).__name__,
        ffmpeg_used=audio.decoder_backend.value == "ffmpeg",
    )
    diagnostics = Diagnostics(tuple(warnings))
    return AnalysisResult(
        events,
        track,
        audio.duration_seconds,
        audio.sample_rate,
        effective_config,
        audio.source,
        backend,
        diagnostics,
    )


def _normalize_audio(audio: AudioBuffer, config: AnalysisConfig) -> AudioBuffer:
    """Optionally scale the audio so a robust amplitude reference hits a target dBFS.

    The reference is a high percentile of the absolute samples, not the peak. A single
    loud transient -- a mic bump, a plosive, a desk knock, all routine in browser
    recordings -- sets the peak of an otherwise quiet clip, so peak normalization
    scales the actual voice *further down* and the dBFS-driven segmentation then
    discards it. Percentile 100 reproduces exact peak normalization for callers that
    want it. Disabled by default; empty, silent, or non-finite input stays safe.
    """
    if not config.normalize_audio or audio.samples.size == 0:
        return audio
    samples = np.asarray(audio.samples, dtype=np.float64)
    finite = np.isfinite(samples)
    if not np.any(finite):
        return AudioBuffer(
            np.zeros_like(samples),
            audio.sample_rate,
            audio.source,
            audio.decoder_backend,
            audio.decoder_policy,
        )
    clean = np.where(finite, samples, 0.0)
    reference = float(np.percentile(np.abs(clean), config.normalization_reference_percentile))
    if not math.isfinite(reference) or reference <= 0.0:
        # A mostly-silent clip can have a zero percentile while still carrying signal;
        # fall back to the peak so such input is normalized instead of passed through.
        reference = float(np.max(np.abs(clean)))
    if not math.isfinite(reference) or reference <= 0.0:
        return AudioBuffer(
            clean, audio.sample_rate, audio.source, audio.decoder_backend, audio.decoder_policy
        )
    target = math.pow(10.0, config.normalization_target_dbfs / 20.0)
    scale = target / reference
    return AudioBuffer(
        clean * scale, audio.sample_rate, audio.source, audio.decoder_backend, audio.decoder_policy
    )


def _smooth_pitch_track(track: PitchTrack, config: PitchConfig) -> PitchTrack:
    """Total-variation denoising of the voiced pitch, in semitones.

    Semitones rather than hertz because the weight then means the same thing high
    and low, and because that is the space the note decoder compares in.
    """

    weight = float(getattr(config, "smoothing_tv_weight", 0.0) or 0.0)
    median_width = int(getattr(config, "smoothing_median_frames", 0) or 0)
    if weight <= 0 and median_width <= 1:
        return track
    from .smoothing import denoise_voiced_runs, median_filter_voiced

    frequencies = np.asarray(track.frequencies_hz, dtype=np.float64)
    voiced = np.isfinite(frequencies) & (frequencies > 0)
    if not np.any(voiced):
        return track
    semitones = np.full(frequencies.shape, np.nan, dtype=np.float64)
    semitones[voiced] = 69.0 + 12.0 * np.log2(frequencies[voiced] / 440.0)
    filled = np.where(voiced, semitones, 0.0)
    smoothed = median_filter_voiced(filled, voiced, median_width)
    smoothed = denoise_voiced_runs(smoothed, voiced, weight)
    updated = frequencies.copy()
    updated[voiced] = 440.0 * np.power(2.0, (smoothed[voiced] - 69.0) / 12.0)
    return replace(track, frequencies_hz=updated)


def _attach_spectral_flux_onsets(
    audio: AudioBuffer,
    track: PitchTrack,
    config: SegmentationConfig,
    *,
    normalized_envelope: NDArray[np.float64] | None = None,
) -> PitchTrack:
    if config.onset_feature not in (
        "spectral_flux",
        "energy_and_spectral_flux",
    ):
        return track
    if audio.samples.size == 0 or len(track.times_seconds) == 0:
        return track
    hop_length = config.spectral_flux_hop_length
    envelope = normalized_envelope
    if envelope is None:
        envelope = dsp.normalized_onset_strength(
            np.asarray(audio.samples, dtype=np.float64), audio.sample_rate, hop_length
        )
    if envelope.size == 0:
        return track
    envelope_times = np.arange(envelope.size, dtype=np.float64) * hop_length / audio.sample_rate
    strengths = np.interp(track.times_seconds, envelope_times, envelope, left=0.0, right=0.0)
    # dataclasses.replace, not a positional rebuild: enumerating the fields by hand is
    # how the candidate arrays went missing on exactly the profiles that use this path,
    # and how AnalysisConfig lost fields before that. Anything added to PitchTrack now
    # survives here without this function being touched.
    return replace(track, onset_strengths=strengths)


def analyze_file(
    path: str | Path,
    config: AnalysisConfig | None = None,
    *,
    loader: AudioLoader | None = None,
) -> AnalysisResult:
    resolved = config or AnalysisConfig()
    audio_loader = loader or SoundFileLoader(resolved.loader)
    return analyze(audio_loader.load(path), resolved)


def analyze_audio(
    source: str | Path | AudioBuffer | NDArray[Any],
    config: AnalysisConfig | None = None,
    *,
    sample_rate: int | None = None,
    loader: AudioLoader | None = None,
) -> AnalysisResult:
    """Analyze a path, explicit buffer, or raw mono array with a mandatory sample rate."""
    if isinstance(source, AudioBuffer):
        if sample_rate is not None:
            raise TypeError("sample_rate must not be supplied with AudioBuffer")
        return analyze(source, config)
    if isinstance(source, (str, Path)):
        if sample_rate is not None:
            raise TypeError("sample_rate is only valid for ndarray input")
        return analyze_file(source, config, loader=loader)
    if isinstance(source, np.ndarray):
        if sample_rate is None:
            raise TypeError(
                "sample_rate is required for ndarray input; alternatively wrap samples in "
                "AudioBuffer(samples, sample_rate)"
            )
        samples = np.asarray(source, dtype=np.float64)
        if config is not None and config.normalize_audio:
            samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
        audio = AudioBuffer(
            samples,
            sample_rate,
            source=SourceMetadata("array"),
        )
        return analyze(audio, config)
    raise TypeError("source must be a path, AudioBuffer, or one-dimensional NumPy ndarray")
