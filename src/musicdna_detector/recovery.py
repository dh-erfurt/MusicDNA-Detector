"""Optional post-segmentation recovery for raw-onset coverage gaps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np

from . import dsp
from .models import AudioBuffer, FloatArray, NoteEvent, PitchTrack, SegmentationConfig


@dataclass(frozen=True, slots=True)
class _OnsetCandidate:
    time_seconds: float
    strength: float


class RawOnsetCoverageRecoverer:
    """Recover short plausible notes from raw-audio onsets in uncovered gaps.

    This is intentionally not a segmenter. It runs after normal segmentation and only
    inserts mini-spans where existing events leave a coverage gap. A candidate must have
    raw onset evidence plus nearby audible, voiced pitch-track frames; this avoids globally
    lowering segment duration or confidence thresholds.
    """

    def __init__(self, config: SegmentationConfig | None = None) -> None:
        self.config = config or SegmentationConfig()

    def recover(
        self,
        audio: AudioBuffer,
        track: PitchTrack,
        events: tuple[NoteEvent, ...],
        *,
        raw_onset_envelope: FloatArray | None = None,
    ) -> tuple[NoteEvent, ...]:
        if self.config.coverage_recovery == "off":
            return events
        if self.config.coverage_recovery != "raw_onset":
            raise ValueError("unsupported coverage recovery mode")
        if audio.samples.size == 0 or len(track.times_seconds) == 0:
            return events

        recovered = list(sorted(events, key=lambda event: event.start_seconds))
        candidates = self._raw_onset_candidates(audio, raw_onset_envelope=raw_onset_envelope)
        frame_duration = _frame_duration(track, audio.duration_seconds)
        for candidate in candidates:
            note = self._build_recovered_note(
                track,
                recovered,
                candidate,
                frame_duration,
                audio.duration_seconds,
            )
            if note is not None:
                recovered.append(note)
                recovered.sort(key=lambda event: event.start_seconds)
        return tuple(recovered)

    def _raw_onset_candidates(
        self, audio: AudioBuffer, *, raw_onset_envelope: FloatArray | None = None
    ) -> tuple[_OnsetCandidate, ...]:
        if audio.samples.size == 0:
            return ()
        hop_length = self.config.raw_onset_recovery_hop_length
        normalized = raw_onset_envelope
        if normalized is None:
            normalized = dsp.normalized_onset_strength(
                np.asarray(audio.samples, dtype=np.float64), audio.sample_rate, hop_length
            )
        if normalized.size == 0:
            return ()
        frames = dsp.onset_detect(normalized, audio.sample_rate, hop_length)
        candidates: list[_OnsetCandidate] = []
        for frame in frames:
            if frame < 0 or frame >= normalized.size:
                continue
            strength = float(normalized[frame])
            if strength < self.config.raw_onset_recovery_min_onset_strength:
                continue
            candidates.append(
                _OnsetCandidate(
                    time_seconds=float(frame * hop_length / audio.sample_rate),
                    strength=strength,
                )
            )
        return tuple(sorted(candidates, key=lambda candidate: candidate.time_seconds))

    def _build_recovered_note(
        self,
        track: PitchTrack,
        events: list[NoteEvent],
        candidate: _OnsetCandidate,
        frame_duration: float,
        audio_duration: float,
    ) -> NoteEvent | None:
        config = self.config
        gap = _coverage_gap(
            events,
            candidate.time_seconds,
            audio_duration=audio_duration,
            margin=config.raw_onset_recovery_event_margin_seconds,
        )
        if gap is None:
            return None
        gap_start, gap_end = gap
        start = max(gap_start, candidate.time_seconds)
        max_end = min(gap_end, start + config.raw_onset_recovery_max_duration_seconds)
        if max_end - start < config.raw_onset_recovery_min_duration_seconds:
            return None

        levels = cast(FloatArray, track.levels_dbfs)
        times = track.times_seconds
        context_start = max(0.0, start - frame_duration)
        context_end = min(max_end, start + config.raw_onset_recovery_context_seconds)
        voiced = np.flatnonzero(
            (times >= context_start)
            & (times <= context_end)
            & np.isfinite(track.frequencies_hz)
            & (track.voiced_probabilities >= config.offset_probability)
            & np.isfinite(levels)
            & (levels >= config.silence_threshold_dbfs)
        )
        if voiced.size < config.raw_onset_recovery_min_voiced_frames:
            return None

        last_voiced_end = float(times[voiced[-1]] + frame_duration)
        end = min(
            max_end, max(start + config.raw_onset_recovery_min_duration_seconds, last_voiced_end)
        )
        if end - start < config.raw_onset_recovery_min_duration_seconds:
            return None
        frequencies = track.frequencies_hz[voiced]
        semitones = 12.0 * np.log2(frequencies / config.tuning_reference_hz)
        max_spread = config.raw_onset_recovery_max_pitch_spread_semitones
        if max_spread is not None and float(np.std(semitones)) > max_spread:
            return None
        if config.raw_onset_recovery_pitch_method == "modal_median":
            frequency = _modal_median_frequency(frequencies, semitones)
        else:
            frequency = float(np.median(frequencies))
        confidence = float(np.mean(track.voiced_probabilities[voiced]) * candidate.strength)
        confidence = min(1.0, max(0.0, confidence))
        return NoteEvent(
            start,
            end,
            frequency,
            confidence,
            tuning_reference_hz=config.tuning_reference_hz,
        )


def recover_coverage_gaps(
    audio: AudioBuffer,
    track: PitchTrack,
    events: tuple[NoteEvent, ...],
    config: SegmentationConfig,
    *,
    raw_onset_envelope: FloatArray | None = None,
) -> tuple[NoteEvent, ...]:
    """Apply the configured coverage recovery mode to segmented note events."""
    return RawOnsetCoverageRecoverer(config).recover(
        audio, track, events, raw_onset_envelope=raw_onset_envelope
    )


def _modal_median_frequency(frequencies: FloatArray, semitones: FloatArray) -> float:
    """Median frequency over the frames in the most populated semitone bin.

    Unlike the plain median, this cannot land between two clusters when the recovery
    window straddles a note boundary or contains stray octave-error frames.
    """
    bins = np.round(semitones)
    values, counts = np.unique(bins, return_counts=True)
    modal = values[int(np.argmax(counts))]
    in_bin = bins == modal
    return float(np.median(frequencies[in_bin]))


def _frame_duration(track: PitchTrack, audio_duration: float) -> float:
    if len(track.times_seconds) > 1:
        duration = float(np.median(np.diff(track.times_seconds)))
        if math.isfinite(duration) and duration > 0:
            return duration
    return audio_duration if audio_duration > 0 else 0.0


def _coverage_gap(
    events: list[NoteEvent],
    time_seconds: float,
    *,
    audio_duration: float,
    margin: float,
) -> tuple[float, float] | None:
    if not math.isfinite(time_seconds) or time_seconds < 0 or time_seconds >= audio_duration:
        return None
    gap_start = 0.0
    gap_end = audio_duration
    for event in events:
        if event.start_seconds - margin <= time_seconds <= event.end_seconds + margin:
            return None
        if event.end_seconds + margin < time_seconds:
            gap_start = max(gap_start, event.end_seconds + margin)
        elif event.start_seconds - margin > time_seconds:
            gap_end = min(gap_end, event.start_seconds - margin)
            break
    if gap_end <= gap_start:
        return None
    return gap_start, gap_end
