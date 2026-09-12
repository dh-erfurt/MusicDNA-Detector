"""MusicDNA HMM note decoder with per-pitch attack states.

Extends the experimental REST/MIDI Viterbi decoder with ideas from Mauch et al.'s
published note-transcription model (TENOR'15). This is a MusicDNA-specific
decoder, not an implementation of pYIN-Notes or its GPL Vamp plugin:

1. **Attack states**: every note must be entered through ATTACK(m); a
   SUSTAIN(m) -> ATTACK(m) transition models the re-articulation of the same
   pitch, which the plain REST/MIDI decoder can never split.
2. **Soft emissions**: squared cent distance under a per-state sigma (widened
   during attack, where the pitch track is transient), voicing probability as
   support evidence, RMS level for REST.
3. **Onset-evidence-modulated transitions**: energy rise and (when attached)
   spectral-flux onset strengths reduce the cost of entering an attack state,
   so boundaries snap to acoustic onsets.

The global REST state, onset-modulated costs, same-pitch re-articulation rule and
boundary snapping are MusicDNA-specific choices. Repository history and the current
implementation indicate that no code from the GPL Vamp plugin was used.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
from numpy.typing import NDArray

from .mapping import frequency_to_midi_float, round_half_up
from .models import FloatArray, NoteEvent, PitchTrack, SegmentationConfig

_REST = 0
_ATTACK = 1
_SUSTAIN = 2


class NoteHmmSegmenter:
    """Whole-track Viterbi decoder over REST / ATTACK(m) / SUSTAIN(m) states."""

    def __init__(self, config: SegmentationConfig | None = None) -> None:
        self.config = config or SegmentationConfig()

    def segment(self, track: PitchTrack, *, audio_duration: float) -> tuple[NoteEvent, ...]:
        if not math.isfinite(audio_duration) or audio_duration < 0:
            raise ValueError("audio_duration must be finite and non-negative")
        if len(track.times_seconds) == 0:
            return ()
        midis = self._candidate_midis(track)
        if len(midis) == 0:
            return ()
        frame_duration = _frame_duration(track, audio_duration)
        phases, pitches = self._state_space(midis)
        onset_rewards = self._onset_rewards(track)
        path = self._decode(track, phases, pitches, onset_rewards)
        events = self._events_from_path(
            track, phases, pitches, path, frame_duration, audio_duration
        )
        return self._snap_onsets(events, track, onset_rewards, frame_duration)

    def _candidate_midis(self, track: PitchTrack) -> NDArray[np.float64]:
        config = self.config
        voiced = (
            np.isfinite(track.frequencies_hz)
            & (track.voiced_probabilities >= config.offset_probability)
            & (cast(FloatArray, track.levels_dbfs) >= config.silence_threshold_dbfs)
        )
        if not np.any(voiced):
            return np.array([], dtype=np.float64)
        per_semitone = max(1, int(config.note_hmm_states_per_semitone))
        if per_semitone == 1:
            midis = {
                round_half_up(frequency_to_midi_float(float(frequency), config.tuning_reference_hz))
                for frequency in track.frequencies_hz[voiced]
            }
            expanded = set(midis)
            for midi in midis:
                expanded.add(midi - 1)
                expanded.add(midi + 1)
            return np.array(sorted(expanded), dtype=np.float64)
        # A finer grid: snap each voiced frame to the nearest 1/k semitone, then pad
        # by a whole semitone on both sides as the integer path does.
        steps = {
            round(
                frequency_to_midi_float(float(frequency), config.tuning_reference_hz) * per_semitone
            )
            for frequency in track.frequencies_hz[voiced]
        }
        expanded = set(steps)
        for step in steps:
            for offset in range(1, per_semitone + 1):
                expanded.add(step - offset)
                expanded.add(step + offset)
        return np.array(sorted(expanded), dtype=np.float64) / per_semitone

    def _state_space(
        self, midis: NDArray[np.float64]
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Return (phase, midi) arrays; state 0 is REST."""
        count = len(midis)
        phases = np.concatenate(
            [
                np.array([_REST], dtype=np.int64),
                np.full(count, _ATTACK, dtype=np.int64),
                np.full(count, _SUSTAIN, dtype=np.int64),
            ]
        )
        pitches = np.concatenate([np.array([0.0], dtype=np.float64), midis, midis])
        return phases, pitches

    def _decode(
        self,
        track: PitchTrack,
        phases: NDArray[np.int64],
        pitches: NDArray[np.float64],
        onset_rewards: NDArray[np.float64],
    ) -> NDArray[np.int64]:
        emissions = self._emission_costs(track, phases, pitches)
        transitions = self._transition_template(phases, pitches)
        attack_mask = (phases == _ATTACK).astype(np.float64)[np.newaxis, :]
        frame_count, state_count = emissions.shape
        costs = np.empty((frame_count, state_count), dtype=np.float64)
        back = np.zeros((frame_count, state_count), dtype=np.int64)
        costs[0] = emissions[0]
        for frame in range(1, frame_count):
            # Attack entries may become net-negative under strong onset evidence:
            # that is the mechanism that makes a same-pitch re-articulation split
            # cheaper than staying in sustain. Costs stay acyclic per frame, so
            # negative edges are safe for Viterbi.
            effective = transitions - onset_rewards[frame] * attack_mask
            candidates = costs[frame - 1][:, np.newaxis] + effective
            back[frame] = np.argmin(candidates, axis=0)
            costs[frame] = emissions[frame] + np.min(candidates, axis=0)
        path = np.empty(frame_count, dtype=np.int64)
        path[-1] = int(np.argmin(costs[-1]))
        for frame in range(frame_count - 1, 0, -1):
            path[frame - 1] = back[frame, path[frame]]
        return path

    def _emission_costs(
        self,
        track: PitchTrack,
        phases: NDArray[np.int64],
        pitches: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        config = self.config
        frame_count = len(track.times_seconds)
        levels = cast(FloatArray, track.levels_dbfs)
        finite_pitch = np.isfinite(track.frequencies_hz)
        audible = np.isfinite(levels) & (levels >= config.silence_threshold_dbfs)
        confident = track.voiced_probabilities >= config.offset_probability
        midi_values = np.full(frame_count, np.nan, dtype=np.float64)
        for index, frequency in enumerate(track.frequencies_hz):
            if math.isfinite(float(frequency)) and frequency > 0:
                midi_values[index] = frequency_to_midi_float(
                    float(frequency), config.tuning_reference_hz
                )

        emissions = np.empty((frame_count, len(phases)), dtype=np.float64)
        rest_cost = np.where(
            audible & finite_pitch & confident,
            config.state_voiced_silence_penalty * track.voiced_probabilities,
            0.0,
        )
        support_cost = np.where(
            audible & confident,
            1.0 - track.voiced_probabilities,
            config.state_unvoiced_note_penalty,
        )
        attack_support_cost = np.where(
            audible & confident,
            1.0 - track.voiced_probabilities,
            config.note_hmm_attack_unvoiced_penalty,
        )
        for state_index, (phase, midi) in enumerate(zip(phases, pitches, strict=True)):
            if phase == _REST:
                emissions[:, state_index] = rest_cost
                continue
            sigma = config.state_pitch_sigma_semitones
            if phase == _ATTACK:
                sigma *= config.note_hmm_attack_sigma_factor
            pitch_error = np.abs(midi_values - float(midi))
            pitch_cost = np.where(
                finite_pitch,
                (pitch_error / sigma) ** 2,
                config.state_unvoiced_note_penalty
                if phase == _SUSTAIN
                else config.note_hmm_attack_unvoiced_penalty,
            )
            emissions[:, state_index] = pitch_cost + (
                attack_support_cost if phase == _ATTACK else support_cost
            )
        # Costs are negative log probabilities, so a factor here is an exponent on
        # the probability: the observation temperature. Below 1 the frames argue more
        # quietly and the transition model decides more of the path.
        if config.note_hmm_observation_trust != 1.0:
            emissions *= config.note_hmm_observation_trust
        return emissions

    def _transition_template(
        self,
        phases: NDArray[np.int64],
        pitches: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        config = self.config
        infinity = np.inf
        previous_phase = phases[:, np.newaxis]
        previous_pitch = pitches[:, np.newaxis]
        current_phase = phases[np.newaxis, :]
        current_pitch = pitches[np.newaxis, :]
        same_pitch = previous_pitch == current_pitch

        costs = cast(
            NDArray[np.float64],
            np.full((len(phases), len(phases)), infinity, dtype=np.float64),
        )
        # REST self-loop and note entry.
        costs = cast(
            NDArray[np.float64],
            np.where((previous_phase == _REST) & (current_phase == _REST), 0.0, costs),
        )
        costs = cast(
            NDArray[np.float64],
            np.where(
                (previous_phase == _REST) & (current_phase == _ATTACK),
                config.state_silence_transition_penalty,
                costs,
            ),
        )
        # Attack resolves into its own sustain and may linger briefly.
        costs = cast(
            NDArray[np.float64],
            np.where(
                (previous_phase == _ATTACK) & (current_phase == _SUSTAIN) & same_pitch,
                0.0,
                costs,
            ),
        )
        costs = cast(
            NDArray[np.float64],
            np.where(
                (previous_phase == _ATTACK) & (current_phase == _ATTACK) & same_pitch,
                config.note_hmm_attack_self_cost,
                costs,
            ),
        )
        # Sustain continues, ends, changes pitch, or re-articulates the same pitch.
        costs = cast(
            NDArray[np.float64],
            np.where(
                (previous_phase == _SUSTAIN) & (current_phase == _SUSTAIN) & same_pitch,
                0.0,
                costs,
            ),
        )
        costs = cast(
            NDArray[np.float64],
            np.where(
                (previous_phase == _SUSTAIN) & (current_phase == _REST),
                config.state_silence_transition_penalty,
                costs,
            ),
        )
        # Scalar by default and an array once distance weighting is enabled.
        change_cost: float | NDArray[np.float64] = float(config.state_change_penalty)
        distance_sigma = config.state_change_distance_sigma
        if distance_sigma is not None and math.isfinite(distance_sigma) and distance_sigma > 0:
            # Negative log of a Gaussian over the interval, up to a constant: a
            # leap is less likely than a step, which is how melodies behave.
            distance = np.abs(current_pitch - previous_pitch).astype(np.float64)
            change_cost = change_cost + (distance / distance_sigma) ** 2 / 2.0
        costs = cast(
            NDArray[np.float64],
            np.where(
                (previous_phase == _SUSTAIN) & (current_phase == _ATTACK) & ~same_pitch,
                change_cost,
                costs,
            ),
        )
        costs = cast(
            NDArray[np.float64],
            np.where(
                (previous_phase == _SUSTAIN) & (current_phase == _ATTACK) & same_pitch,
                config.note_hmm_rearticulation_penalty,
                costs,
            ),
        )
        return costs

    def _onset_rewards(self, track: PitchTrack) -> NDArray[np.float64]:
        """Energy-rise rewards, boosted by spectral-flux onsets when attached."""
        config = self.config
        levels = cast(FloatArray, track.levels_dbfs)
        rewards = np.zeros(len(levels), dtype=np.float64)
        window = config.state_onset_window_frames
        for frame in range(1, len(levels)):
            local = levels[frame]
            if not math.isfinite(float(local)):
                continue
            before = levels[max(0, frame - window) : frame]
            before = before[np.isfinite(before)]
            if before.size == 0:
                continue
            rise = float(local) - float(np.median(before))
            if rise > 0:
                rewards[frame] = min(config.state_onset_reward, rise / 12.0)
        onset_strengths = getattr(track, "onset_strengths", None)
        if onset_strengths is not None and len(onset_strengths) == len(rewards):
            flux = np.asarray(onset_strengths, dtype=np.float64)
            flux_reward = np.where(
                flux >= config.spectral_flux_onset_threshold,
                config.state_onset_reward * np.clip(flux, 0.0, None),
                0.0,
            )
            rewards = np.maximum(rewards, flux_reward)
        return rewards

    def _events_from_path(
        self,
        track: PitchTrack,
        phases: NDArray[np.int64],
        pitches: NDArray[np.float64],
        path: NDArray[np.int64],
        frame_duration: float,
        audio_duration: float,
    ) -> tuple[NoteEvent, ...]:
        config = self.config
        events: list[NoteEvent] = []
        frame_phase = phases[path]
        sustained = frame_phase == _SUSTAIN
        note_open = False
        start_frame = 0
        for frame in range(len(path)):
            is_note = frame_phase[frame] != _REST
            starts_attack = frame_phase[frame] == _ATTACK and (
                not note_open or frame_phase[frame - 1] != _ATTACK
            )
            if note_open and (not is_note or starts_attack):
                events.extend(
                    self._build_note(
                        track, start_frame, frame - 1, frame_duration, audio_duration, sustained
                    )
                )
                note_open = False
            if is_note and not note_open:
                note_open = True
                start_frame = frame
        if note_open:
            events.extend(
                self._build_note(
                    track, start_frame, len(path) - 1, frame_duration, audio_duration, sustained
                )
            )
        return tuple(
            event for event in events if event.duration_seconds >= config.minimum_duration_seconds
        )

    def _build_note(
        self,
        track: PitchTrack,
        lo: int,
        hi: int,
        frame_duration: float,
        audio_duration: float,
        sustained: NDArray[np.bool_],
    ) -> list[NoteEvent]:
        config = self.config
        indices = np.arange(lo, hi + 1)
        pitched = indices[np.isfinite(track.frequencies_hz[indices])]
        begin = float(track.times_seconds[lo])
        end = min(audio_duration, float(track.times_seconds[hi] + frame_duration))
        if not len(pitched) or begin >= end:
            return []
        pitch_frames = pitched
        if config.note_pitch_frames == "sustain":
            # The attack is where the decoder itself expects the pitch track to be
            # transient. Keep the note's extent, but let the sustain define its pitch --
            # unless the note was decoded without one, where all frames is all there is.
            core = pitched[sustained[pitched]]
            if core.size:
                pitch_frames = core
        frequencies = track.frequencies_hz[pitch_frames]
        if config.note_pitch_aggregate == "cents_mean":
            # Average in the log-frequency domain: on a vibrato or a glide the median
            # returns an arbitrary point of the sweep, the log mean returns its centre --
            # the pitch a listener names. Frequencies are finite and positive here.
            frequency = float(2.0 ** np.mean(np.log2(frequencies)))
        else:
            frequency = float(np.median(frequencies))
        confidence = float(np.mean(track.voiced_probabilities[pitched]))
        return [
            NoteEvent(
                begin,
                end,
                frequency,
                min(1.0, max(0.0, confidence)),
                tuning_reference_hz=config.tuning_reference_hz,
            )
        ]

    def _snap_onsets(
        self,
        events: tuple[NoteEvent, ...],
        track: PitchTrack,
        onset_rewards: NDArray[np.float64],
        frame_duration: float,
    ) -> tuple[NoteEvent, ...]:
        """Snap note starts to the strongest onset evidence near the boundary.

        The pYIN track turns voiced systematically late on sung onsets, so the
        decoded boundary trails the acoustic onset. Each start may move to the
        highest-evidence frame within [start - snap, start + snap/4], never
        crossing the previous note's end.
        """
        snap = self.config.note_hmm_onset_snap_seconds
        if snap <= 0 or not events or frame_duration <= 0:
            return events
        times = track.times_seconds
        snapped: list[NoteEvent] = []
        previous_end = 0.0
        for event in events:
            lo_time = max(previous_end, event.start_seconds - snap)
            hi_time = min(event.end_seconds, event.start_seconds + snap / 4)
            lo = int(np.searchsorted(times, lo_time, side="left"))
            hi = int(np.searchsorted(times, hi_time, side="right"))
            start = event.start_seconds
            if hi > lo:
                window = onset_rewards[lo:hi]
                best = int(np.argmax(window))
                if window[best] > 0:
                    start = float(times[lo + best])
            if event.end_seconds - start >= self.config.minimum_duration_seconds:
                snapped.append(
                    NoteEvent(
                        start,
                        event.end_seconds,
                        event.frequency_hz,
                        event.confidence,
                        tuning_reference_hz=self.config.tuning_reference_hz,
                    )
                )
            else:
                snapped.append(event)
            previous_end = snapped[-1].end_seconds
        return tuple(snapped)


def _frame_duration(track: PitchTrack, audio_duration: float) -> float:
    if len(track.times_seconds) > 1:
        duration = float(np.median(np.diff(track.times_seconds)))
        if duration > 0:
            return duration
    if len(track.times_seconds) == 1 and audio_duration > 0:
        return audio_duration
    return 0.01
