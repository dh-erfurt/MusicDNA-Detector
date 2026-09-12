"""Replaceable interfaces for the analysis pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .models import AudioBuffer, NoteEvent, PitchTrack


class AudioLoader(Protocol):
    def load(self, path: str | Path) -> AudioBuffer: ...


class PitchEstimator(Protocol):
    def estimate(self, audio: AudioBuffer) -> PitchTrack: ...


class PitchSegmenter(Protocol):
    def segment(self, track: PitchTrack, *, audio_duration: float) -> tuple[NoteEvent, ...]: ...
