"""Segmenter selection.

The public v0 configuration names the note-HMM decoder. This wrapper keeps that
selection explicit and leaves a controlled extension point for a future decoder.
"""

from __future__ import annotations

from .models import SegmentationConfig
from .protocols import PitchSegmenter


def build_segmenter(config: SegmentationConfig | None = None) -> PitchSegmenter:
    """Construct the configured segmenter.

    ``SegmenterName`` currently has one member, so this is deliberately not a runtime
    strategy switch yet. The field stays as a serialized, named extension point so a
    future decoder can be introduced profile-scoped and against a named alternative,
    rather than by silently changing what this function returns.
    """
    from .note_hmm import NoteHmmSegmenter

    return NoteHmmSegmenter(config or SegmentationConfig())
