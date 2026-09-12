"""Versioned machine-readable contracts for serialized detector results."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from importlib.resources import files
from typing import Any, cast

ANALYSIS_RESULT_SCHEMA_VERSION = "musicdna-analysis-v0"
_SCHEMA_FILENAME = "musicdna-analysis-v0.schema.json"
# Event boundaries are computed on a sample/frame grid, so adjacent events can
# differ by a few ulps after independent floating-point operations. That is not
# a semantic overlap and must be treated consistently by the object and JSON APIs.
EVENT_ORDER_TOLERANCE_SECONDS = 1e-12


def load_analysis_result_schema() -> dict[str, Any]:
    """Load the ``musicdna-analysis-v0`` schema for complete detector results."""
    resource = files("musicdna_detector").joinpath("schemas").joinpath(_SCHEMA_FILENAME)
    with resource.open("r", encoding="utf-8") as handle:
        return cast(dict[str, Any], json.load(handle))


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return cast(Mapping[str, Any], value)


def _array(value: object, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array")
    return value


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be a finite number")
    return result


def validate_analysis_result_payload(payload: Mapping[str, Any]) -> None:
    """Validate ``musicdna-analysis-v0`` invariants JSON Schema cannot express.

    Callers should first validate structure and field types against
    :func:`load_analysis_result_schema`, then call this function for relationships
    between fields and array elements. A ``ValueError`` identifies the first
    semantic violation.
    """
    root = _mapping(payload, "payload")
    if root.get("schema_version") != ANALYSIS_RESULT_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {ANALYSIS_RESULT_SCHEMA_VERSION!r} for this validator"
        )

    duration = _number(root.get("duration_seconds"), "duration_seconds")
    events = _array(root.get("events"), "events")
    previous_end = 0.0
    for index, raw_event in enumerate(events):
        path = f"events[{index}]"
        event = _mapping(raw_event, path)
        start = _number(event.get("start_seconds"), f"{path}.start_seconds")
        end = _number(event.get("end_seconds"), f"{path}.end_seconds")
        event_duration = _number(event.get("duration_seconds"), f"{path}.duration_seconds")
        duration_ms = _number(event.get("duration_ms"), f"{path}.duration_ms")
        if start < 0 or end <= start:
            raise ValueError(f"{path} must have a non-negative start and positive duration")
        if not math.isclose(event_duration, end - start):
            raise ValueError(f"{path}.duration_seconds must equal end_seconds - start_seconds")
        if not math.isclose(duration_ms, event_duration * 1000.0):
            raise ValueError(f"{path}.duration_ms must equal duration_seconds * 1000")
        if (
            index > 0
            and start < previous_end
            and not math.isclose(
                start, previous_end, rel_tol=0.0, abs_tol=EVENT_ORDER_TOLERANCE_SECONDS
            )
        ):
            raise ValueError("events must be ordered and non-overlapping")
        if end > duration:
            raise ValueError(f"{path}.end_seconds must not exceed duration_seconds")
        previous_end = end

    track = _mapping(root.get("pitch_track"), "pitch_track")
    arrays = {
        name: _array(track.get(name), f"pitch_track.{name}")
        for name in (
            "times_seconds",
            "frequencies_hz",
            "voiced_probabilities",
            "levels_dbfs",
        )
    }
    onset_strengths = track.get("onset_strengths")
    if onset_strengths is not None:
        arrays["onset_strengths"] = _array(onset_strengths, "pitch_track.onset_strengths")
    lengths = {len(values) for values in arrays.values()}
    if len(lengths) > 1:
        raise ValueError("pitch-track arrays must have equal length")

    previous_time = 0.0
    for index, value in enumerate(arrays["times_seconds"]):
        time = _number(value, f"pitch_track.times_seconds[{index}]")
        if time < 0 or (index > 0 and time < previous_time):
            raise ValueError("pitch_track.times_seconds must be non-negative and ordered")
        previous_time = time
