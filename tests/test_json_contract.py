import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator, ValidationError

from musicdna_detector import (
    ANALYSIS_RESULT_SCHEMA_VERSION,
    AnalysisConfig,
    AnalysisResult,
    NoteEvent,
    PitchTrack,
    load_analysis_result_schema,
    validate_analysis_result_payload,
)

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "musicdna-analysis-v0.json"


def _result() -> AnalysisResult:
    return AnalysisResult(
        (NoteEvent(0.0, 0.25, 440.0, 0.9),),
        PitchTrack(
            np.array([0.0, 0.1]),
            np.array([440.0, np.nan]),
            np.array([0.9, 0.0]),
            np.array([-12.0, -np.inf]),
        ),
        0.25,
        22_050,
        AnalysisConfig(),
    )


def test_musicdna_analysis_v0_matches_packaged_schema_and_golden_payload() -> None:
    schema = load_analysis_result_schema()
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    payload = json.loads(_result().to_json())

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(golden)
    Draft202012Validator(schema).validate(payload)
    validate_analysis_result_payload(payload)
    assert payload == golden
    assert schema["properties"]["schema_version"]["const"] == (ANALYSIS_RESULT_SCHEMA_VERSION)
    assert schema["$id"].endswith("/musicdna-analysis-v0.schema.json")
    assert "musicdna-analysis-v0" in schema["title"]
    assert "analysis envelope" in schema["description"]
    assert "musicdna-events-v0" in schema["$comment"]


def test_musicdna_analysis_v0_schema_is_closed_and_named() -> None:
    schema = load_analysis_result_schema()
    validator = Draft202012Validator(schema)
    payload = json.loads(_result().to_json())

    with pytest.raises(ValidationError):
        validator.validate({**payload, "future_field": True})
    incompatible = copy.deepcopy(payload)
    incompatible["schema_version"] = "musicdna-analysis-v1"
    with pytest.raises(ValidationError):
        validator.validate(incompatible)

    with pytest.raises(ValueError, match="musicdna-analysis-v0"):
        validate_analysis_result_payload(incompatible)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [("pyin_beam", 40.0), ("grid_independent_voicing", True)],
)
def test_schema_rejects_native_only_options_for_librosa(field_name: str, value: object) -> None:
    payload = json.loads(_result().to_json())
    payload["config"]["pitch"]["pyin_backend"] = "librosa"
    payload["config"]["pitch"][field_name] = value

    with pytest.raises(ValidationError):
        Draft202012Validator(load_analysis_result_schema()).validate(payload)


def test_semantic_validator_rejects_inconsistent_event_durations() -> None:
    payload = json.loads(_result().to_json())
    payload["events"][0]["duration_seconds"] = 0.2

    with pytest.raises(ValueError, match="duration_seconds must equal"):
        validate_analysis_result_payload(payload)

    payload = json.loads(_result().to_json())
    payload["events"][0]["duration_ms"] = 200.0
    with pytest.raises(ValueError, match="duration_ms must equal"):
        validate_analysis_result_payload(payload)


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (-0.1, 0.25),
        (0.25, 0.25),
    ],
)
def test_semantic_validator_rejects_invalid_event_ranges(start: float, end: float) -> None:
    payload = json.loads(_result().to_json())
    payload["events"][0]["start_seconds"] = start
    payload["events"][0]["end_seconds"] = end

    with pytest.raises(ValueError, match="non-negative start and positive duration"):
        validate_analysis_result_payload(payload)


def test_semantic_validator_rejects_events_beyond_analysis_duration() -> None:
    payload = json.loads(_result().to_json())
    payload["duration_seconds"] = 0.2

    with pytest.raises(ValueError, match="must not exceed duration_seconds"):
        validate_analysis_result_payload(payload)


def test_semantic_validator_rejects_overlapping_events() -> None:
    payload = json.loads(_result().to_json())
    payload["duration_seconds"] = 0.4
    payload["events"].append(
        {
            **payload["events"][0],
            "start_seconds": 0.2,
            "end_seconds": 0.4,
            "duration_seconds": 0.2,
            "duration_ms": 200.0,
        }
    )

    with pytest.raises(ValueError, match="ordered and non-overlapping"):
        validate_analysis_result_payload(payload)


def test_semantic_validator_accepts_float_rounding_at_adjacent_boundary() -> None:
    payload = json.loads(_result().to_json())
    payload["duration_seconds"] = 0.4
    payload["events"].append(
        {
            **payload["events"][0],
            "start_seconds": math.nextafter(0.25, 0.0),
            "end_seconds": 0.4,
            "duration_seconds": 0.15,
            "duration_ms": 150.0,
        }
    )

    validate_analysis_result_payload(payload)


def test_semantic_validator_rejects_unequal_pitch_track_lengths() -> None:
    payload = json.loads(_result().to_json())
    payload["pitch_track"]["frequencies_hz"] = [440.0]

    with pytest.raises(ValueError, match="arrays must have equal length"):
        validate_analysis_result_payload(payload)


def test_semantic_validator_includes_onset_strengths_in_length_check() -> None:
    payload = json.loads(_result().to_json())
    payload["pitch_track"]["onset_strengths"] = [0.1]

    with pytest.raises(ValueError, match="arrays must have equal length"):
        validate_analysis_result_payload(payload)


@pytest.mark.parametrize("times", [[0.1, 0.0], [-0.1, 0.0]])
def test_semantic_validator_rejects_invalid_pitch_track_times(times: list[float]) -> None:
    payload = json.loads(_result().to_json())
    payload["pitch_track"]["times_seconds"] = times

    with pytest.raises(ValueError, match="non-negative and ordered"):
        validate_analysis_result_payload(payload)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf")])
def test_semantic_validator_rejects_non_finite_numeric_values(value: object) -> None:
    payload = json.loads(_result().to_json())
    payload["events"][0]["duration_ms"] = value

    with pytest.raises(ValueError, match="must be a finite number"):
        validate_analysis_result_payload(payload)
