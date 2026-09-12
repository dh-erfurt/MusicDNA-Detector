from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from musicdna_detector import AnalysisConfig, AudioBuffer, SegmentationConfig, analyze_audio
from musicdna_detector.models import FloatArray, NoteEvent, PitchTrack


def test_analyze_audio_accepts_buffer() -> None:
    sample_rate = 8_000
    times = np.arange(sample_rate // 2) / sample_rate
    audio = AudioBuffer(np.sin(2 * np.pi * 440 * times), sample_rate)
    result = analyze_audio(audio)
    assert result.sample_rate == sample_rate
    assert result.duration_seconds == pytest.approx(0.5)


def test_analyze_audio_requires_rate_for_raw_array() -> None:
    with pytest.raises(TypeError, match="AudioBuffer"):
        analyze_audio(np.zeros(100))  # type: ignore[arg-type]


def test_analyze_audio_accepts_path(tmp_path: Path) -> None:
    path = tmp_path / "tone.wav"
    samples = np.sin(2 * np.pi * 220 * np.arange(4_000) / 8_000)
    sf.write(path, samples, 8_000)
    result = analyze_audio(path, AnalysisConfig())
    assert result.sample_rate == 22_050
    assert result.events


def test_analyze_audio_accepts_injected_loader() -> None:
    class StubLoader:
        def load(self, path: str | Path) -> AudioBuffer:
            assert path == "virtual.wav"
            times = np.arange(4_000) / 8_000
            return AudioBuffer(np.sin(2 * np.pi * 220 * times), 8_000)

    result = analyze_audio("virtual.wav", loader=StubLoader())
    assert result.sample_rate == 8_000


def test_analyze_attaches_spectral_flux_onsets_when_configured() -> None:
    from musicdna_detector import analyze

    class StubEstimator:
        def estimate(self, audio: AudioBuffer) -> PitchTrack:
            assert audio.sample_rate == 8_000
            times = np.arange(0.0, 1.0, 0.05)
            return PitchTrack(
                times,
                np.full(times.shape, 440.0),
                np.full(times.shape, 0.95),
                np.full(times.shape, -12.0),
            )

    class CapturingSegmenter:
        seen_strengths = False

        def segment(self, track: PitchTrack, *, audio_duration: float) -> tuple[NoteEvent, ...]:
            assert audio_duration == pytest.approx(1.0)
            assert track.onset_strengths is not None
            assert len(track.onset_strengths) == len(track.times_seconds)
            self.seen_strengths = True
            return (NoteEvent(0.0, 0.5, 440.0, 0.9),)

    sample_rate = 8_000
    samples = np.zeros(sample_rate, dtype=np.float64)
    samples[2_000:4_000] = np.sin(2 * np.pi * 440 * np.arange(2_000) / sample_rate)
    segmenter = CapturingSegmenter()

    analyze(
        AudioBuffer(samples, sample_rate),
        AnalysisConfig(onset_feature="spectral_flux"),
        estimator=StubEstimator(),
        segmenter=segmenter,
    )

    assert segmenter.seen_strengths is True


def test_analyze_reuses_one_raw_onset_envelope_for_both_consumers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from musicdna_detector import analyze, dsp

    class StubEstimator:
        def estimate(self, audio: AudioBuffer) -> PitchTrack:
            times = np.arange(0.0, audio.duration_seconds, 0.05)
            return PitchTrack(
                times,
                np.full(times.shape, 440.0),
                np.full(times.shape, 0.95),
                np.full(times.shape, -12.0),
            )

    calls = 0
    original = dsp.normalized_onset_strength

    def counting_onset_strength(
        samples: FloatArray, sample_rate: float, hop_length: int
    ) -> FloatArray:
        nonlocal calls
        calls += 1
        return original(samples, sample_rate, hop_length)

    monkeypatch.setattr(dsp, "normalized_onset_strength", counting_onset_strength)
    sample_rate = 8_000
    times = np.arange(sample_rate, dtype=np.float64) / sample_rate
    audio = AudioBuffer(np.sin(2 * np.pi * 440 * times), sample_rate)

    analyze(
        audio,
        AnalysisConfig(onset_feature="spectral_flux"),
        estimator=StubEstimator(),
    )

    assert calls == 1


def test_analyze_preserves_spectral_flux_same_pitch_rearticulation() -> None:
    from musicdna_detector import analyze

    class StubEstimator:
        def estimate(self, audio: AudioBuffer) -> PitchTrack:
            assert audio.sample_rate == 8_000
            times = np.arange(0.0, audio.duration_seconds, 0.02)
            return PitchTrack(
                times,
                np.full(times.shape, 440.0),
                np.full(times.shape, 0.95),
                np.full(times.shape, -12.0),
            )

    sample_rate = 8_000
    duration_seconds = 0.40
    samples = np.zeros(int(sample_rate * duration_seconds), dtype=np.float64)
    for start, duration in ((0.0, 0.16), (0.16, 0.20)):
        offset = int(start * sample_rate)
        size = int(duration * sample_rate)
        times = np.arange(size, dtype=np.float64) / sample_rate
        attack = min(320, size)
        envelope = np.ones(size, dtype=np.float64)
        envelope[:attack] = np.linspace(0.15, 1.0, attack)
        samples[offset : offset + size] += 0.35 * envelope * np.sin(2 * np.pi * 440 * times)

    def segment_with(**overrides: object) -> tuple[NoteEvent, ...]:
        return analyze(
            AudioBuffer(samples, sample_rate),
            AnalysisConfig(
                segmentation=SegmentationConfig(
                    onset_feature="spectral_flux",
                    spectral_flux_onset_threshold=0.05,
                    minimum_duration_seconds=0.04,
                    coverage_recovery="off",
                    **overrides,  # type: ignore[arg-type]
                )
            ),
            estimator=StubEstimator(),
        ).events

    # The decoder and consolidation stages have separate responsibilities here: an
    # onset reward alone must not force a boundary when the rest of the evidence is
    # insufficient.
    assert len(segment_with()) == 1
    assert len(segment_with(state_onset_reward=6.0)) == 1


_PROBE_FRAME = 256


class _LevelProbe:
    """Pitch estimator recording the frame-wise RMS dBFS the pipeline actually sees.

    Framing matters here: a single loud transient inflates a whole-buffer RMS but
    only one frame of a real level track, which is what segmentation gates on.
    """

    def __init__(self) -> None:
        self.median_level_dbfs: list[float] = []

    def estimate(self, audio: AudioBuffer) -> PitchTrack:
        usable = (audio.samples.size // _PROBE_FRAME) * _PROBE_FRAME
        frames = audio.samples[:usable].reshape(-1, _PROBE_FRAME)
        rms = np.sqrt(np.mean(np.square(frames), axis=1))
        with np.errstate(divide="ignore"):
            levels = 20 * np.log10(rms)
        self.median_level_dbfs.append(float(np.median(levels)))
        count = len(levels)
        return PitchTrack(
            np.arange(count) * _PROBE_FRAME / audio.sample_rate,
            np.full(count, 440.0),
            np.full(count, 0.99),
            levels,
        )


class _AudibleSegmenter:
    """Emit one note only while the clip stays above the silence threshold."""

    def segment(self, track: PitchTrack, *, audio_duration: float) -> tuple[NoteEvent, ...]:
        if float(np.median(track.levels_dbfs)) <= -60:
            return ()
        return (NoteEvent(0.0, audio_duration, 440.0, 0.9),)


def _quiet_sine(*, attenuation_dbfs: float = -60.0) -> FloatArray:
    return np.asarray(
        np.sin(2 * np.pi * 440 * np.arange(8_000) / 8_000) * 10 ** (attenuation_dbfs / 20),
        dtype=np.float64,
    )


def test_audio_normalization_is_opt_in_and_recovers_attenuated_signal() -> None:
    from musicdna_detector import analyze

    samples = _quiet_sine()
    default_probe = _LevelProbe()
    default = analyze(
        AudioBuffer(samples, 8_000), estimator=default_probe, segmenter=_AudibleSegmenter()
    )
    optin_probe = _LevelProbe()
    optin = analyze(
        AudioBuffer(samples, 8_000),
        AnalysisConfig(normalize_audio=True),
        estimator=optin_probe,
        segmenter=_AudibleSegmenter(),
    )

    assert default_probe.median_level_dbfs[0] < -60
    assert optin_probe.median_level_dbfs[0] > -60
    assert not default.events
    assert optin.events


def test_audio_normalization_survives_a_single_loud_transient() -> None:
    """A lone click must not drag the whole clip back below the silence threshold.

    Peak normalization scales such a clip by the click, pushing the actual signal
    *further* down than doing nothing at all. The percentile reference ignores it.
    """
    from musicdna_detector import analyze

    samples = _quiet_sine()
    clicked = samples.copy()
    clicked[100] = 0.95

    percentile_probe = _LevelProbe()
    percentile = analyze(
        AudioBuffer(clicked, 8_000),
        AnalysisConfig(normalize_audio=True),
        estimator=percentile_probe,
        segmenter=_AudibleSegmenter(),
    )
    peak_probe = _LevelProbe()
    analyze(
        AudioBuffer(clicked, 8_000),
        AnalysisConfig(normalize_audio=True, normalization_reference_percentile=100.0),
        estimator=peak_probe,
        segmenter=_AudibleSegmenter(),
    )
    clean_probe = _LevelProbe()
    analyze(
        AudioBuffer(samples, 8_000),
        AnalysisConfig(normalize_audio=True),
        estimator=clean_probe,
        segmenter=_AudibleSegmenter(),
    )

    assert percentile.events
    assert peak_probe.median_level_dbfs[0] < -60 < percentile_probe.median_level_dbfs[0]
    # The click must barely perturb the achieved level relative to the clean clip.
    assert percentile_probe.median_level_dbfs[0] == pytest.approx(
        clean_probe.median_level_dbfs[0], abs=0.5
    )


def test_audio_normalization_percentile_100_reproduces_peak_normalization() -> None:
    from musicdna_detector import analyze

    probe = _LevelProbe()
    analyze(
        AudioBuffer(_quiet_sine(attenuation_dbfs=-30.0), 8_000),
        AnalysisConfig(normalize_audio=True, normalization_reference_percentile=100.0),
        estimator=probe,
        segmenter=_AudibleSegmenter(),
    )
    # A sine normalized to a -18 dBFS peak has an RMS of -18 - 3.01 dBFS; the frame
    # grid does not fall on whole periods, so allow a hundredth of a dB.
    assert probe.median_level_dbfs[0] == pytest.approx(-18.0 - 20 * np.log10(np.sqrt(2)), abs=0.01)


def test_audio_normalization_rejects_invalid_reference_percentile() -> None:
    for invalid in (0.0, -1.0, 100.5, float("nan")):
        with pytest.raises(ValueError, match="normalization_reference_percentile"):
            AnalysisConfig(normalize_audio=True, normalization_reference_percentile=invalid)


def test_audio_normalization_handles_silence_and_nonfinite_ndarray() -> None:
    config = AnalysisConfig(normalize_audio=True)
    silence = analyze_audio(np.zeros(256), config, sample_rate=8_000)
    invalid = analyze_audio(np.array([np.nan, np.inf, -np.inf]), config, sample_rate=8_000)
    assert silence.duration_seconds == pytest.approx(256 / 8_000)
    assert invalid.duration_seconds == pytest.approx(3 / 8_000)
    assert not silence.events
    assert not invalid.events


def test_audio_normalization_is_serialized() -> None:
    config = AnalysisConfig(
        normalize_audio=True,
        normalization_target_dbfs=-12.0,
        normalization_reference_percentile=95.0,
    )
    payload = config.to_dict()
    assert payload["normalize_audio"] is True
    assert payload["normalization_target_dbfs"] == pytest.approx(-12.0)
    assert payload["normalization_reference_percentile"] == pytest.approx(95.0)
    assert '"normalize_audio": true' in config.to_json().lower()
