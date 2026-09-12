"""Parity tests for the in-house pYIN implementation.

librosa is the reference here, not a dependency under test. Assertions use exact
equality because ``allclose`` would hide output changes at this compatibility boundary.
"""

from __future__ import annotations

import librosa
import librosa.core.pitch as librosa_pitch  # lazy_loader hides this behind librosa.core
import numpy as np
import pytest
import scipy.stats
from numpy.typing import NDArray

from musicdna_detector import AnalysisConfig, AudioBuffer
from musicdna_detector.analysis import analyze
from musicdna_detector.dsp import pyin as native_pyin
from musicdna_detector.dsp import pyin_detailed
from musicdna_detector.dsp import viterbi as native_viterbi
from musicdna_detector.dsp import yin as native_yin
from musicdna_detector.dsp.observation import observation_probabilities, threshold_prior
from musicdna_detector.models import PitchConfig

SAMPLE_RATE = 22_050
FRAME_LENGTH = 2_048
HOP_LENGTH = 256
FMIN = 65.406
FMAX = 2_093.005


def _signal(kind: str, seconds: float = 1.2) -> NDArray[np.float64]:
    """Deterministic test material covering voiced, unvoiced and silent stretches."""
    rng = np.random.default_rng(20260727)
    n = int(SAMPLE_RATE * seconds)
    times = np.arange(n) / SAMPLE_RATE
    if kind == "tone":
        return np.sin(2 * np.pi * 220.0 * times)
    if kind == "glide":
        sweep = 180.0 * 2 ** (times * 1.5)
        return np.sin(2 * np.pi * np.cumsum(sweep) / SAMPLE_RATE)
    if kind == "vibrato":
        modulation = 1 + 0.03 * np.sin(2 * np.pi * 5.5 * times)
        return np.sin(2 * np.pi * np.cumsum(330.0 * modulation) / SAMPLE_RATE)
    if kind == "noise":
        return rng.standard_normal(n) * 0.05
    if kind == "silence":
        return np.zeros(n)
    if kind == "mixed":
        parts = [
            np.sin(2 * np.pi * 196.0 * times[: n // 3]),
            np.zeros(n // 3),
            np.sin(2 * np.pi * 392.0 * times[: n - 2 * (n // 3)]) * 0.4,
        ]
        return np.concatenate(parts) + rng.standard_normal(n) * 0.002
    raise ValueError(kind)


ALL_SIGNALS = ["tone", "glide", "vibrato", "noise", "silence", "mixed"]


def _framed(signal: NDArray[np.float64]) -> NDArray[np.float64]:
    padded = np.pad(signal, (FRAME_LENGTH // 2, FRAME_LENGTH // 2), mode="constant")
    return native_yin.frame_signal(padded, FRAME_LENGTH, HOP_LENGTH)


def _periods() -> tuple[int, int]:
    min_period = int(np.floor(SAMPLE_RATE / FMAX))
    max_period = min(int(np.ceil(SAMPLE_RATE / FMIN)), FRAME_LENGTH - 1)
    return min_period, max_period


@pytest.mark.parametrize("kind", ALL_SIGNALS)
def test_framing_matches_librosa(kind: str) -> None:
    signal = _signal(kind)
    padded = np.pad(signal, (FRAME_LENGTH // 2, FRAME_LENGTH // 2), mode="constant")
    expected = librosa.util.frame(padded, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    assert np.array_equal(expected, native_yin.frame_signal(padded, FRAME_LENGTH, HOP_LENGTH))


@pytest.mark.parametrize("kind", ALL_SIGNALS)
def test_difference_function_matches_librosa(kind: str) -> None:
    frames = _framed(_signal(kind))
    min_period, max_period = _periods()
    expected = librosa_pitch._cumulative_mean_normalized_difference(frames, min_period, max_period)
    actual = native_yin.cumulative_mean_normalized_difference(frames, min_period, max_period)
    assert np.array_equal(expected, actual)


@pytest.mark.parametrize("kind", ALL_SIGNALS)
def test_parabolic_interpolation_matches_librosa(kind: str) -> None:
    frames = _framed(_signal(kind))
    min_period, max_period = _periods()
    difference = native_yin.cumulative_mean_normalized_difference(frames, min_period, max_period)
    expected = librosa_pitch._parabolic_interpolation(difference)
    assert np.array_equal(expected, native_yin.parabolic_interpolation(difference))


def test_threshold_prior_matches_librosa_setup() -> None:
    thresholds, beta_probs = threshold_prior(100, (2.0, 18.0))
    expected_thresholds = np.linspace(0, 1, 101)
    expected_probs = np.diff(scipy.stats.beta.cdf(expected_thresholds, 2.0, 18.0))
    assert np.array_equal(expected_thresholds, thresholds)
    assert np.array_equal(expected_probs, beta_probs)


@pytest.mark.parametrize("kind", ALL_SIGNALS)
def test_observation_probabilities_match_librosa(kind: str) -> None:
    frames = _framed(_signal(kind))
    min_period, max_period = _periods()
    difference = native_yin.cumulative_mean_normalized_difference(frames, min_period, max_period)
    shifts = native_yin.parabolic_interpolation(difference)
    thresholds, beta_probs = threshold_prior(100, (2.0, 18.0))
    bins_per_semitone = 10
    n_pitch_bins = int(np.floor(12 * bins_per_semitone * np.log2(FMAX / FMIN))) + 1

    helper = next(value for name, value in vars(librosa_pitch).items() if "pyin_helper" in name)
    expected_probs, expected_voiced = helper(
        difference,
        shifts,
        SAMPLE_RATE,
        thresholds,
        2.0,
        beta_probs,
        0.01,
        min_period,
        FMIN,
        n_pitch_bins,
        bins_per_semitone,
    )
    actual_probs, actual_voiced = observation_probabilities(
        difference,
        shifts,
        sr=SAMPLE_RATE,
        thresholds=thresholds,
        boltzmann_parameter=2.0,
        beta_probs=beta_probs,
        no_trough_prob=0.01,
        min_period=min_period,
        fmin=FMIN,
        n_pitch_bins=n_pitch_bins,
        n_bins_per_semitone=bins_per_semitone,
    )
    assert np.array_equal(expected_probs[0], actual_probs)
    assert np.array_equal(expected_voiced[0], actual_voiced)


@pytest.mark.parametrize("n_states,width", [(31, 7), (120, 51), (481, 51)])
def test_transition_band_matches_librosa(n_states: int, width: int) -> None:
    expected = librosa.sequence.transition_local(n_states, width, window="triangle", wrap=False)
    assert np.array_equal(expected, native_viterbi.transition_local(n_states, width))


def test_transition_loop_matches_librosa() -> None:
    expected = librosa.sequence.transition_loop(2, 0.99)
    assert np.array_equal(expected, native_viterbi.transition_loop(2, 0.99))


@pytest.mark.parametrize("kind", ALL_SIGNALS)
def test_pyin_matches_librosa(kind: str) -> None:
    """The whole point: same f0, same voicing, same probabilities, bit for bit."""
    signal = _signal(kind)
    expected_f0, expected_voiced, expected_probability = librosa.pyin(
        signal,
        sr=SAMPLE_RATE,
        fmin=FMIN,
        fmax=FMAX,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )
    actual_f0, actual_voiced, actual_probability = native_pyin(
        signal,
        sr=SAMPLE_RATE,
        fmin=FMIN,
        fmax=FMAX,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )
    assert np.array_equal(expected_voiced, actual_voiced)
    assert np.array_equal(expected_probability, actual_probability)
    assert np.array_equal(expected_f0[expected_voiced], actual_f0[actual_voiced])
    assert np.array_equal(np.isnan(expected_f0), np.isnan(actual_f0))


def test_pyin_matches_librosa_on_a_capped_range() -> None:
    """The humming profile's C6 cap changes the state count; equality must survive it."""
    signal = _signal("mixed")
    kwargs = dict(
        sr=SAMPLE_RATE,
        fmin=FMIN,
        fmax=1_046.502,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )
    expected_f0, expected_voiced, expected_probability = librosa.pyin(signal, **kwargs)
    actual_f0, actual_voiced, actual_probability = native_pyin(signal, **kwargs)
    assert np.array_equal(expected_voiced, actual_voiced)
    assert np.array_equal(expected_probability, actual_probability)
    assert np.array_equal(expected_f0[expected_voiced], actual_f0[actual_voiced])


def test_pyin_matches_librosa_at_a_non_default_resolution() -> None:
    signal = _signal("mixed")
    kwargs = dict(
        sr=SAMPLE_RATE,
        fmin=FMIN,
        fmax=FMAX,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        resolution=0.25,
    )
    expected_f0, expected_voiced, expected_probability = librosa.pyin(signal, **kwargs)
    actual_f0, actual_voiced, actual_probability = native_pyin(signal, **kwargs)

    assert np.array_equal(expected_voiced, actual_voiced)
    assert np.array_equal(expected_probability, actual_probability)
    assert np.array_equal(expected_f0[expected_voiced], actual_f0[actual_voiced])


def test_pyin_rejects_multichannel_input() -> None:
    with pytest.raises(ValueError, match="mono"):
        native_pyin(np.zeros((2, 4_096)), sr=SAMPLE_RATE, fmin=FMIN, fmax=FMAX)


def test_pyin_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError, match="fmin"):
        native_pyin(np.zeros(4_096), sr=SAMPLE_RATE, fmin=FMAX, fmax=FMIN)


@pytest.mark.parametrize("profile", ["default", "humming", "singing"])
@pytest.mark.parametrize("kind", ["mixed", "vibrato", "noise"])
def test_analyze_is_identical_across_backends(profile: str, kind: str) -> None:
    """The gate that actually matters: identical events out of the full pipeline."""
    from dataclasses import replace

    audio = AudioBuffer(_signal(kind, seconds=2.0), SAMPLE_RATE)
    reference_config = AnalysisConfig(profile=profile)
    native_config = replace(
        reference_config, pitch=replace(reference_config.pitch, pyin_backend="native")
    )

    reference = analyze(audio, reference_config)
    native = analyze(audio, native_config)

    assert len(reference.events) == len(native.events)
    for expected, actual in zip(reference.events, native.events, strict=True):
        assert expected.start_seconds == actual.start_seconds
        assert expected.end_seconds == actual.end_seconds
        assert expected.frequency_hz == actual.frequency_hz
        assert expected.midi_pitch == actual.midi_pitch
        assert expected.cents == actual.cents
        assert expected.confidence == actual.confidence
    assert np.array_equal(
        reference.pitch_track.voiced_probabilities, native.pitch_track.voiced_probabilities
    )
    assert np.array_equal(reference.pitch_track.levels_dbfs, native.pitch_track.levels_dbfs)


def test_pitch_config_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="pyin_backend"):
        PitchConfig(pyin_backend="banded")  # type: ignore[arg-type]


def test_pitch_config_defaults_to_the_native_core() -> None:
    """librosa is optional and therefore cannot be the runtime default."""
    assert PitchConfig().pyin_backend == "native"


@pytest.mark.parametrize("kind", ["tone", "glide", "vibrato", "mixed"])
def test_a_wide_beam_leaves_the_decode_unchanged(kind: str) -> None:
    """The beam is an approximation; wide enough, it stops being one.

    This asserts that a sufficiently wide beam preserves the exact decode, so a change
    to the pruning geometry cannot silently narrow the reachable state set.
    """
    signal = _signal(kind)
    kwargs = dict(
        sr=SAMPLE_RATE, fmin=FMIN, fmax=FMAX, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    )
    exact_f0, exact_voiced, _ = native_pyin(signal, **kwargs)
    beamed_f0, beamed_voiced, _ = native_pyin(signal, beam=40.0, **kwargs)

    assert np.array_equal(exact_voiced, beamed_voiced)
    assert np.array_equal(exact_f0[exact_voiced], beamed_f0[beamed_voiced])


def test_a_beam_rejects_non_positive_widths() -> None:
    with pytest.raises(ValueError, match="beam"):
        native_pyin(
            _signal("tone"),
            sr=SAMPLE_RATE,
            fmin=FMIN,
            fmax=FMAX,
            frame_length=FRAME_LENGTH,
            hop_length=HOP_LENGTH,
            beam=0.0,
        )


def test_pitch_config_defaults_to_an_exact_decode() -> None:
    assert PitchConfig().pyin_beam is None


def test_pitch_config_rejects_a_beam_on_the_librosa_backend() -> None:
    """librosa has no beam; silently ignoring the field would misreport what ran."""
    with pytest.raises(ValueError, match="native"):
        PitchConfig(pyin_backend="librosa", pyin_beam=40.0)


@pytest.mark.parametrize("kind", ["tone", "glide", "mixed"])
def test_candidates_do_not_change_the_decode(kind: str) -> None:
    """Keeping the candidates is an observation, not a decision."""
    signal = _signal(kind)
    kwargs = dict(
        sr=SAMPLE_RATE, fmin=FMIN, fmax=FMAX, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    )
    plain = pyin_detailed(signal, **kwargs)
    detailed = pyin_detailed(signal, n_candidates=5, **kwargs)

    assert np.array_equal(plain.voiced_flag, detailed.voiced_flag)
    assert np.array_equal(plain.voiced_prob, detailed.voiced_prob)
    assert np.array_equal(plain.f0[plain.voiced_flag], detailed.f0[detailed.voiced_flag])
    assert plain.candidate_frequencies_hz is None


@pytest.mark.parametrize("n_candidates", [1, 3, 8])
def test_candidates_are_ordered_and_shaped(n_candidates: int) -> None:
    signal = _signal("mixed")
    result = pyin_detailed(
        signal,
        sr=SAMPLE_RATE,
        fmin=FMIN,
        fmax=FMAX,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        n_candidates=n_candidates,
    )
    frequencies = result.candidate_frequencies_hz
    probabilities = result.candidate_probabilities
    assert frequencies is not None and probabilities is not None
    assert frequencies.shape == (result.f0.size, n_candidates)
    assert probabilities.shape == frequencies.shape
    # Strongest first, and an absent candidate is NaN with zero mass rather than a
    # plausible-looking frequency nobody voted for.
    assert np.all(np.diff(probabilities, axis=1) <= 0)
    assert np.all(np.isnan(frequencies[probabilities <= 0]))
    assert np.all(np.isfinite(frequencies[probabilities > 0]))


def test_the_strongest_candidate_carries_the_frames_the_decode_kept() -> None:
    """Not an identity: the decode also weighs transitions, so it may overrule the peak.

    On a steady tone it should rarely need to, which is what this pins down -- if the
    candidates ever stopped describing the same frames as the decode, this drifts first.
    """
    signal = _signal("tone")
    result = pyin_detailed(
        signal,
        sr=SAMPLE_RATE,
        fmin=FMIN,
        fmax=FMAX,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        n_candidates=1,
    )
    frequencies = result.candidate_frequencies_hz
    assert frequencies is not None
    voiced = result.voiced_flag & np.isfinite(frequencies[:, 0])
    agreement = np.abs(1200.0 * np.log2(frequencies[voiced, 0] / result.f0[voiced])) <= 50.0
    assert agreement.mean() > 0.9


def test_the_spectral_flux_pass_preserves_every_other_track_field() -> None:
    """Rebuilding a PitchTrack field by field is how fields go missing.

    Candidate arrays were once dropped on profiles that run this pass because fields
    were listed positionally. The pass uses dataclasses.replace so future fields remain
    intact as well.
    """
    from dataclasses import fields, replace

    from musicdna_detector.analysis import _attach_spectral_flux_onsets
    from musicdna_detector.models import AudioBuffer, PitchTrack

    audio = AudioBuffer(_signal("mixed", seconds=1.5), SAMPLE_RATE)
    config = AnalysisConfig(profile="humming").segmentation
    frames = 40
    track = PitchTrack(
        np.arange(frames) * 0.01,
        np.full(frames, 220.0),
        np.full(frames, 0.9),
        np.full(frames, -20.0),
    )

    updated = _attach_spectral_flux_onsets(audio, track, config)

    assert updated.onset_strengths is not None
    for field in fields(PitchTrack):
        if field.name == "onset_strengths":
            continue
        before = getattr(track, field.name)
        after = getattr(updated, field.name)
        if before is None:
            assert after is None, field.name
        else:
            assert np.array_equal(before, after), field.name
    # replace() on the untouched track is the identity the pass relies on.
    assert np.array_equal(replace(track).times_seconds, track.times_seconds)


def test_grid_independent_voicing_is_a_no_op_at_the_default_resolution() -> None:
    """The fix must not move anything where the grid already is what it assumes."""
    signal = _signal("mixed")
    kwargs = dict(
        sr=SAMPLE_RATE, fmin=FMIN, fmax=FMAX, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    )
    plain = pyin_detailed(signal, **kwargs)
    fixed = pyin_detailed(signal, grid_independent_voicing=True, **kwargs)

    assert np.array_equal(plain.voiced_flag, fixed.voiced_flag)
    assert np.array_equal(plain.f0[plain.voiced_flag], fixed.f0[fixed.voiced_flag])


@pytest.mark.parametrize("resolution", [0.2, 0.25])
def test_grid_independent_voicing_holds_the_voicing_balance(resolution: float) -> None:
    """librosa divides the unvoiced mass by the bin count, so a coarser grid votes unvoiced.

    The pitch grid is a throughput choice; the voiced/unvoiced decision should not follow
    it. This asserts the decoupling in the direction that matters: with the fix the voiced
    fraction stays closer to the default grid's than without it.
    """
    signal = _signal("mixed", seconds=2.5)
    kwargs = dict(
        sr=SAMPLE_RATE, fmin=FMIN, fmax=FMAX, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    )
    reference = pyin_detailed(signal, **kwargs).voiced_flag.mean()
    librosa_like = pyin_detailed(signal, resolution=resolution, **kwargs).voiced_flag.mean()
    decoupled = pyin_detailed(
        signal, resolution=resolution, grid_independent_voicing=True, **kwargs
    ).voiced_flag.mean()

    assert abs(decoupled - reference) <= abs(librosa_like - reference)
