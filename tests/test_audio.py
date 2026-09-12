import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from musicdna_detector import AudioDecodingError, LoaderConfig, SoundFileLoader, ffmpeg_available


def test_loader_downmixes_and_resamples(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    stereo = np.column_stack((np.ones(800), np.zeros(800)))
    sf.write(path, stereo, 8_000, subtype="FLOAT")
    audio = SoundFileLoader(LoaderConfig(target_sample_rate=4_000)).load(path)
    assert audio.sample_rate == 4_000
    assert len(audio.samples) == 400
    assert float(np.mean(audio.samples)) == pytest.approx(0.5, abs=0.01)


def test_loader_reports_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        SoundFileLoader().load("does-not-exist.wav")


def test_loader_wraps_unsupported_or_corrupt_audio(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.wav"
    path.write_bytes(b"not audio")
    with pytest.raises(AudioDecodingError, match="FFmpeg is not invoked automatically"):
        SoundFileLoader().load(path)


@pytest.mark.parametrize(("executable", "expected"), [(None, False), ("/usr/bin/ffmpeg", True)])
def test_ffmpeg_environment_diagnostic(
    monkeypatch: pytest.MonkeyPatch, executable: str | None, expected: bool
) -> None:
    monkeypatch.setattr("musicdna_detector.audio.shutil.which", lambda name: executable)
    assert ffmpeg_available() is expected


def test_explicit_ffmpeg_fallback_is_safe_and_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "input with spaces.aac"
    path.write_bytes(b"encoded")
    calls: list[tuple[list[str], dict[str, object]]] = []
    decoded_paths: list[Path] = []

    def fake_decode(candidate: Path) -> tuple[np.ndarray, int]:
        if candidate == path:
            raise sf.LibsndfileError(1, "unsupported")
        decoded_paths.append(candidate)
        return np.ones((80, 1)), 8_000

    def fake_run(command: list[str], **kwargs: object) -> object:
        calls.append((command, kwargs))
        return type("Completed", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr("musicdna_detector.audio._decode_soundfile", fake_decode)
    monkeypatch.setattr("musicdna_detector.audio.shutil.which", lambda name: "ffmpeg")
    monkeypatch.setattr("musicdna_detector.audio.subprocess.run", fake_run)
    loader = SoundFileLoader(
        LoaderConfig(target_sample_rate=None, decoder_policy="native_then_ffmpeg")
    )
    audio = loader.load(path)
    assert calls[0][1]["shell"] is False
    assert calls[0][1]["timeout"] == 30.0
    assert str(path) in calls[0][0]
    assert audio.decoder_backend.value == "ffmpeg"
    assert audio.source.format == "aac"
    assert decoded_paths and not decoded_paths[0].exists()


def test_native_only_never_starts_a_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "unsupported.aac"
    path.write_bytes(b"encoded")
    monkeypatch.setattr(
        "musicdna_detector.audio._decode_soundfile",
        lambda candidate: (_ for _ in ()).throw(sf.LibsndfileError(1, "unsupported")),
    )
    monkeypatch.setattr(
        "musicdna_detector.audio.subprocess.run",
        lambda *args, **kwargs: pytest.fail("native_only must not start FFmpeg"),
    )
    with pytest.raises(AudioDecodingError, match="FFmpeg is not invoked automatically"):
        SoundFileLoader(LoaderConfig(decoder_policy="native_only")).load(path)


def test_ffmpeg_fallback_reports_missing_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "unsupported.aac"
    path.write_bytes(b"encoded")
    monkeypatch.setattr(
        "musicdna_detector.audio._decode_soundfile",
        lambda candidate: (_ for _ in ()).throw(sf.LibsndfileError(1, "unsupported")),
    )
    monkeypatch.setattr("musicdna_detector.audio.shutil.which", lambda name: None)
    with pytest.raises(AudioDecodingError, match="no ffmpeg executable was found on PATH"):
        SoundFileLoader(LoaderConfig(decoder_policy="native_then_ffmpeg")).load(path)


@pytest.mark.parametrize(
    ("completed", "message"),
    [
        (subprocess.CompletedProcess([], 2, "", "invalid input"), "invalid input"),
        (subprocess.TimeoutExpired([], 0.01), "exceeded"),
        (OSError("executable disappeared"), "could not be started"),
    ],
)
def test_ffmpeg_fallback_reports_process_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    completed: subprocess.CompletedProcess[str] | subprocess.TimeoutExpired | OSError,
    message: str,
) -> None:
    path = tmp_path / "unsupported.aac"
    path.write_bytes(b"encoded")
    monkeypatch.setattr(
        "musicdna_detector.audio._decode_soundfile",
        lambda candidate: (_ for _ in ()).throw(sf.LibsndfileError(1, "unsupported")),
    )
    monkeypatch.setattr("musicdna_detector.audio.shutil.which", lambda name: "/bin/ffmpeg")

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        if isinstance(completed, subprocess.TimeoutExpired):
            raise completed
        if isinstance(completed, OSError):
            raise completed
        return completed

    monkeypatch.setattr("musicdna_detector.audio.subprocess.run", fake_run)
    with pytest.raises(AudioDecodingError, match=message):
        SoundFileLoader(
            LoaderConfig(decoder_policy="native_then_ffmpeg", ffmpeg_timeout_seconds=0.01)
        ).load(path)


def test_opt_in_ffmpeg_fallback_runs_real_transcode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not ffmpeg_available():
        pytest.skip("FFmpeg executable is not available on PATH")
    source_wav = tmp_path / "source.wav"
    encoded = tmp_path / "encoded.m4a"
    samples = np.sin(2 * np.pi * 440 * np.arange(8_000) / 8_000)
    sf.write(source_wav, samples, 8_000)
    completed = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(source_wav),
            "-c:a",
            "aac",
            str(encoded),
        ],
        shell=False,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip(f"FFmpeg AAC encoder unavailable: {completed.stderr.strip()}")

    from musicdna_detector import audio as audio_module

    original_decode = audio_module._decode_soundfile

    def fail_native_only(candidate: Path) -> tuple[np.ndarray, int]:
        if candidate == encoded:
            raise sf.LibsndfileError(1, "force FFmpeg fallback for real transcode smoke")
        return original_decode(candidate)

    monkeypatch.setattr("musicdna_detector.audio._decode_soundfile", fail_native_only)
    loaded = SoundFileLoader(
        LoaderConfig(target_sample_rate=None, decoder_policy="native_then_ffmpeg")
    ).load(encoded)
    assert loaded.decoder_backend.value == "ffmpeg"
    assert loaded.sample_rate == 8_000
    assert loaded.duration_seconds == pytest.approx(1.0, abs=0.05)
