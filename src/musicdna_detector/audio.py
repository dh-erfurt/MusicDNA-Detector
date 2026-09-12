"""Audio decoding and sample-rate conversion."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from scipy.signal import resample_poly

from .models import AudioBuffer, DecoderBackend, LoaderConfig, SourceMetadata


class AudioDecodingError(RuntimeError):
    """Raised when the configured decoders cannot read an input file."""


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _decode_soundfile(path: Path) -> tuple[NDArray[np.float64], int]:
    samples, sample_rate = sf.read(path, dtype="float64", always_2d=True)
    return np.asarray(samples, dtype=np.float64), int(sample_rate)


class SoundFileLoader:
    """Decode natively, with an explicit and safely executed FFmpeg fallback policy."""

    def __init__(self, config: LoaderConfig | None = None) -> None:
        self.config = config or LoaderConfig()

    def _decode_ffmpeg(self, source: Path) -> tuple[NDArray[np.float64], int]:
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise AudioDecodingError(
                "Native libsndfile decoding failed and decoder_policy='native_then_ffmpeg', "
                "but no ffmpeg executable was found on PATH. Install FFmpeg or use a "
                "libsndfile-supported WAV/FLAC/OGG input."
            )
        with tempfile.TemporaryDirectory(prefix="musicdna-") as directory:
            decoded = Path(directory) / "decoded.wav"
            command = [
                executable,
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(source),
                "-vn",
                "-acodec",
                "pcm_f32le",
                "-y",
                str(decoded),
            ]
            try:
                completed = subprocess.run(
                    command,
                    shell=False,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.config.ffmpeg_timeout_seconds,
                )
            except subprocess.TimeoutExpired as error:
                raise AudioDecodingError(
                    f"FFmpeg decoding exceeded {self.config.ffmpeg_timeout_seconds:g} seconds"
                ) from error
            except OSError as error:
                raise AudioDecodingError(
                    f"FFmpeg executable '{executable}' could not be started: {error}"
                ) from error
            if completed.returncode != 0:
                detail = completed.stderr.strip().splitlines()[-1:] or ["unknown decoder error"]
                raise AudioDecodingError(f"FFmpeg could not decode the audio: {detail[0]}")
            try:
                return _decode_soundfile(decoded)
            except (sf.LibsndfileError, OSError) as error:
                raise AudioDecodingError("FFmpeg output could not be read as WAV") from error

    def load(self, path: str | Path) -> AudioBuffer:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {source}")
        backend = DecoderBackend.NATIVE
        try:
            samples, sample_rate = _decode_soundfile(source)
        except (sf.LibsndfileError, OSError) as native_error:
            if self.config.decoder_policy == "native_only":
                raise AudioDecodingError(
                    "Could not decode the input with SoundFile/libsndfile. WAV, FLAC, and OGG "
                    "are commonly supported; MP3/AAC support depends on libsndfile. To permit "
                    "an installed FFmpeg executable on PATH, explicitly select "
                    "decoder_policy='native_then_ffmpeg'. FFmpeg is not invoked automatically."
                ) from native_error
            samples, sample_rate = self._decode_ffmpeg(source)
            backend = DecoderBackend.FFMPEG
        original_sample_rate = sample_rate
        channels = samples.shape[1]
        duration = len(samples) / sample_rate
        mono = np.asarray(np.mean(samples, axis=1, dtype=np.float64), dtype=np.float64)
        target = self.config.target_sample_rate
        if target is not None and target != sample_rate:
            divisor = gcd(sample_rate, target)
            mono = np.asarray(
                resample_poly(mono, target // divisor, sample_rate // divisor), dtype=np.float64
            )
            sample_rate = target
        return AudioBuffer(
            mono,
            sample_rate,
            source=SourceMetadata(
                kind="path",
                format=source.suffix.lower().lstrip(".") or None,
                original_sample_rate=original_sample_rate,
                analyzed_sample_rate=sample_rate,
                channels=channels,
                duration_s=duration,
                resampled=original_sample_rate != sample_rate,
            ),
            decoder_backend=backend,
            decoder_policy=self.config.decoder_policy,
        )
