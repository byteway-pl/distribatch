"""Audio decoding for local speech models: any input -> mono float32 at a target rate.

`soundfile` (libsndfile: wav, flac, ogg, mp3) first; formats it can't read (m4a/aac,
webm, ...) go through an `ffmpeg` binary when one is on PATH. Resampling uses scipy's
polyphase filter (`distribatch[audio]`).
"""

from __future__ import annotations

import io
import shutil
import subprocess
from math import gcd
from typing import Any

from distribatch._deps import require
from distribatch.types import AudioInput, InvalidInputError

WHISPER_SAMPLING_RATE = 16_000


def decode(audio: AudioInput, sampling_rate: int = WHISPER_SAMPLING_RATE) -> tuple[Any, float]:
    """(mono float32 numpy samples at `sampling_rate`, duration in seconds)."""
    np = require("numpy", "audio")
    sf = require("soundfile", "audio")
    source: Any = audio.path if audio.path is not None else io.BytesIO(audio.data or b"")
    try:
        samples, rate = sf.read(source, dtype="float32", always_2d=True)
    except Exception as exc:
        if shutil.which("ffmpeg") is None:
            raise InvalidInputError(
                f"cannot decode audio {audio.label}: {exc} (install ffmpeg for m4a/aac/webm)"
            ) from exc
        return _decode_ffmpeg(audio, sampling_rate)
    mono = samples.mean(axis=1).astype(np.float32)
    if mono.size == 0:
        raise InvalidInputError(f"audio {audio.label} has no samples")
    return resample(mono, rate, sampling_rate), mono.size / rate


def resample(samples: Any, rate: int, target: int) -> Any:
    if rate == target:
        return samples
    signal = require("scipy.signal", "audio")
    divisor = gcd(rate, target)
    out = signal.resample_poly(samples, target // divisor, rate // divisor)
    return out.astype("float32")


def _decode_ffmpeg(audio: AudioInput, sampling_rate: int) -> tuple[Any, float]:
    np = require("numpy", "audio")
    source = audio.path if audio.path is not None else "pipe:0"
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-i", source,
        "-f", "f32le", "-ac", "1", "-ar", str(sampling_rate), "pipe:1",
    ]  # fmt: skip
    result = subprocess.run(
        cmd, input=audio.data if audio.path is None else None, capture_output=True
    )
    if result.returncode != 0 or not result.stdout:
        message = result.stderr.decode(errors="replace").strip() or "no audio decoded"
        raise InvalidInputError(f"cannot decode audio {audio.label}: {message}")
    samples = np.frombuffer(result.stdout, dtype=np.float32)
    return samples, samples.size / sampling_rate
