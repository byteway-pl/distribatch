import io
import shutil
from pathlib import Path

import pytest

from distribatch import InvalidInputError
from distribatch.audio import decode
from distribatch.types import AudioInput

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")
pytest.importorskip("scipy")


def _tone(seconds: float, rate: int, channels: int = 1) -> object:
    t = np.arange(int(seconds * rate)) / rate
    wave = (0.3 * np.sin(2 * np.pi * 440 * t)).astype("float32")
    return np.stack([wave] * channels, axis=1) if channels > 1 else wave


def _wav_bytes(seconds: float, rate: int, channels: int = 1, fmt: str = "WAV") -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, _tone(seconds, rate, channels), rate, format=fmt)
    return buffer.getvalue()


def test_decode_resamples_stereo_to_mono_16k(tmp_path: Path) -> None:
    path = tmp_path / "a.wav"
    path.write_bytes(_wav_bytes(1.5, 44_100, channels=2))
    samples, seconds = decode(AudioInput(path=str(path)))
    assert samples.dtype == np.float32
    assert samples.ndim == 1
    assert samples.size == 24_000
    assert seconds == pytest.approx(1.5)


def test_decode_bytes_flac() -> None:
    samples, seconds = decode(AudioInput(data=_wav_bytes(0.5, 16_000, fmt="FLAC")))
    assert samples.size == 8_000
    assert seconds == pytest.approx(0.5)


@pytest.mark.skipif(shutil.which("ffmpeg") is not None, reason="ffmpeg would try instead")
def test_undecodable_without_ffmpeg() -> None:
    with pytest.raises(InvalidInputError, match="install ffmpeg"):
        decode(AudioInput(data=b"definitely not audio"))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_undecodable_with_ffmpeg() -> None:
    with pytest.raises(InvalidInputError, match="cannot decode audio <20 bytes>"):
        decode(AudioInput(data=b"definitely not audio"))


def test_missing_file() -> None:
    with pytest.raises(InvalidInputError, match="cannot decode audio /nope.wav"):
        decode(AudioInput(path="/nope.wav"))
