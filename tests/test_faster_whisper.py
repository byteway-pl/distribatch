"""faster-whisper engine with whisper-tiny converted to CTranslate2 (needs
faster-whisper + torch/transformers for the conversion; skipped otherwise)."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import distribatch as dib
from distribatch import ConfigError
from distribatch.engines.base import candidate_engines, resolve_engine

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture
def speech(tmp_path: Path) -> Path:
    if shutil.which("say") is None:
        pytest.skip("needs macOS `say` for speech audio")
    pytest.importorskip("faster_whisper")
    out = tmp_path / "speech.aiff"
    subprocess.run(["say", "-o", str(out), "Good morning, this is a test."], check=True)
    return out


def _job(tiny_models: dict[str, Path], **kw: object) -> dib.TranscribeJob:
    return dib.TranscribeJob(
        model=dib.models.path(str(tiny_models["whisper"])),
        audio_col="a",
        language="en",
        engine="faster_whisper",
        engine_options={"compute_type": "int8", "beam_size": 1},
        **kw,  # type: ignore[arg-type]
    )


def test_hf_whisper_is_converted_once_and_transcribes(
    tiny_models: dict[str, Path], speech: Path
) -> None:
    out = _job(tiny_models).run([{"a": str(speech)}, {"a": speech.read_bytes()}, {"a": b"junk"}])
    for row in out[:2]:
        assert row["output_error"] is None, row["output_error"]
        assert "morning" in row["output"].lower()
        assert row["output_meta"]["audio_seconds"] > 1.0
    assert out[2]["output_error"].startswith("InvalidInputError: cannot decode audio")
    converted = [p for p in Path(_cache_root()).glob("ct2-*") if (p / "model.bin").exists()]
    assert len(converted) == 1
    assert (converted[0] / "tokenizer.json").exists()


def test_timestamps_and_parallel_workers(tiny_models: dict[str, Path], speech: Path) -> None:
    job = dib.TranscribeJob(
        model=dib.models.path(str(tiny_models["whisper"])),
        audio_col="a",
        language="en",
        engine="faster_whisper",
        timestamps=True,
        engine_options={"compute_type": "int8", "beam_size": 1, "num_workers": 2},
    )
    out = job.run([{"a": str(speech)}] * 3)
    for row in out:
        value = row["output"]
        assert "morning" in value["text"].lower()
        assert value["language"] == "en"
        assert value["segments"] and value["segments"][0]["end"] > value["segments"][0]["start"]


def test_ct2_directory_is_used_as_is(tiny_models: dict[str, Path], speech: Path) -> None:
    _job(tiny_models).run([{"a": str(speech)}])  # produces the converted directory
    (converted,) = [p for p in Path(_cache_root()).glob("ct2-*") if (p / "model.bin").exists()]
    row = dib.TranscribeJob(
        model=dib.models.path(str(converted)),
        audio_col="a",
        engine="faster_whisper",
        engine_options={"compute_type": "int8"},
    ).run([{"a": str(speech)}])[0]
    assert "morning" in row["output"].lower()


def test_non_whisper_model_rejected(tmp_path: Path) -> None:
    pytest.importorskip("faster_whisper")
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "bert"}))
    (tmp_path / "model.safetensors").write_bytes(b"w")
    job = dib.TranscribeJob(
        model=dib.models.path(str(tmp_path)), audio_col="a", engine="faster_whisper"
    )
    with pytest.raises(ConfigError, match="needs a Whisper model"):
        job.run([{"a": b"x"}])


def test_auto_prefers_faster_whisper_when_installed() -> None:
    pytest.importorskip("faster_whisper")
    spec = dib.models.path("/w")
    assert candidate_engines(spec, "auto", "transcribe") == ("faster_whisper", "transformers")
    assert resolve_engine(spec, "auto", "transcribe") == "faster_whisper"


def test_options_validated_on_driver() -> None:
    with pytest.raises(ConfigError, match="num_workers"):
        dib.TranscribeJob(
            model="/w", audio_col="a", engine="faster_whisper", engine_options={"num_workers": 0}
        )


def _cache_root() -> str:
    import os

    return os.environ["DISTRIBATCH_CACHE_DIR"]
