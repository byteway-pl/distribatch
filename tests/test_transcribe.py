import pandas as pd
import pytest

import distribatch as dib
from distribatch import ConfigError, InvalidInputError
from distribatch.job import audio_input
from distribatch.types import AudioInput


def test_transcribe_paths_and_bytes() -> None:
    job = dib.TranscribeJob(model="fake", audio_col="audio")
    out = job.run([{"audio": "/Volumes/m/raw/a.wav"}, {"audio": b"hello from bytes"}])
    assert out[0]["output"] == "transcript of /Volumes/m/raw/a.wav"
    assert out[1]["output"] == "hello from bytes"
    assert out[0]["output_meta"]["audio_seconds"] == 1.0


def test_transcribe_with_timestamps() -> None:
    job = dib.TranscribeJob(model="fake", audio_col="a", timestamps=True, language="pl")
    value = job.run(pd.DataFrame({"a": ["/x.wav"]}))["output"].iloc[0]
    assert value == {
        "text": "transcript of /x.wav",
        "language": "pl",
        "segments": [{"start": 0.0, "end": 1.0, "text": "transcript of /x.wav"}],
    }


def test_bad_audio_cells_are_row_errors() -> None:
    job = dib.TranscribeJob(model=dib.models.fake(fail_marker="broken"), audio_col="a")
    out = job.run([{"a": None}, {"a": 42}, {"a": ""}, {"a": "/broken.wav"}, {"a": "/ok.wav"}])
    errors = [r["output_error"] for r in out]
    assert errors[0] == "InvalidInputError: audio is null"
    assert errors[1].startswith("InvalidInputError: audio column must hold")
    assert errors[2] == "InvalidInputError: audio path is empty"
    assert errors[3] == "EngineError: fake failure"
    assert errors[4] is None


def test_audio_input_normalization() -> None:
    assert audio_input("dbfs:/mnt/a.wav") == AudioInput(path="/dbfs/mnt/a.wav")
    assert audio_input("dbfs:/Volumes/m/raw/a.wav") == AudioInput(path="/Volumes/m/raw/a.wav")
    assert audio_input(bytearray(b"ab")) == AudioInput(data=b"ab")
    assert audio_input(memoryview(b"ab")).data == b"ab"
    with pytest.raises(InvalidInputError):
        audio_input(float("nan"))
    with pytest.raises(ValueError, match="exactly one"):
        AudioInput()


def test_finish_shapes() -> None:
    plain = dib.TranscribeJob(model="fake", audio_col="a")
    assert plain.finish({"text": "hi", "segments": []}) == "hi"
    with pytest.raises(dib.EngineError):
        plain.finish(None)
    ts = dib.TranscribeJob(model="fake", audio_col="a", timestamps=True)
    with pytest.raises(dib.EngineError, match="malformed"):
        ts.finish({"text": "hi", "segments": [{"start": 0}]})
    with pytest.raises(dib.EngineError, match="'text'"):
        ts.finish("hi")


def test_transcribe_config_errors() -> None:
    with pytest.raises(ConfigError, match="missing"):
        dib.TranscribeJob(model="fake", audio_col="audio").run(pd.DataFrame({"path": ["/a"]}))
    with pytest.raises(ConfigError, match="does not support task 'transcribe'"):
        dib.TranscribeJob(model="/w", audio_col="a", engine="vllm")
