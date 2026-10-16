import json
import time
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

import distribatch as dib
from distribatch import ConfigError
from distribatch.engines.base import clear_engine_cache, get_engine
from distribatch.engines.openai_api import resolve_credentials
from openai_stub import OpenAIStub


class Label(BaseModel):
    label: str


FAST_RETRY = {"backoff_base_s": 0.001, "backoff_max_s": 0.01}


def _model(stub: OpenAIStub, name: str = "stub-model") -> dib.models.OpenAIModel:
    return dib.models.openai(name, base_url=stub.base_url, api_key_env="STUB_KEY")


# --- generate -------------------------------------------------------------------------


def test_chat_completion(openai_stub: OpenAIStub) -> None:
    job = dib.GenerateJob(
        model=_model(openai_stub),
        prompt="Echo {t}",
        system="sys",
        sampling=dib.Sampling(temperature=0.3, max_tokens=7, stop=("END",), seed=1),
    )
    row = job.run([{"t": "hello"}])[0]
    assert row["output"] == "Echo hello"
    assert row["output_error"] is None
    meta = row["output_meta"]
    assert (meta["prompt_tokens"], meta["completion_tokens"]) == (3, 2)
    assert (meta["attempts"], meta["model_id"], meta["finish_reason"]) == (1, "stub-model", "stop")
    sent = openai_stub.state.calls[0][1]
    assert sent["messages"][0] == {"role": "system", "content": "sys"}
    assert (sent["temperature"], sent["max_tokens"], sent["stop"], sent["seed"]) == (
        0.3,
        7,
        ["END"],
        1,
    )
    assert "response_format" not in sent


def test_native_structured_output(openai_stub: OpenAIStub) -> None:
    openai_stub.state.chat_response = '{"label": "pos"}'
    job = dib.GenerateJob(model=_model(openai_stub), prompt="{t}", output_schema=Label)
    assert job.run([{"t": "x"}])[0]["output"] == {"label": "pos"}
    fmt = openai_stub.state.calls[0][1]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["schema"] == Label.model_json_schema()


def test_response_format_rejected_falls_back_to_inline_schema(openai_stub: OpenAIStub) -> None:
    openai_stub.state.chat_response = '{"label": "neg"}'
    openai_stub.state.reject_response_format = True
    job = dib.GenerateJob(
        model=_model(openai_stub), prompt="{t}", output_schema=Label, engine_options=FAST_RETRY
    )
    out = job.run([{"t": "a"}, {"t": "b"}, {"t": "c"}], batch_size=1)
    assert [r["output"] for r in out] == [{"label": "neg"}] * 3
    calls = [payload for _, payload in openai_stub.state.calls]
    # First row: rejected native attempt + inline retry; later rows go inline directly.
    assert len(calls) == 4
    assert "response_format" in calls[0]
    assert all("response_format" not in c for c in calls[1:])
    assert "JSON schema" in calls[1]["messages"][0]["content"]
    assert out[0]["output_meta"]["attempts"] == 2


def test_retries_on_429_and_500(openai_stub: OpenAIStub) -> None:
    job = dib.GenerateJob(model=_model(openai_stub), prompt="{t}", engine_options=FAST_RETRY)
    out = job.run([{"t": "RATE429:2"}, {"t": "ERR500:1"}])
    assert [r["output"] for r in out] == ["RATE429:2", "ERR500:1"]
    assert [r["output_meta"]["attempts"] for r in out] == [3, 2]


def test_gives_up_after_max_retries(openai_stub: OpenAIStub) -> None:
    job = dib.GenerateJob(
        model=_model(openai_stub), prompt="{t}", engine_options={**FAST_RETRY, "max_retries": 1}
    )
    row = job.run([{"t": "RATE429:5"}])[0]
    assert row["output"] is None
    assert row["output_error"].startswith("RateLimitError")
    assert row["output_meta"]["attempts"] == 2


def test_client_errors_are_not_retried(openai_stub: OpenAIStub) -> None:
    out = dib.GenerateJob(model=_model(openai_stub), prompt="{t}").run(
        [{"t": "BAD400"}, {"t": "fine"}]
    )
    assert out[0]["output_error"].startswith("BadRequestError")
    assert out[0]["output_meta"]["attempts"] == 1
    assert out[1]["output"] == "fine"


def test_concurrency_is_bounded(openai_stub: OpenAIStub) -> None:
    openai_stub.state.delay_s = 0.05
    job = dib.GenerateJob(
        model=_model(openai_stub), prompt="{t}", engine_options={"max_concurrency": 4}
    )
    out = job.run([{"t": str(i)} for i in range(16)], batch_size=16)
    assert [r["output"] for r in out] == [str(i) for i in range(16)]
    assert 1 < openai_stub.state.max_in_flight <= 4


def test_requests_per_minute(openai_stub: OpenAIStub) -> None:
    job = dib.GenerateJob(
        model=_model(openai_stub), prompt="{t}", engine_options={"requests_per_minute": 600}
    )
    start = time.perf_counter()
    job.run([{"t": str(i)} for i in range(4)], batch_size=4)
    assert time.perf_counter() - start >= 0.3  # 3 gaps of 0.1 s


def test_max_completion_tokens_param(openai_stub: OpenAIStub) -> None:
    job = dib.GenerateJob(
        model=_model(openai_stub),
        prompt="{t}",
        engine_options={"max_tokens_param": "max_completion_tokens"},
    )
    job.run([{"t": "x"}])
    sent = openai_stub.state.calls[0][1]
    assert "max_tokens" not in sent
    assert sent["max_completion_tokens"] == 512


# --- embed ----------------------------------------------------------------------------


def test_embeddings_are_batched(openai_stub: OpenAIStub) -> None:
    job = dib.EmbedJob(
        model=_model(openai_stub, "emb"),
        text="{t}",
        dimensions=3,
        engine_options={"embed_max_inputs": 4},
    )
    out = job.run([{"t": "x" * (i + 1)} for i in range(10)], batch_size=10)
    assert [r["output"] for r in out] == [[float(i + 1), 1.0, 1.0] for i in range(10)]
    calls = openai_stub.state.calls
    assert sorted(len(p["input"]) for _, p in calls) == [2, 4, 4]  # concurrent: any order
    assert all(p["dimensions"] == 3 for _, p in calls)
    # Usage is only attributable to a row for single-input requests.
    assert out[0]["output_meta"]["prompt_tokens"] is None


def test_embedding_bad_input_is_isolated(openai_stub: OpenAIStub) -> None:
    job = dib.EmbedJob(model=_model(openai_stub, "emb"), text="{t}")
    out = job.run([{"t": "ok one"}, {"t": "BAD400"}, {"t": "ok"}])
    assert out[0]["output"] == [6.0, 1.0, 1.0, 1.0]
    assert out[1]["output_error"].startswith("BadRequestError")
    assert out[2]["output"] == [2.0, 1.0, 1.0, 1.0]
    assert out[0]["output_meta"]["prompt_tokens"] == 2  # re-sent alone


def test_embedding_retry(openai_stub: OpenAIStub) -> None:
    job = dib.EmbedJob(model=_model(openai_stub, "emb"), text="{t}", engine_options=FAST_RETRY)
    row = job.run([{"t": "RATE429:1"}])[0]
    assert row["output_error"] is None
    assert row["output_meta"]["attempts"] == 2


# --- transcribe -----------------------------------------------------------------------


def test_transcribe_path_and_bytes(openai_stub: OpenAIStub, tmp_path: Path) -> None:
    audio = tmp_path / "call.mp3"
    audio.write_bytes(b"hello from file")
    wav = b"RIFF\x00\x00\x00\x00WAVEfmt spoken bytes"
    job = dib.TranscribeJob(
        model=_model(openai_stub, "whisper-1"), audio_col="a", language="pl", prompt="hint"
    )
    out = job.run([{"a": str(audio)}, {"a": wav}])
    assert out[0]["output"] == "hello from file"
    assert out[1]["output"].endswith("spoken bytes")
    # Requests run concurrently: index the calls by uploaded file name.
    fields = {f["file"][0]: f for _, f in openai_stub.state.calls}
    assert set(fields) == {"call.mp3", "audio.wav"}
    sent = fields["call.mp3"]
    assert (sent["language"], sent["prompt"], sent["response_format"]) == (
        "pl",
        "hint",
        "json",
    )


def test_transcribe_timestamps(openai_stub: OpenAIStub) -> None:
    job = dib.TranscribeJob(model=_model(openai_stub, "whisper-1"), audio_col="a", timestamps=True)
    row = job.run([{"a": b"some speech"}])[0]
    assert row["output"] == {
        "text": "some speech",
        "language": "english",
        "segments": [{"start": 0.0, "end": 2.5, "text": "some speech"}],
    }
    assert row["output_meta"]["audio_seconds"] == 2.5
    sent = openai_stub.state.calls[0][1]
    assert sent["response_format"] == "verbose_json"
    assert sent["timestamp_granularities[]"] == "segment"


def test_translate_route(openai_stub: OpenAIStub) -> None:
    job = dib.TranscribeJob(
        model=_model(openai_stub, "whisper-1"), audio_col="a", translate=True, language="de"
    )
    assert job.run([{"a": b"guten tag"}])[0]["output"] == "EN: guten tag"
    path, fields = openai_stub.state.calls[0]
    assert path == "/v1/audio/translations"
    assert "language" not in fields


def test_audio_errors_are_row_level(openai_stub: OpenAIStub, tmp_path: Path) -> None:
    job = dib.TranscribeJob(
        model=_model(openai_stub, "whisper-1"),
        audio_col="a",
        engine_options={"max_upload_mb": 0.00001},
    )
    out = job.run([{"a": str(tmp_path / "missing.wav")}, {"a": b"x" * 100}])
    assert out[0]["output_error"].startswith("InvalidInputError: cannot read audio file")
    assert out[1]["output_error"].startswith("InvalidInputError: audio is 0.0 MB, over the")
    assert openai_stub.state.calls == []


# --- config / credentials -------------------------------------------------------------


def test_engine_options_are_validated_on_the_driver() -> None:
    with pytest.raises(ConfigError, match="max_concurency"):
        dib.GenerateJob(model="openai:m", prompt="{t}", engine_options={"max_concurency": 3})
    with pytest.raises(ConfigError, match="max_concurrency"):
        dib.GenerateJob(model="openai:m", prompt="{t}", engine_options={"max_concurrency": 0})


def test_missing_api_key_env_fails_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOPE_KEY", raising=False)
    job = dib.GenerateJob(
        model=dib.models.openai("m", base_url="http://127.0.0.1:9", api_key_env="NOPE_KEY"),
        prompt="{t}",
    )
    with pytest.raises(ConfigError, match="NOPE_KEY is not set"):
        job.run([{"t": "x"}])


def test_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("OPENAI_API_KEY", "DATABRICKS_HOST", "DATABRICKS_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        resolve_credentials(dib.models.openai("m"))
    assert resolve_credentials(dib.models.openai("m", base_url="http://vllm:8000/v1")) == (
        "http://vllm:8000/v1",
        "unused",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-1")
    assert resolve_credentials(dib.models.openai("m")) == (None, "sk-1")

    monkeypatch.setenv("DATABRICKS_HOST", "adb-123.azuredatabricks.net/")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-x")
    assert resolve_credentials(dib.models.databricks("llama")) == (
        "https://adb-123.azuredatabricks.net/serving-endpoints",
        "dapi-x",
    )
    monkeypatch.setenv("MY_TOKEN", "dapi-mine")
    assert resolve_credentials(dib.models.databricks("llama", api_key_env="MY_TOKEN"))[1] == (
        "dapi-mine"
    )


def test_databricks_uri() -> None:
    spec = dib.models.parse("databricks:databricks-gte-large-en")
    assert spec == dib.models.databricks("databricks-gte-large-en")
    with pytest.raises(ValueError, match="base_url"):
        dib.models.openai("m", base_url="localhost:8000")


def test_engine_works_inside_a_running_event_loop(openai_stub: OpenAIStub) -> None:
    import asyncio

    job = dib.GenerateJob(model=_model(openai_stub), prompt="{t}")

    async def notebook_cell() -> Any:  # Jupyter/Databricks run cells inside a loop
        return job.run([{"t": "inside"}])

    assert asyncio.run(notebook_cell())[0]["output"] == "inside"


def test_close_stops_the_loop_thread(openai_stub: OpenAIStub) -> None:
    engine: Any = get_engine(_model(openai_stub), "auto", {}, "generate")
    loop = engine._loop
    assert loop.alive
    clear_engine_cache()
    assert not loop.alive


def test_json_payloads_are_plain(openai_stub: OpenAIStub) -> None:
    dib.GenerateJob(model=_model(openai_stub), prompt="{t}").run([{"t": "x"}])
    json.dumps(openai_stub.state.calls)  # only JSON-native values were sent
