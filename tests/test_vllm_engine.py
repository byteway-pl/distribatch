"""vLLM engine against a fake `vllm` module (real vLLM needs a CUDA GPU; it is
verified on Databricks via dev/ notebooks)."""

import sys
import types
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

import distribatch as dib
from distribatch import ConfigError


class Params:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def _fake_vllm(*, new_api: bool) -> types.ModuleType:
    module = types.ModuleType("vllm")
    created: list[dict[str, Any]] = []

    class LLM:
        def __init__(self, **kwargs: Any) -> None:
            if "runner" in kwargs and not new_api:
                raise TypeError("unexpected keyword 'runner'")
            created.append(kwargs)
            self.kwargs = kwargs

        def chat(self, conversations: list[Any], sampling_params: list[Any], use_tqdm: bool) -> Any:
            out = []
            for conv, params in zip(conversations, sampling_params, strict=True):
                user = conv[-1]["content"]
                if "TOO LONG" in user:
                    raise ValueError("prompt is longer than max_model_len")
                text = '{"label": "ok"}' if _structured(params) else f"echo {user}"
                out.append(
                    types.SimpleNamespace(
                        prompt_token_ids=[1] * len(user.split()),
                        outputs=[
                            types.SimpleNamespace(text=text, token_ids=[7, 8], finish_reason="stop")
                        ],
                        params=params,
                    )
                )
            return out

        def embed(self, texts: list[str], use_tqdm: bool, pooling_params: Any = None) -> Any:
            dims = pooling_params.dimensions if pooling_params else 3
            return [
                types.SimpleNamespace(
                    prompt_token_ids=[1, 2],
                    outputs=types.SimpleNamespace(embedding=[float(len(t))] * dims),
                )
                for t in texts
            ]

    sampling_params = types.ModuleType("vllm.sampling_params")
    if new_api:
        sampling_params.StructuredOutputsParams = Params  # type: ignore[attr-defined]
    else:
        sampling_params.GuidedDecodingParams = Params  # type: ignore[attr-defined]
    module.LLM = LLM  # type: ignore[attr-defined]
    module.SamplingParams = Params  # type: ignore[attr-defined]
    module.PoolingParams = Params  # type: ignore[attr-defined]
    module.sampling_params = sampling_params  # type: ignore[attr-defined]
    module.created = created  # type: ignore[attr-defined]
    return module


def _structured(params: Any) -> bool:
    return hasattr(params, "structured_outputs") or hasattr(params, "guided_decoding")


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"w")
    return tmp_path


class Label(BaseModel):
    label: str


@pytest.mark.parametrize("new_api", [True, False])
def test_generate(monkeypatch: pytest.MonkeyPatch, model_dir: Path, new_api: bool) -> None:
    vllm = _fake_vllm(new_api=new_api)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    job = dib.GenerateJob(
        model=dib.models.path(str(model_dir)),
        engine="vllm",
        prompt="{t}",
        output_schema=Label,
        sampling=dib.Sampling(temperature=0.2, max_tokens=9, stop=("X",), seed=3, top_p=0.9),
        engine_options={"tensor_parallel_size": 2, "enforce_eager": True},
    )
    out = job.run([{"t": "a b"}, {"t": "c"}])
    assert [r["output"] for r in out] == [{"label": "ok"}] * 2
    meta = out[0]["output_meta"]
    assert (meta["prompt_tokens"], meta["completion_tokens"], meta["finish_reason"]) == (
        2,
        2,
        "stop",
    )
    (llm_kwargs,) = vllm.created  # type: ignore[attr-defined]
    assert llm_kwargs["model"] == str(model_dir)
    assert llm_kwargs["tensor_parallel_size"] == 2
    assert llm_kwargs["enforce_eager"] is True  # unknown options pass through


def test_sampling_params_mapping(monkeypatch: pytest.MonkeyPatch, model_dir: Path) -> None:
    vllm = _fake_vllm(new_api=True)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    from distribatch.engines.base import get_engine
    from distribatch.types import ChatMessage, ChatRequest

    engine: Any = get_engine(dib.models.path(str(model_dir)), "vllm", {}, "generate")
    sampling = dib.Sampling(temperature=0.2, max_tokens=9, stop=("X",), seed=3, top_p=0.9)
    params = engine._sampling_params(
        ChatRequest(0, (ChatMessage("user", "q"),), sampling, {"type": "object"})
    )
    assert (params.temperature, params.max_tokens, params.stop, params.seed, params.top_p) == (
        0.2,
        9,
        ["X"],
        3,
        0.9,
    )
    assert params.structured_outputs.json == {"type": "object"}


def test_bad_prompt_is_isolated(monkeypatch: pytest.MonkeyPatch, model_dir: Path) -> None:
    monkeypatch.setitem(sys.modules, "vllm", _fake_vllm(new_api=True))
    job = dib.GenerateJob(model=dib.models.path(str(model_dir)), engine="vllm", prompt="{t}")
    out = job.run([{"t": "fine"}, {"t": "TOO LONG"}, {"t": "also fine"}])
    assert [r["output"] for r in out] == ["echo fine", None, "echo also fine"]
    assert out[1]["output_error"] == "ValueError: prompt is longer than max_model_len"


@pytest.mark.parametrize("new_api", [True, False])
def test_embed(monkeypatch: pytest.MonkeyPatch, model_dir: Path, new_api: bool) -> None:
    vllm = _fake_vllm(new_api=new_api)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    job = dib.EmbedJob(
        model=dib.models.path(str(model_dir)), engine="vllm", text="{t}", dimensions=4
    )
    out = job.run([{"t": "abc"}, {"t": "de"}])
    assert [r["output"] for r in out] == [[3.0] * 4, [2.0] * 4]
    (llm_kwargs,) = vllm.created  # type: ignore[attr-defined]
    assert llm_kwargs.get("runner") == "pooling" if new_api else llm_kwargs["task"] == "embed"


def test_options_validated_on_driver() -> None:
    with pytest.raises(ConfigError, match="tensor_parallel_size"):
        dib.GenerateJob(
            model="/w", prompt="{t}", engine="vllm", engine_options={"tensor_parallel_size": 0}
        )
    with pytest.raises(ConfigError, match="does not support task 'transcribe'"):
        dib.TranscribeJob(model="/w", audio_col="a", engine="vllm")


def test_auto_prefers_transformers_without_gpu() -> None:
    from distribatch.engines.base import resolve_engine

    # No CUDA here (and no vllm installed): auto must fall back to transformers.
    assert resolve_engine(dib.models.path("/w"), "auto", "generate") == "transformers"
