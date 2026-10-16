import pytest

from distribatch import ConfigError, Sampling, models
from distribatch.engines.base import check_engine_available, get_engine, resolve_engine
from distribatch.engines.fake import FakeEngine
from distribatch.types import ChatMessage, ChatRequest


def _request(row_id: int, text: str) -> ChatRequest:
    return ChatRequest(row_id, (ChatMessage("user", text),), Sampling())


def test_engine_is_loaded_once_per_spec() -> None:
    a = get_engine(models.fake(), "auto", {}, "generate")
    b = get_engine(models.fake(), "fake", {}, "generate")
    c = get_engine(models.fake(response="x"), "auto", {}, "generate")
    assert a is b
    assert c is not a
    assert FakeEngine.load_count == 2


def test_engine_options_are_part_of_the_cache_key() -> None:
    assert get_engine(models.fake(), "auto", {"k": 1}, "generate") is not get_engine(
        models.fake(), "auto", {"k": 2}, "generate"
    )


def test_incompatible_engine_and_model() -> None:
    with pytest.raises(ConfigError, match="cannot run"):
        resolve_engine(models.fake(), "openai", "generate")
    with pytest.raises(ConfigError, match="cannot run"):
        resolve_engine(models.openai("m"), "vllm", "generate")


def test_unimplemented_engine_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    import dataclasses

    from distribatch.engines import base

    planned = dataclasses.replace(base._ENGINES["faster_whisper"], impl=None)
    monkeypatch.setitem(base._ENGINES, "faster_whisper", planned)
    with pytest.raises(ConfigError, match="not implemented"):
        check_engine_available(models.uc("a.b.c", alias="x"), "faster_whisper", "transcribe")


def test_fake_engine_behaviour() -> None:
    engine = get_engine(models.fake(fail_marker="BAD", crash_marker="BOOM"), "auto", {}, "generate")
    out = engine.generate([_request(0, "hello world"), _request(1, "BAD row")])
    assert out[0].output == "hello world"
    assert out[0].meta.completion_tokens == 2
    assert out[1].output is None
    assert out[1].error == "EngineError: fake failure"
    with pytest.raises(RuntimeError, match="crash"):
        engine.generate([_request(0, "ok"), _request(1, "BOOM")])


def test_task_support_is_checked_up_front() -> None:
    with pytest.raises(ConfigError, match="does not support task 'transcribe'"):
        resolve_engine(models.path("/w"), "vllm", "transcribe")
    with pytest.raises(ConfigError, match="unknown engine"):
        resolve_engine(models.fake(), "nope", "generate")


def test_auto_engine_order_per_task() -> None:
    from distribatch.engines.base import candidate_engines

    path = models.path("/w")
    assert candidate_engines(path, "auto", "generate") == ("vllm", "transformers")
    assert candidate_engines(path, "auto", "embed") == ("vllm", "transformers")
    assert candidate_engines(path, "auto", "transcribe") == ("faster_whisper", "transformers")


def test_engine_rejects_unsupported_task_method() -> None:
    from distribatch.engines.base import Engine

    class GenerateOnly(Engine):
        name = "generate_only"

    with pytest.raises(Exception, match="does not support task 'embed'"):
        GenerateOnly(models.fake(), {}).embed([])
