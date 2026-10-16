import pickle

import pandas as pd
import pytest
from pydantic import BaseModel, ValidationError

import distribatch as dib
from distribatch import ConfigError
from distribatch.engines.fake import FakeEngine


class Label(BaseModel):
    label: str


def test_run_on_records() -> None:
    job = dib.GenerateJob(model="fake", prompt="Say: {text}")
    out = job.run([{"id": 1, "text": "hi"}, {"id": 2, "text": "yo"}], output_col="ans")
    assert [r["ans"] for r in out] == ["Say: hi", "Say: yo"]
    assert [r["ans_error"] for r in out] == [None, None]
    assert out[0]["id"] == 1
    meta = out[0]["ans_meta"]
    assert meta["model_id"] == "fake"
    assert meta["attempts"] == 1
    assert set(meta) == {
        "prompt_tokens",
        "completion_tokens",
        "audio_seconds",
        "latency_ms",
        "attempts",
        "model_id",
        "finish_reason",
    }


def test_run_on_pandas_preserves_index_and_order() -> None:
    df = pd.DataFrame({"text": ["a", "b", "c"]}, index=[10, 5, 7])
    out = dib.GenerateJob(model="fake", prompt="{text}").run(df, batch_size=2)
    assert list(out.index) == [10, 5, 7]
    assert list(out["output"]) == ["a", "b", "c"]
    assert out["output_error"].isna().all()
    assert "output" not in df.columns  # input untouched


def test_row_failures_do_not_fail_the_job() -> None:
    job = dib.GenerateJob(model=dib.models.fake(fail_marker="BAD"), prompt="{t}")
    out = job.run([{"t": "ok"}, {"t": "BAD"}, {"t": "fine"}])
    assert [r["output"] for r in out] == ["ok", None, "fine"]
    assert out[1]["output_error"] == "EngineError: fake failure"


def test_batch_crash_only_fails_its_batch() -> None:
    job = dib.GenerateJob(model=dib.models.fake(crash_marker="BOOM"), prompt="{t}")
    out = job.run([{"t": "a"}, {"t": "BOOM"}, {"t": "c"}, {"t": "d"}], batch_size=2)
    assert [r["output"] for r in out] == [None, None, "c", "d"]
    assert out[0]["output_error"].startswith("EngineError: batch failed: RuntimeError")


def test_missing_key_in_one_record_is_a_row_error() -> None:
    out = dib.GenerateJob(model="fake", prompt="{t}").run([{"t": "a"}, {"other": 1}])
    assert out[0]["output"] == "a"
    assert out[1]["output_error"].startswith("PromptRenderError")
    assert out[1]["output_meta"]["attempts"] == 0


def test_structured_output() -> None:
    job = dib.GenerateJob(
        model=dib.models.fake(response='```json\n{"label": "pos"}\n```'),
        prompt="{t}",
        output_schema=Label,
    )
    assert job.run([{"t": "x"}])[0]["output"] == {"label": "pos"}


def test_structured_output_validation_error_is_row_level() -> None:
    job = dib.GenerateJob(
        model=dib.models.fake(response='{"lbl": 1}'), prompt="{t}", output_schema=Label
    )
    row = job.run([{"t": "x"}])[0]
    assert row["output"] is None
    assert row["output_error"].startswith("SchemaValidationError: label: Field required")


def test_inline_schema_for_engines_without_structured_output() -> None:
    # The fake engine echoes the user message; the schema must land in the system one.
    job = dib.GenerateJob(
        model=dib.models.fake(structured_output=False), prompt="{t}", output_schema=Label
    )
    row = job.run([{"t": '{"label": "neg"}'}])[0]
    assert row["output"] == {"label": "neg"}
    assert row["output_meta"]["prompt_tokens"] > 5  # system prompt with the schema counted


def test_config_errors_raise_before_any_work() -> None:
    job = dib.GenerateJob(model="fake", prompt="{missing}")
    with pytest.raises(ConfigError, match="missing"):
        job.run(pd.DataFrame({"t": ["a"]}))
    with pytest.raises(ConfigError, match="already exist"):
        dib.GenerateJob(model="fake", prompt="{t}").run([{"t": "a", "output_error": 1}])
    with pytest.raises(ConfigError, match="batch_size"):
        dib.GenerateJob(model="fake", prompt="{t}").run([{"t": "a"}], batch_size=0)
    with pytest.raises(ConfigError, match="unsupported input"):
        dib.GenerateJob(model="fake", prompt="{t}").run("not a table")
    assert FakeEngine.load_count == 0


def test_job_validation() -> None:
    with pytest.raises(ConfigError, match="cannot run"):
        dib.GenerateJob(model="fake", prompt="{t}", engine="vllm")
    with pytest.raises(ValidationError):
        dib.GenerateJob(model="fake", prompt="{t}", temperature=0.1)  # type: ignore[call-arg]
    with pytest.raises(ConfigError, match="positional"):
        dib.GenerateJob(model="fake", prompt="{}")


def test_empty_input() -> None:
    assert dib.GenerateJob(model="fake", prompt="{t}").run([]) == []


def test_preview() -> None:
    df = pd.DataFrame({"t": list("abcdefg")})
    out = dib.GenerateJob(model="fake", prompt="{t}").preview(df, n=3)
    assert list(out["output"]) == ["a", "b", "c"]


def test_job_is_picklable() -> None:
    job = dib.GenerateJob(model="fake", prompt="{t}", output_schema=Label, system="s")
    clone = pickle.loads(pickle.dumps(job))
    assert clone == job
    assert clone.run([{"t": '{"label": "x"}'}])[0]["output"] == {"label": "x"}


def test_engine_loaded_once_across_runs() -> None:
    job = dib.GenerateJob(model="fake", prompt="{t}")
    job.run([{"t": "a"}] * 10, batch_size=3)
    job.run([{"t": "b"}])
    assert FakeEngine.load_count == 1
