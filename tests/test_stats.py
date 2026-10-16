import sys
import types
from typing import Any

import pandas as pd
import pytest

import distribatch as dib
from distribatch import ConfigError


def _out() -> list[dict[str, Any]]:
    job = dib.GenerateJob(model=dib.models.fake(fail_marker="BAD"), prompt="{t}")
    return job.run([{"t": "one two"}, {"t": "BAD"}, {"t": "three"}, {"t": "one two"}], dedupe=True)


def test_summarize_records() -> None:
    stats = dib.summarize(_out())
    assert (stats.rows, stats.succeeded, stats.failed, stats.reused) == (4, 3, 1, 1)
    assert stats.errors == {"EngineError": 1}
    assert stats.prompt_tokens == 3  # the reused row is not counted twice
    assert stats.completion_tokens == 3
    assert stats.latency_ms_p50 is not None
    assert stats.success_rate == 0.75
    text = str(stats)
    assert "4 rows: 3 ok, 1 failed" in text and "1 reused" in text and "EngineError x1" in text


def test_summarize_pandas_and_custom_column() -> None:
    df = dib.GenerateJob(model="fake", prompt="{t}").run(
        pd.DataFrame({"t": ["a", "b"]}), output_col="ans"
    )
    assert dib.summarize(df, output_col="ans").succeeded == 2
    with pytest.raises(ConfigError, match="output_col"):
        dib.summarize(df)


def test_cost() -> None:
    stats = dib.RunStats(
        rows=1, succeeded=1, failed=0, reused=0, retries=0, prompt_tokens=2_000_000,
        completion_tokens=500_000, audio_seconds=120.0, latency_ms_p50=None,
        latency_ms_p95=None,
    )  # fmt: skip
    assert stats.cost(input_per_million=0.5, output_per_million=2.0, per_audio_minute=0.006) == (
        pytest.approx(1.0 + 1.0 + 0.012)
    )


def test_log_to_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    logged: dict[str, Any] = {}
    fake = types.ModuleType("mlflow")
    fake.log_metrics = lambda m: logged.update(metrics=m)  # type: ignore[attr-defined]
    fake.log_params = lambda p: logged.update(params=p)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    job = dib.GenerateJob(model="fake", prompt="{t}")
    dib.summarize(_out()).log_to_mlflow(job)
    assert logged["metrics"]["distribatch.rows"] == 4.0
    assert logged["metrics"]["distribatch.errors.EngineError"] == 1.0
    assert logged["params"]["distribatch.task"] == "generate"
    assert logged["params"]["distribatch.fingerprint"] == job.fingerprint()


def test_fingerprint_ignores_operational_options() -> None:
    base = dib.GenerateJob(model="openai:m", prompt="{t}")
    same = dib.GenerateJob(
        model=dib.models.openai("m", api_key_env="OTHER"),
        prompt="{t}",
        engine_options={"max_concurrency": 3},
    )
    other = dib.GenerateJob(model="openai:m", prompt="Q: {t}")
    assert base.fingerprint() == same.fingerprint()
    assert base.fingerprint() != other.fingerprint()
    assert dib.EmbedJob(model="openai:m", text="{t}").fingerprint() != base.fingerprint()
