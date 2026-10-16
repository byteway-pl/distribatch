"""Multiprocessing backend. Spawned workers are slow to start (~1 s), so the tests
use few workers and cover several behaviours per run where it stays readable."""

import os
import pickle
from typing import Any

import pandas as pd
import pytest
from pydantic import BaseModel

import distribatch as dib
from distribatch import ConfigError
from distribatch.engines.fake import FakeEngine
from distribatch.executors import process_pool
from openai_stub import OpenAIStub


def test_results_in_order_and_engine_loaded_in_workers() -> None:
    df = pd.DataFrame({"t": [f"row {i}" for i in range(23)]}, index=range(100, 123))
    job = dib.GenerateJob(model=dib.models.fake(fail_marker="row 7"), prompt="{t}")
    out = job.run(df, backend="multiprocessing", workers=2, batch_size=4)
    assert list(out.index) == list(range(100, 123))
    assert out["output"].iloc[0] == "row 0"
    assert out["output"].iloc[22] == "row 22"
    assert out["output_error"].iloc[7] == "EngineError: fake failure"
    assert out["output_error"].drop(index=107).isna().all()
    assert FakeEngine.load_count == 0  # loaded in the workers, not in this process


def test_all_tasks_and_notebook_defined_schema() -> None:
    # A class local to the test is not importable by workers: proves the job is
    # shipped with cloudpickle (as notebook-defined classes in __main__ must be).
    class Local(BaseModel):
        label: str

    gen = dib.GenerateJob(
        model=dib.models.fake(response='{"label": "x"}'), prompt="{t}", output_schema=Local
    )
    assert gen.run([{"t": "a"}] * 3, backend="multiprocessing", workers=2)[2]["output"] == {
        "label": "x"
    }
    emb = dib.EmbedJob(model="fake", text="{t}", normalize=True)
    local = emb.run([{"t": "same"}])[0]["output"]
    assert emb.run([{"t": "same"}], backend="multiprocessing")[0]["output"] == local
    asr = dib.TranscribeJob(model="fake", audio_col="a", timestamps=True)
    row = asr.run([{"a": b"bytes travel"}], backend="multiprocessing")[0]
    assert row["output"]["text"] == "bytes travel"


def test_dead_worker_is_bisected_to_the_poison_row() -> None:
    rows = [{"t": f"ok {i}"} for i in range(12)]
    rows[5] = {"t": "EXIT here"}
    job = dib.GenerateJob(model=dib.models.fake(exit_marker="EXIT"), prompt="{t}")
    out = job.run(rows, backend="multiprocessing", workers=2, batch_size=4)
    errors = [r["output_error"] for r in out]
    assert errors[5] == "WorkerCrashedError: worker process died on this row"
    assert errors[:5] + errors[6:] == [None] * 11
    assert [r["output"] for r in out][6] == "ok 6"


def test_restart_limit_marks_remaining_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(process_pool, "MAX_POOL_RESTARTS", 0)
    rows = [{"t": "EXIT"}] + [{"t": f"ok {i}"} for i in range(5)]
    job = dib.GenerateJob(model=dib.models.fake(exit_marker="EXIT"), prompt="{t}")
    out = job.run(rows, backend="multiprocessing", workers=1, batch_size=2)
    errors = [r["output_error"] for r in out]
    assert all(e is not None and "restart limit" in e for e in errors)


def test_engine_load_error_aborts_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOPE_KEY", raising=False)
    job = dib.GenerateJob(
        model=dib.models.openai("m", base_url="http://127.0.0.1:9", api_key_env="NOPE_KEY"),
        prompt="{t}",
    )
    with pytest.raises(ConfigError, match="NOPE_KEY is not set"):
        job.run([{"t": "x"}], backend="multiprocessing", workers=1)


def test_openai_engine_in_workers(openai_stub: OpenAIStub) -> None:
    model = dib.models.openai("m", base_url=openai_stub.base_url, api_key_env="STUB_KEY")
    out = dib.GenerateJob(model=model, prompt="{t}").run(
        [{"t": str(i)} for i in range(10)], backend="multiprocessing", workers=2, batch_size=3
    )
    assert [r["output"] for r in out] == [str(i) for i in range(10)]


def test_empty_input_starts_no_pool() -> None:
    assert dib.GenerateJob(model="fake", prompt="{t}").run([], backend="multiprocessing") == []


def test_workers_validation() -> None:
    job = dib.GenerateJob(model="fake", prompt="{t}")
    with pytest.raises(ConfigError, match="only applies"):
        job.run([{"t": "x"}], workers=2)
    with pytest.raises(ConfigError, match=">= 1"):
        job.run([{"t": "x"}], backend="multiprocessing", workers=0)


@pytest.mark.parametrize(
    ("env", "tp", "expected"),
    [
        ("0,1,2,3", 1, (4, ["0", "1", "2", "3"])),
        ("0,1,2,3", 2, (2, ["0,1", "2,3"])),
        ("4,6,7", 2, (1, ["4,6"])),
        ("", 1, (1, [])),
        ("-1", 1, (1, [])),
    ],
)
def test_plan_workers_for_local_models(
    monkeypatch: pytest.MonkeyPatch, env: str, tp: int, expected: tuple[int, list[str]]
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", env)
    job = dib.GenerateJob(model="/w", prompt="{t}", engine_options={"tensor_parallel_size": tp})
    assert process_pool.plan_workers(job, None) == expected
    assert process_pool.plan_workers(job, 7)[0] == 7


def test_plan_workers_rejects_oversized_tensor_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    job = dib.GenerateJob(model="/w", prompt="{t}", engine_options={"tensor_parallel_size": 4})
    with pytest.raises(ConfigError, match="tensor_parallel_size=4"):
        process_pool.plan_workers(job, None)


def test_plan_workers_for_api_models() -> None:
    job = dib.GenerateJob(model="fake", prompt="{t}")
    n, slots = process_pool.plan_workers(job, None)
    assert 1 <= n <= process_pool.DEFAULT_CPU_WORKERS
    assert slots == []


def test_worker_initializer_pins_gpu_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    import cloudpickle

    class OneSlot:
        def get_nowait(self) -> str:
            return "2,3"

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setattr(process_pool, "_WORKER", {})
    job = dib.GenerateJob(model="fake", prompt="{t}")
    process_pool._init_worker(cloudpickle.dumps(job), OneSlot())
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert "error" not in process_pool._WORKER
    wire: Any = process_pool._run_chunk([{"t": "hi"}], [0], 4)
    assert wire[0][:3] == (0, "hi", None)


def test_missing_dependency_error_pickles() -> None:
    err = dib.MissingDependencyError("vllm", "vllm")
    clone = pickle.loads(pickle.dumps(err))
    assert (clone.module, clone.extra, str(clone)) == ("vllm", "vllm", str(err))
