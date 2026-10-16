"""Spark backend on a local[2] session (needs Java; skipped otherwise)."""

import os
from typing import Any

import pytest
from pydantic import BaseModel

import distribatch as dib
from distribatch import ConfigError
from openai_stub import OpenAIStub

pytestmark = pytest.mark.spark


class Review(BaseModel):
    label: str
    score: float
    tags: list[str]
    extra: dict[str, int]


def _rows(df: Any, *cols: str) -> list[tuple[Any, ...]]:
    return [tuple(r[c] for c in cols) for r in df.orderBy("id").collect()]


def test_generate_keeps_input_columns_and_types(spark: Any) -> None:
    df = spark.createDataFrame(
        [(1, "a", b"\x00\x01", [1, 2]), (2, "BAD", b"", [])],
        "id long, t string, blob binary, nums array<int>",
    )
    job = dib.GenerateJob(model=dib.models.fake(fail_marker="BAD"), prompt="say {t}")
    out = job.run(df, backend="spark", output_col="ans")
    assert out.columns == ["id", "t", "blob", "nums", "ans", "ans_error", "ans_meta"]
    assert dict(out.dtypes)["blob"] == "binary"
    assert dict(out.dtypes)["nums"] == "array<int>"
    rows = _rows(out, "id", "blob", "nums", "ans", "ans_error")
    assert rows[0] == (1, bytearray(b"\x00\x01"), [1, 2], "say a", None)
    assert rows[1][3:] == (None, "EngineError: fake failure")
    meta = out.orderBy("id").first()["ans_meta"]
    assert (meta["model_id"], meta["attempts"]) == ("fake", 1)


def test_structured_output_becomes_a_struct(spark: Any) -> None:
    response = '{"label": "pos", "score": 1, "tags": ["a"], "extra": {"k": 2}}'
    job = dib.GenerateJob(
        model=dib.models.fake(response=response), prompt="{t}", output_schema=Review
    )
    out = job.run(spark.createDataFrame([(1, "x")], "id long, t string"), backend="spark")
    assert dict(out.dtypes)["output"] == (
        "struct<label:string,score:double,tags:array<string>,extra:map<string,bigint>>"
    )
    value = out.first()["output"].asDict()
    assert value == {"label": "pos", "score": 1.0, "tags": ["a"], "extra": {"k": 2}}


def test_dict_schema_mismatch_is_a_row_error(spark: Any) -> None:
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
    job = dib.GenerateJob(
        model=dib.models.fake(response='{"n": "not a number"}'), prompt="{t}", output_schema=schema
    )
    row = job.run(spark.createDataFrame([(1, "x")], "id long, t string"), backend="spark").first()
    assert row["output"] is None
    assert row["output_error"] == "OutputTypeError: n: expected an integer, got str"


def test_embeddings_are_float_arrays(spark: Any) -> None:
    df = spark.createDataFrame([(i, f"text {i}") for i in range(20)], "id long, t string")
    out = dib.EmbedJob(model="fake", text="{t}", dimensions=6).run(
        df, backend="spark", partitions=3, batch_size=4
    )
    assert dict(out.dtypes)["output"] == "array<float>"
    vectors = [r["output"] for r in out.orderBy("id").collect()]
    assert len(vectors) == 20
    assert all(len(v) == 6 for v in vectors)
    local = dib.EmbedJob(model="fake", text="{t}", dimensions=6).run([{"t": "text 3"}])
    assert vectors[3] == pytest.approx(local[0]["output"], rel=1e-6)  # float32 round trip


def test_transcribe_binary_column(spark: Any) -> None:
    # The shape of spark.read.format("binaryFile"): path + content.
    df = spark.createDataFrame(
        [("/Volumes/a.wav", b"hello audio"), ("/Volumes/b.wav", None)],
        "path string, content binary",
    )
    out = dib.TranscribeJob(model="fake", audio_col="content", timestamps=True).run(
        df, backend="spark", output_col="tr"
    )
    rows = {r["path"]: r for r in out.collect()}
    good = rows["/Volumes/a.wav"]["tr"].asDict(recursive=True)
    assert good["text"] == "hello audio"
    assert good["segments"] == [{"start": 0.0, "end": 1.0, "text": "hello audio"}]
    assert rows["/Volumes/b.wav"]["tr_error"] == "InvalidInputError: audio is null"


def test_openai_engine_on_spark(spark: Any, openai_stub: OpenAIStub) -> None:
    model = dib.models.openai("m", base_url=openai_stub.base_url, api_key_env="STUB_KEY")
    df = spark.createDataFrame([(i, f"q{i}") for i in range(12)], "id long, t string")
    out = dib.GenerateJob(model=model, prompt="{t}").run(df, backend="spark", partitions=2)
    assert [r[0] for r in _rows(out, "output")] == [f"q{i}" for i in range(12)]


def test_engine_load_error_fails_the_spark_job(spark: Any) -> None:
    model = dib.models.openai("m", base_url="http://127.0.0.1:9", api_key_env="NEVER_SET_KEY")
    out = dib.GenerateJob(model=model, prompt="{t}").run(
        spark.createDataFrame([(1, "x")], "id long, t string"), backend="spark"
    )
    with pytest.raises(Exception, match="NEVER_SET_KEY is not set"):
        out.collect()


def test_config_errors_before_any_spark_job(spark: Any) -> None:
    df = spark.createDataFrame([(1, "x")], "id long, t string")
    with pytest.raises(ConfigError, match="missing"):
        dib.GenerateJob(model="fake", prompt="{nope}").run(df, backend="spark")
    with pytest.raises(ConfigError, match="already exist"):
        dib.GenerateJob(model="fake", prompt="{t}").run(df, backend="spark", output_col="t")
    with pytest.raises(ConfigError, match="need backend='spark'"):
        dib.GenerateJob(model="fake", prompt="{t}").run(df)
    with pytest.raises(ConfigError, match="needs a Spark DataFrame"):
        dib.GenerateJob(model="fake", prompt="{t}").run([{"t": "x"}], backend="spark")
    with pytest.raises(ConfigError, match="partitions"):
        dib.GenerateJob(model="fake", prompt="{t}").run([{"t": "x"}], partitions=2)


def test_empty_dataframe(spark: Any) -> None:
    df = spark.createDataFrame([], "id long, t string")
    assert dib.GenerateJob(model="fake", prompt="{t}").run(df, backend="spark").count() == 0


def test_preview_of_a_spark_dataframe(spark: Any) -> None:
    df = spark.createDataFrame([(i, str(i)) for i in range(10)], "id long, t string")
    out = dib.GenerateJob(model="fake", prompt="{t}").preview(df, n=3)
    assert list(out["output"]) == ["0", "1", "2"]


def test_dedupe_and_summarize_on_spark(spark: Any) -> None:
    df = spark.createDataFrame(
        [(i, "same" if i % 2 else f"u{i}") for i in range(8)], "id long, t string"
    )
    out = dib.GenerateJob(model="fake", prompt="{t}").run(
        df.coalesce(1), backend="spark", dedupe=True
    )
    stats = dib.summarize(out)
    assert (stats.rows, stats.succeeded, stats.reused) == (8, 8, 3)  # 4x "same" -> 1 call
    assert stats.latency_ms_p50 is not None


def test_spark_checkpoint_resume(spark: Any, tmp_path: Any) -> None:
    ckpt = str(tmp_path / "ckpt")
    job = dib.GenerateJob(model=dib.models.fake(fail_marker="BAD"), prompt="{t}")
    first = spark.createDataFrame([(i, f"r{i}") for i in range(5)], "id long, t string")
    out = job.run(first, backend="spark", checkpoint=ckpt, id_col="id", checkpoint_every=2)
    assert sorted(r["output"] for r in out.collect()) == [f"r{i}" for i in range(5)]
    stored = spark.read.parquet(f"{ckpt}/results")
    assert stored.count() == 5

    full = spark.createDataFrame(
        [(i, "BAD" if i == 7 else f"r{i}") for i in range(10)], "id long, t string"
    )
    out = job.run(full, backend="spark", checkpoint=ckpt, id_col="id")
    assert spark.read.parquet(f"{ckpt}/results").count() == 10  # only 5 new rows ran
    rows = {r["id"]: r for r in out.collect()}
    assert out.columns == ["id", "t", "output", "output_error", "output_meta"]
    assert rows[9]["output"] == "r9"
    assert rows[7]["output_error"] == "EngineError: fake failure"

    job.run(full, backend="spark", checkpoint=ckpt, id_col="id")  # retries id 7 only
    assert spark.read.parquet(f"{ckpt}/results").count() == 11
    job.run(full, backend="spark", checkpoint=ckpt, id_col="id", retry_failed=False)
    assert spark.read.parquet(f"{ckpt}/results").count() == 11


def test_spark_checkpoint_validation(spark: Any, tmp_path: Any) -> None:
    ckpt = str(tmp_path / "ckpt")
    df = spark.createDataFrame([(1, "a"), (1, "b")], "id long, t string")
    job = dib.GenerateJob(model="fake", prompt="{t}")
    with pytest.raises(ConfigError, match="duplicate"):
        job.run(df, backend="spark", checkpoint=ckpt, id_col="id")
    with pytest.raises(ConfigError, match="not in the input"):
        job.run(df, backend="spark", checkpoint=ckpt, id_col="nope")
    ok = spark.createDataFrame([(1, "a")], "id long, t string")
    job.run(ok, backend="spark", checkpoint=ckpt, id_col="id")
    with pytest.raises(ConfigError, match="different job"):
        dib.GenerateJob(model="fake", prompt="Q {t}").run(
            ok, backend="spark", checkpoint=ckpt, id_col="id"
        )


def test_gpu_pinning_from_task_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    import pyspark

    from distribatch.executors import spark as spark_executor

    resources = {"gpu": types.SimpleNamespace(addresses=["2", "3"])}
    context = types.SimpleNamespace(resources=lambda: resources)
    monkeypatch.setattr(pyspark.TaskContext, "get", staticmethod(lambda: context))
    monkeypatch.delenv("DISTRIBATCH_GPU_PINNED", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    spark_executor.pin_task_gpus()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"
    resources["gpu"] = types.SimpleNamespace(addresses=["0"])
    spark_executor.pin_task_gpus()  # once per process: the loaded engine stays put
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"
