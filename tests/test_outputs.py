import dataclasses
import subprocess
import sys
import types
from datetime import datetime
from enum import Enum

import pytest
from pydantic import BaseModel

import distribatch as dib
from distribatch import outputs
from distribatch._serialize import dumps_job
from distribatch.outputs import (
    BOOL,
    DOUBLE,
    LONG,
    STRING,
    OutputTypeError,
    array,
    coerce,
    from_json_schema,
    map_of,
    struct,
)
from distribatch.types import ResultMeta


class Color(str, Enum):
    red = "red"


class Inner(BaseModel):
    x: int


class Outer(BaseModel):
    name: str
    maybe: float | None = None
    inner: Inner
    inners: list[Inner]
    counts: dict[str, int]
    color: Color
    when: datetime
    mixed: int | str
    flag: bool


class Node(BaseModel):
    value: int
    children: list["Node"] = []


def test_pydantic_schema_mapping() -> None:
    t = from_json_schema(Outer.model_json_schema())
    assert t == struct(
        name=STRING,
        maybe=DOUBLE,
        inner=struct(x=LONG),
        inners=array(struct(x=LONG)),
        counts=map_of(LONG),
        color=STRING,
        when=STRING,
        mixed=STRING,
        flag=BOOL,
    )


def test_recursive_model_terminates() -> None:
    t = from_json_schema(Node.model_json_schema())
    assert t.kind == "struct"
    assert dict(t.fields)["value"] == LONG


def test_free_form_objects_become_json_strings() -> None:
    assert from_json_schema({"type": "object"}) == STRING
    assert from_json_schema({}) == STRING
    assert coerce(STRING, {"a": [1]}) == '{"a": [1]}'


def test_coerce() -> None:
    t = struct(n=LONG, xs=array(DOUBLE), m=map_of(BOOL))
    assert coerce(t, {"n": 1, "xs": [1, 2.5], "m": {"k": True}, "extra": 0}) == {
        "n": 1,
        "xs": [1.0, 2.5],
        "m": [("k", True)],
    }
    assert coerce(t, {"n": None}) == {"n": None, "xs": None, "m": None}
    assert coerce(STRING, datetime(2026, 1, 2)) == '"2026-01-02 00:00:00"'
    for bad_type, bad_value, message in [
        (LONG, True, "expected an integer"),
        (LONG, 1.5, "expected an integer"),
        (DOUBLE, "1", "expected a number"),
        (BOOL, 1, "expected a boolean"),
        (array(LONG), "x", "expected a list"),
        (t, [1], "expected an object"),
    ]:
        with pytest.raises(OutputTypeError, match=message):
            coerce(bad_type, bad_value)
    with pytest.raises(OutputTypeError, match=r"xs\[1\]: expected a number"):
        coerce(t, {"xs": [1.0, "no"]})


def test_meta_type_mirrors_result_meta() -> None:
    assert [name for name, _ in outputs.META.fields] == [
        f.name for f in dataclasses.fields(ResultMeta)
    ]
    assert coerce(outputs.META, ResultMeta(prompt_tokens=3, latency_ms=1.5).to_dict())


def test_arrow_types() -> None:
    import pyarrow as pa

    assert outputs.to_arrow(array(outputs.FLOAT)) == pa.list_(pa.float32())
    assert outputs.to_arrow(outputs.TRANSCRIPT).get_field_index("segments") == 2
    value = coerce(map_of(LONG), {"a": 1})
    assert pa.array([value], type=outputs.to_arrow(map_of(LONG))).to_pylist() == [[("a", 1)]]


def test_job_from_driver_only_module_loads_in_a_fresh_process() -> None:
    # Simulates a schema defined in a repo file the workers cannot import.
    module = types.ModuleType("driver_only_schemas")
    exec("from pydantic import BaseModel\nclass Answer(BaseModel):\n    a: int\n", module.__dict__)
    sys.modules["driver_only_schemas"] = module
    try:
        job = dib.GenerateJob(model="fake", prompt="{t}", output_schema=module.Answer)
        data = dumps_job(job)
    finally:
        del sys.modules["driver_only_schemas"]
    code = (
        "import pickle, sys\n"
        "job = pickle.loads(sys.stdin.buffer.read())\n"
        "print(job.run([{'t': '{\"a\": 5}'}])[0]['output'])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], input=data, capture_output=True, check=True
    ).stdout
    assert out.decode().strip() == "{'a': 5}"


def test_structured_values_are_json_native_on_every_backend() -> None:
    response = (
        '{"name": "n", "inner": {"x": 1}, "inners": [], "counts": {}, "color": "red", '
        '"when": "2026-01-02T03:04:05", "mixed": 7, "flag": true}'
    )
    job = dib.GenerateJob(
        model=dib.models.fake(response=response), prompt="{t}", output_schema=Outer
    )
    value = job.run([{"t": "x"}])[0]["output"]
    assert value["when"] == "2026-01-02T03:04:05"
    assert value["color"] == "red"
    t = from_json_schema(Outer.model_json_schema())
    assert coerce(t, value)["when"] == "2026-01-02T03:04:05"
    assert coerce(t, value)["mixed"] == "7"  # union -> JSON text
