"""Typed description of output values, mapped to Spark and Arrow types.

Jobs describe their output value with an `OutType`; the Spark executor derives the
column schema from it on the driver (before any work starts) and uses `coerce` on the
workers to make every value fit, turning mismatches into row errors.

Deliberately small: string, long, double, float, bool, array, map, struct. Dates and
datetimes become ISO strings (no session-timezone surprises); anything a JSON schema
can't pin to one of these (mixed unions, free-form objects) becomes a JSON string.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from distribatch.types import DistribatchError

Kind = Literal["string", "long", "double", "float", "bool", "array", "map", "struct"]


class OutputTypeError(DistribatchError):
    """A value does not fit the declared output type (row-level)."""


@dataclass(frozen=True)
class OutType:
    kind: Kind
    item: OutType | None = None  # array items / map values
    fields: tuple[tuple[str, OutType], ...] = ()  # struct fields, in order


STRING = OutType("string")
LONG = OutType("long")
DOUBLE = OutType("double")
FLOAT = OutType("float")
BOOL = OutType("bool")


def array(item: OutType) -> OutType:
    return OutType("array", item=item)


def map_of(value: OutType) -> OutType:
    return OutType("map", item=value)


def struct(**fields: OutType) -> OutType:
    return OutType("struct", fields=tuple(fields.items()))


# Field order and names mirror types.ResultMeta (enforced by a test).
META = struct(
    prompt_tokens=LONG,
    completion_tokens=LONG,
    audio_seconds=DOUBLE,
    latency_ms=DOUBLE,
    attempts=LONG,
    model_id=STRING,
    finish_reason=STRING,
)

TRANSCRIPT = struct(
    text=STRING,
    language=STRING,
    segments=array(struct(start=DOUBLE, end=DOUBLE, text=STRING)),
)


# --- JSON schema -> OutType ------------------------------------------------------------


def from_json_schema(schema: Mapping[str, Any]) -> OutType:
    """Best-fitting OutType for a JSON schema (as produced by pydantic)."""
    return _from_schema(schema, schema.get("$defs", {}), depth=0)


def _from_schema(schema: Mapping[str, Any], defs: Mapping[str, Any], depth: int) -> OutType:
    if depth > 32:  # recursive models: stop and keep the rest as JSON text
        return STRING
    if "$ref" in schema:
        ref = str(schema["$ref"])
        target = defs.get(ref.rsplit("/", 1)[-1]) if ref.startswith("#/$defs/") else None
        return _from_schema(target, defs, depth + 1) if target is not None else STRING
    for key in ("anyOf", "oneOf"):
        if key in schema:
            variants = [v for v in schema[key] if v.get("type") != "null"]
            return _from_schema(variants[0], defs, depth + 1) if len(variants) == 1 else STRING
    if "allOf" in schema and len(schema["allOf"]) == 1:
        return _from_schema(schema["allOf"][0], defs, depth + 1)

    kind = schema.get("type")
    if isinstance(kind, list):
        non_null = [k for k in kind if k != "null"]
        kind = non_null[0] if len(non_null) == 1 else None
    if kind == "object":
        properties = schema.get("properties")
        if properties:
            return OutType(
                "struct",
                fields=tuple(
                    (name, _from_schema(sub, defs, depth + 1)) for name, sub in properties.items()
                ),
            )
        extra = schema.get("additionalProperties")
        if isinstance(extra, Mapping):
            return map_of(_from_schema(extra, defs, depth + 1))
        return STRING
    if kind == "array":
        items = schema.get("items")
        return array(_from_schema(items, defs, depth + 1) if isinstance(items, Mapping) else STRING)
    return {"integer": LONG, "number": DOUBLE, "boolean": BOOL}.get(str(kind), STRING)


# --- values ---------------------------------------------------------------------------


def coerce(t: OutType, value: Any, path: str = "") -> Any:
    """`value` shaped for `t` (Arrow-ready python), or OutputTypeError."""
    if value is None:
        return None
    where = path or "<value>"
    if t.kind == "string":
        return value if isinstance(value, str) else json.dumps(value, default=str)
    if t.kind == "long":
        if isinstance(value, bool) or not isinstance(value, int):
            raise OutputTypeError(f"{where}: expected an integer, got {type(value).__name__}")
        return value
    if t.kind in ("double", "float"):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise OutputTypeError(f"{where}: expected a number, got {type(value).__name__}")
        return float(value)
    if t.kind == "bool":
        if not isinstance(value, bool):
            raise OutputTypeError(f"{where}: expected a boolean, got {type(value).__name__}")
        return value
    if t.kind == "array":
        assert t.item is not None
        if not isinstance(value, list | tuple):
            raise OutputTypeError(f"{where}: expected a list, got {type(value).__name__}")
        return [coerce(t.item, v, f"{where}[{i}]") for i, v in enumerate(value)]
    if t.kind == "map":
        assert t.item is not None
        if not isinstance(value, Mapping):
            raise OutputTypeError(f"{where}: expected an object, got {type(value).__name__}")
        return [(str(k), coerce(t.item, v, f"{where}.{k}")) for k, v in value.items()]
    if not isinstance(value, Mapping):
        raise OutputTypeError(f"{where}: expected an object, got {type(value).__name__}")
    return {
        name: coerce(sub, value.get(name), f"{path}.{name}" if path else name)
        for name, sub in t.fields
    }


# --- engines' type systems (imported lazily) ------------------------------------------


def to_spark(t: OutType) -> Any:
    from distribatch._deps import require

    T = require("pyspark.sql.types", "spark")
    scalars = {
        "string": T.StringType,
        "long": T.LongType,
        "double": T.DoubleType,
        "float": T.FloatType,
        "bool": T.BooleanType,
    }
    if t.kind in scalars:
        return scalars[t.kind]()
    if t.kind == "array":
        assert t.item is not None
        return T.ArrayType(to_spark(t.item), containsNull=True)
    if t.kind == "map":
        assert t.item is not None
        return T.MapType(T.StringType(), to_spark(t.item), valueContainsNull=True)
    return T.StructType([T.StructField(name, to_spark(sub), True) for name, sub in t.fields])


def to_arrow(t: OutType) -> Any:
    import pyarrow as pa

    scalars = {
        "string": pa.string(),
        "long": pa.int64(),
        "double": pa.float64(),
        "float": pa.float32(),
        "bool": pa.bool_(),
    }
    if t.kind in scalars:
        return scalars[t.kind]
    if t.kind == "array":
        assert t.item is not None
        return pa.list_(to_arrow(t.item))
    if t.kind == "map":
        assert t.item is not None
        return pa.map_(pa.string(), to_arrow(t.item))
    return pa.struct([pa.field(name, to_arrow(sub)) for name, sub in t.fields])
