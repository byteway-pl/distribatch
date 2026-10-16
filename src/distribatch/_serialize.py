"""Job serialization for worker processes (multiprocessing and Spark).

`output_schema` classes often live in modules the workers cannot import: a notebook's
`__main__`, a repo file on the driver only, a test module. cloudpickle stores
`__main__` classes by value but others by reference, which fails on the workers. So
the modules of the schema classes are pickled by value too, for this one dump.
"""

from __future__ import annotations

import pickle
import sys
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from distribatch.job import BaseJob

# Modules workers always have; never pickled by value.
_INSTALLED_PREFIXES = ("distribatch", "pydantic", "builtins", "typing", "enum", "datetime")


def dumps_job(job: BaseJob) -> bytes:
    import cloudpickle

    modules = _by_value_modules(getattr(job, "output_schema", None))
    registered = []
    for module in modules:
        try:
            cloudpickle.register_pickle_by_value(module)
            registered.append(module)
        except ValueError:  # already registered by the user
            pass
    try:
        return bytes(cloudpickle.dumps(job))
    finally:
        for module in registered:
            cloudpickle.unregister_pickle_by_value(module)


def loads_job(data: bytes) -> BaseJob:
    job: BaseJob = pickle.loads(data)
    return job


def _by_value_modules(schema: Any) -> list[Any]:
    """Modules defining `schema` and the pydantic models nested in its fields."""
    found: dict[str, Any] = {}
    seen: set[type] = set()

    def visit(cls: Any) -> None:
        if not (isinstance(cls, type) and issubclass(cls, BaseModel)) or cls in seen:
            return
        seen.add(cls)
        name = cls.__module__
        if name != "__main__" and not name.startswith(_INSTALLED_PREFIXES):
            module = sys.modules.get(name)
            if module is not None:
                found[name] = module
        for field in cls.model_fields.values():
            for arg in _types_in(field.annotation):
                visit(arg)

    visit(schema)
    return list(found.values())


def _types_in(annotation: Any) -> list[Any]:
    """Classes appearing in an annotation (list[X], X | None, dict[str, X], ...)."""
    out = [annotation]
    for arg in getattr(annotation, "__args__", ()) or ():
        out.extend(_types_in(arg))
    return out
