"""Model sources: Unity Catalog, filesystem/Volume path, OpenAI-compatible API."""

from __future__ import annotations

import re

from distribatch.models.specs import (
    DATABRICKS,
    FakeModel,
    ModelSpec,
    OpenAIModel,
    PathModel,
    UCModel,
)
from distribatch.types import ConfigError

__all__ = [
    "FakeModel",
    "ModelSpec",
    "databricks",
    "OpenAIModel",
    "PathModel",
    "UCModel",
    "fake",
    "openai",
    "parse",
    "path",
    "uc",
]


def uc(
    name: str,
    *,
    version: int | None = None,
    alias: str | None = None,
    stage_dir: str | None = None,
) -> UCModel:
    return UCModel(name=name, version=version, alias=alias, stage_dir=stage_dir)


def path(path: str, *, copy_to_local: bool = False) -> PathModel:
    return PathModel(path=path, copy_to_local=copy_to_local)


def openai(
    model: str, *, base_url: str | None = None, api_key_env: str | None = None
) -> OpenAIModel:
    return OpenAIModel(model=model, base_url=base_url, api_key_env=api_key_env)


def databricks(endpoint: str, *, api_key_env: str | None = None) -> OpenAIModel:
    """A Databricks Model Serving endpoint (Foundation Model APIs or a custom endpoint)
    through its OpenAI-compatible API. Host and token come from the worker environment
    (``DATABRICKS_HOST``/``DATABRICKS_TOKEN`` or databricks-sdk auth), or the token
    from `api_key_env`."""
    return OpenAIModel(model=endpoint, base_url=DATABRICKS, api_key_env=api_key_env)


def fake(
    *,
    response: str | None = None,
    fail_marker: str | None = None,
    crash_marker: str | None = None,
    exit_marker: str | None = None,
    delay_ms: float = 0.0,
    structured_output: bool = True,
) -> FakeModel:
    return FakeModel(
        response=response,
        fail_marker=fail_marker,
        crash_marker=crash_marker,
        exit_marker=exit_marker,
        delay_ms=delay_ms,
        structured_output=structured_output,
    )


_UC_URI = re.compile(r"^(?:uc:|models:/)(?P<name>[^@/]+)(?:@(?P<alias>[^@/]+)|/(?P<version>\d+))?$")
_PATH_PREFIXES = ("/", "./", "../", "~", "dbfs:/", "file:/")


def parse(uri: str) -> ModelSpec:
    """Build a spec from a URI string.

    ``uc:cat.schema.model@alias`` / ``uc:cat.schema.model/3`` (or MLflow-style
    ``models:/...``), ``openai:<model>``, ``databricks:<endpoint>``, ``fake``, or a
    filesystem path (``/Volumes/...``, ``dbfs:/...``, ``./weights``).
    """
    uri = uri.strip()
    if uri.startswith(("uc:", "models:/")):
        m = _UC_URI.match(uri)
        if m is None or (m["alias"] is None and m["version"] is None):
            raise ConfigError(
                f"invalid Unity Catalog model URI {uri!r}; "
                "expected 'uc:catalog.schema.model@alias' or 'uc:catalog.schema.model/<version>'"
            )
        version = int(m["version"]) if m["version"] else None
        return UCModel(name=m["name"], version=version, alias=m["alias"])
    if uri.startswith("openai:"):
        return OpenAIModel(model=uri.removeprefix("openai:"))
    if uri.startswith("databricks:"):
        return databricks(uri.removeprefix("databricks:"))
    if uri == "fake":
        return FakeModel()
    if uri.startswith(_PATH_PREFIXES):
        return PathModel(path=uri)
    raise ConfigError(
        f"unrecognized model URI {uri!r}; use 'uc:...', 'openai:...', 'databricks:...', "
        "'fake' or a path starting with '/', './', '~' or 'dbfs:/'"
    )
