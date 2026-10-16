"""Model source specifications.

Specs are pure, picklable data: they travel to Spark/pool workers, which resolve
them (download weights, read credentials from env) on their side.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_UC_NAME = re.compile(r"^[^.\s]+\.[^.\s]+\.[^.\s]+$")


class _Spec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class UCModel(_Spec):
    """A Unity Catalog registered model (``catalog.schema.model``), by version or alias.

    Workers download it with MLflow into the node-local cache. With `stage_dir` (e.g. a
    UC Volume directory), the Spark backend downloads it once on the driver instead and
    executors copy it from there: no registry credentials needed on executors.
    """

    kind: Literal["uc"] = "uc"
    name: str
    version: int | None = Field(default=None, gt=0)
    alias: str | None = None
    stage_dir: str | None = None

    @field_validator("name")
    @classmethod
    def _three_part_name(cls, v: str) -> str:
        if not _UC_NAME.match(v):
            raise ValueError(f"expected a three-part name 'catalog.schema.model', got {v!r}")
        return v

    @model_validator(mode="after")
    def _version_xor_alias(self) -> UCModel:
        if (self.version is None) == (self.alias is None):
            raise ValueError("specify exactly one of `version` or `alias`")
        return self

    @property
    def mlflow_uri(self) -> str:
        if self.alias is not None:
            return f"models:/{self.name}@{self.alias}"
        return f"models:/{self.name}/{self.version}"


class PathModel(_Spec):
    """HF-format weights in a directory: local path, ``/Volumes/...`` or ``dbfs:/...``."""

    kind: Literal["path"] = "path"
    path: str = Field(min_length=1)
    copy_to_local: bool = False


DATABRICKS = "databricks"


class OpenAIModel(_Spec):
    """A model behind an OpenAI-compatible HTTP API.

    `base_url`: None = the official OpenAI API (or ``OPENAI_BASE_URL``); ``"databricks"``
    = this workspace's Model Serving (``<host>/serving-endpoints``, host and token
    resolved on the worker); otherwise an explicit ``http(s)://`` URL.

    Credentials are never stored in the spec: `api_key_env` names the environment
    variable the worker reads the key from.
    """

    kind: Literal["openai"] = "openai"
    model: str = Field(min_length=1)
    base_url: str | None = None
    api_key_env: str | None = None

    @field_validator("base_url")
    @classmethod
    def _valid_base_url(cls, v: str | None) -> str | None:
        if v is None or v == DATABRICKS or v.startswith(("http://", "https://")):
            return v
        raise ValueError(f"base_url must be None, 'databricks' or an http(s) URL, got {v!r}")


class FakeModel(_Spec):
    """Deterministic stand-in model for tests and dry runs (no weights, no network).

    Echoes the input, or returns `response` when set. Rows whose input contains
    `fail_marker` get a row-level error; a batch containing `crash_marker` makes the
    engine call raise; a batch containing `exit_marker` kills the worker process
    (simulated OOM/segfault; raises instead when running in the main process).
    """

    kind: Literal["fake"] = "fake"
    response: str | None = None
    fail_marker: str | None = None
    crash_marker: str | None = None
    exit_marker: str | None = None
    delay_ms: float = Field(default=0.0, ge=0.0)
    structured_output: bool = True


ModelSpec = Annotated[UCModel | PathModel | OpenAIModel | FakeModel, Field(discriminator="kind")]
