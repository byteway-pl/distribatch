"""Prompt templating, chat-message building and structured-output handling.

Templates use ``{column}`` placeholders (``str.format`` syntax, ``{{``/``}}`` for literal
braces, format specs like ``{price:.2f}`` allowed). Only plain column names are valid
fields; attribute/index access and positional fields are rejected up front. The system
prompt is static text (braces in it are literal).
"""

from __future__ import annotations

import json
import string
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from distribatch.types import ChatMessage, ConfigError, PromptRenderError, SchemaValidationError

_FORMATTER = string.Formatter()
_RAW_PREVIEW_CHARS = 200

SCHEMA_INSTRUCTION = (
    "Respond with a single JSON value that conforms to the JSON schema below. "
    "Output only the JSON, without any other text or code fences.\n"
    "JSON schema:\n{schema}"
)


def template_fields(template: str) -> tuple[str, ...]:
    """Column names referenced by `template`, in first-use order (validates syntax)."""
    try:
        parsed = list(_FORMATTER.parse(template))
    except ValueError as exc:
        raise ConfigError(f"invalid prompt template: {exc}") from exc
    fields: list[str] = []
    for _, name, _, _ in parsed:
        if name is None:
            continue
        if name == "" or name.isdigit():
            raise ConfigError(
                "prompt template uses a positional field ('{}' / '{0}'); "
                "reference columns by name, e.g. '{review_text}'"
            )
        if "." in name or "[" in name:
            raise ConfigError(
                f"prompt template field {{{name}}} uses attribute/index access, which is not "
                "supported; reference a column by its plain name (use '{{' / '}}' for "
                "literal braces)"
            )
        if name not in fields:
            fields.append(name)
    return tuple(fields)


def json_schema_of(output_schema: type[BaseModel] | dict[str, Any] | None) -> dict[str, Any] | None:
    if output_schema is None:
        return None
    if isinstance(output_schema, dict):
        return output_schema
    if isinstance(output_schema, type) and issubclass(output_schema, BaseModel):
        return output_schema.model_json_schema()
    raise ConfigError(
        f"output_schema must be a pydantic model class or a JSON-schema dict, "
        f"got {type(output_schema).__name__}"
    )


class _NullAsEmpty(dict[str, Any]):
    """Row view for formatting: nulls (None/NaN/NA) render as an empty string."""

    def __getitem__(self, key: str) -> Any:
        value = super().__getitem__(key)
        return "" if _is_null(value) else value


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    try:
        # NaN and pandas.NA are the only values not equal to themselves.
        return bool(value != value)
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class Template:
    """A validated ``{column}`` template."""

    source: str
    fields: tuple[str, ...]

    @classmethod
    def compile(cls, source: str) -> Template:
        return cls(source=source, fields=template_fields(source))

    def render(self, row: Mapping[str, Any]) -> str:
        try:
            return _FORMATTER.vformat(self.source, (), _NullAsEmpty(row))
        except KeyError as exc:
            raise PromptRenderError(f"row has no column {exc.args[0]!r}") from exc
        except (ValueError, TypeError) as exc:
            raise PromptRenderError(str(exc)) from exc


@dataclass(frozen=True)
class CompiledPrompt:
    template: Template
    system: str | None
    json_schema: dict[str, Any] | None
    schema_model: type[BaseModel] | None

    @property
    def fields(self) -> tuple[str, ...]:
        return self.template.fields

    def render(self, row: Mapping[str, Any]) -> str:
        return self.template.render(row)

    def build_messages(
        self, row: Mapping[str, Any], *, inline_schema: bool
    ) -> tuple[ChatMessage, ...]:
        """Chat messages for one row. With `inline_schema`, the JSON schema is spelled
        out in the system prompt (for engines without native structured output)."""
        user = ChatMessage(role="user", content=self.render(row))
        messages = (
            (ChatMessage(role="system", content=self.system), user) if self.system else (user,)
        )
        if inline_schema and self.json_schema is not None:
            return with_inline_schema(messages, self.json_schema)
        return messages

    def parse_output(self, text: str) -> Any:
        """Text output as-is, or the parsed + validated structured output."""
        if self.json_schema is None:
            return text
        json_text = extract_json(text)
        if self.schema_model is not None:
            try:
                return self.schema_model.model_validate_json(json_text).model_dump(mode="json")
            except ValidationError as exc:
                problems = "; ".join(
                    f"{'.'.join(map(str, e['loc'])) or '<root>'}: {e['msg']}" for e in exc.errors()
                )
                raise SchemaValidationError(f"{problems}; raw output: {_preview(text)}") from exc
        # A dict schema is passed to the engine but only checked for being valid JSON.
        return json.loads(json_text)


def with_inline_schema(
    messages: tuple[ChatMessage, ...], json_schema: dict[str, Any]
) -> tuple[ChatMessage, ...]:
    """`messages` with the JSON-schema instruction appended to the system prompt (one is
    added if missing). Used by jobs and by engines falling back from native schemas."""
    instruction = SCHEMA_INSTRUCTION.format(schema=json.dumps(json_schema))
    if messages and messages[0].role == "system":
        system = ChatMessage(role="system", content=f"{messages[0].content}\n\n{instruction}")
        return (system, *messages[1:])
    return (ChatMessage(role="system", content=instruction), *messages)


def compile_prompt(
    template: str,
    system: str | None,
    output_schema: type[BaseModel] | dict[str, Any] | None,
) -> CompiledPrompt:
    schema_model = output_schema if isinstance(output_schema, type) else None
    return CompiledPrompt(
        template=Template.compile(template),
        system=system,
        json_schema=json_schema_of(output_schema),
        schema_model=schema_model,
    )


def extract_json(text: str) -> str:
    """The JSON part of a model response: tolerates code fences and surrounding prose."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    if _is_json(stripped):
        return stripped
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = stripped.find(opener), stripped.rfind(closer)
        if start != -1 and end > start and _is_json(stripped[start : end + 1]):
            return stripped[start : end + 1]
    raise SchemaValidationError(f"no valid JSON found in output: {_preview(text)}")


def _is_json(text: str) -> bool:
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def _preview(text: str) -> str:
    if len(text) <= _RAW_PREVIEW_CHARS:
        return repr(text)
    return repr(text[:_RAW_PREVIEW_CHARS]) + "…"
