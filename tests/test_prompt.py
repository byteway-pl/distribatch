import json
import math

import pytest
from pydantic import BaseModel

from distribatch import ConfigError, PromptRenderError, SchemaValidationError
from distribatch.prompt import compile_prompt, extract_json, template_fields


class Sentiment(BaseModel):
    label: str
    confidence: float


def test_template_fields_in_first_use_order() -> None:
    assert template_fields("{b} and {a} then {b:>5} {{literal}}") == ("b", "a")


@pytest.mark.parametrize("template", ["{}", "{0}", "{row.text}", "{row[0]}", "{unclosed"])
def test_template_rejects(template: str) -> None:
    with pytest.raises(ConfigError):
        template_fields(template)


def test_render_nulls_as_empty_and_formats_values() -> None:
    p = compile_prompt("[{a}][{b}][{c:.1f}]", None, None)
    assert p.render({"a": None, "b": math.nan, "c": 2.25}) == "[][][2.2]"


def test_render_missing_column_is_row_error() -> None:
    p = compile_prompt("{a}", None, None)
    with pytest.raises(PromptRenderError, match="'a'"):
        p.render({"b": 1})


def test_messages_with_and_without_system() -> None:
    p = compile_prompt("Q: {q}", "Be brief.", None)
    msgs = p.build_messages({"q": "why"}, inline_schema=False)
    assert [(m.role, m.content) for m in msgs] == [("system", "Be brief."), ("user", "Q: why")]
    assert len(compile_prompt("{q}", None, None).build_messages({"q": 1}, inline_schema=True)) == 1


def test_inline_schema_goes_into_system_prompt() -> None:
    p = compile_prompt("{q}", "Be brief.", Sentiment)
    system = p.build_messages({"q": "x"}, inline_schema=True)[0]
    assert system.role == "system"
    assert system.content.startswith("Be brief.\n\n")
    assert json.dumps(Sentiment.model_json_schema()) in system.content
    native = p.build_messages({"q": "x"}, inline_schema=False)[0]
    assert native.content == "Be brief."


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        'Sure! Here it is: {"a": 1} Hope that helps.',
    ],
)
def test_extract_json(text: str) -> None:
    assert json.loads(extract_json(text)) == {"a": 1}


def test_extract_json_fails_without_json() -> None:
    with pytest.raises(SchemaValidationError, match="no valid JSON"):
        extract_json("I cannot answer that.")


def test_parse_output_with_pydantic_schema() -> None:
    p = compile_prompt("{q}", None, Sentiment)
    assert p.parse_output('{"label": "pos", "confidence": 0.9}') == {
        "label": "pos",
        "confidence": 0.9,
    }
    with pytest.raises(SchemaValidationError, match="confidence"):
        p.parse_output('{"label": "pos"}')


def test_parse_output_with_dict_schema_and_text() -> None:
    assert compile_prompt("{q}", None, {"type": "object"}).parse_output('{"x": [1]}') == {"x": [1]}
    assert compile_prompt("{q}", None, None).parse_output("plain") == "plain"


def test_invalid_output_schema_type() -> None:
    with pytest.raises(ConfigError, match="output_schema"):
        compile_prompt("{q}", None, str)  # type: ignore[arg-type]
