import math

import pandas as pd
import pytest

import distribatch as dib
from distribatch import ConfigError


def test_embed_records() -> None:
    job = dib.EmbedJob(model="fake", text="{title}: {body}")
    out = job.run([{"title": "a", "body": "x"}, {"title": "a", "body": "x"}, {"title": "b"}])
    v0, v1 = out[0]["output"], out[1]["output"]
    assert isinstance(v0, list) and len(v0) == 8
    assert all(isinstance(x, float) for x in v0)
    assert v0 == v1  # deterministic for identical text
    assert out[2]["output_error"].startswith("PromptRenderError")


def test_embed_dimensions_and_normalize() -> None:
    job = dib.EmbedJob(model="fake", text="{t}", dimensions=5, normalize=True)
    vec = job.run(pd.DataFrame({"t": ["hello"]}))["output"].iloc[0]
    assert len(vec) == 5
    assert math.isclose(math.sqrt(sum(v * v for v in vec)), 1.0)


def test_empty_text_is_row_error() -> None:
    out = dib.EmbedJob(model="fake", text="{t}").run([{"t": "  "}, {"t": None}, {"t": "ok"}])
    assert out[0]["output_error"] == "InvalidInputError: text to embed is empty"
    assert out[1]["output_error"] == "InvalidInputError: text to embed is empty"
    assert out[2]["output_error"] is None
    assert out[0]["output_meta"]["attempts"] == 0


@pytest.mark.parametrize(
    ("output", "error"),
    [
        ([], "non-empty vector"),
        ("nope", "non-empty vector"),
        ([1.0, "x"], "non-numeric"),
        ([1.0, math.nan], "NaN"),
    ],
)
def test_invalid_engine_vectors(output: object, error: str) -> None:
    job = dib.EmbedJob(model="fake", text="{t}")
    with pytest.raises(dib.EngineError, match=error):
        job.finish(output)


def test_wrong_dimension_count() -> None:
    job = dib.EmbedJob(model="fake", text="{t}", dimensions=3)
    with pytest.raises(dib.EngineError, match="expected 3 dimensions"):
        job.finish([1.0, 2.0])


def test_embed_config_errors() -> None:
    with pytest.raises(ConfigError, match="missing"):
        dib.EmbedJob(model="fake", text="{body}").run([{"title": "a"}])
    with pytest.raises(ConfigError, match="does not support task 'embed'"):
        dib.EmbedJob(model="/w", text="{t}", engine="faster_whisper")
