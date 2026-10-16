import pickle

import pytest
from pydantic import ValidationError

from distribatch import ConfigError, models
from distribatch.models import FakeModel, OpenAIModel, PathModel, UCModel


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("uc:main.llm.llama@champion", UCModel(name="main.llm.llama", alias="champion")),
        ("uc:main.llm.llama/3", UCModel(name="main.llm.llama", version=3)),
        ("models:/main.llm.llama@prod", UCModel(name="main.llm.llama", alias="prod")),
        ("/Volumes/main/llm/weights/llama", PathModel(path="/Volumes/main/llm/weights/llama")),
        ("dbfs:/models/llama", PathModel(path="dbfs:/models/llama")),
        ("./weights", PathModel(path="./weights")),
        ("openai:gpt-4o-mini", OpenAIModel(model="gpt-4o-mini")),
        ("fake", FakeModel()),
    ],
)
def test_parse(uri: str, expected: object) -> None:
    assert models.parse(uri) == expected


@pytest.mark.parametrize(
    "uri", ["uc:main.llm.llama", "uc:llama@prod", "llama-3-8b", "hf:meta/llama", ""]
)
def test_parse_rejects(uri: str) -> None:
    with pytest.raises((ConfigError, ValidationError)):
        models.parse(uri)


def test_uc_requires_exactly_one_of_version_alias() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        models.uc("a.b.c")
    with pytest.raises(ValidationError, match="exactly one"):
        models.uc("a.b.c", version=1, alias="x")


def test_uc_mlflow_uri() -> None:
    assert models.uc("a.b.c", alias="champion").mlflow_uri == "models:/a.b.c@champion"
    assert models.uc("a.b.c", version=4).mlflow_uri == "models:/a.b.c/4"


def test_specs_are_picklable_and_hashable() -> None:
    spec = models.openai("m", base_url="http://x", api_key_env="KEY")
    assert pickle.loads(pickle.dumps(spec)) == spec
    assert hash(spec) == hash(models.openai("m", base_url="http://x", api_key_env="KEY"))
