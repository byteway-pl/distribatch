"""Examples stay syntactically valid and use only the public API."""

import ast
from pathlib import Path

import pytest

import distribatch as dib

EXAMPLES = sorted((Path(__file__).resolve().parent.parent / "examples").glob("*.py"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_parses_and_uses_public_api(path: Path) -> None:
    tree = ast.parse(path.read_text())
    used = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "dib"
    }
    assert used <= set(dib.__all__), used - set(dib.__all__)


def test_local_quickstart_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import runpy

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(EXAMPLES[-1]), run_name="__main__")
