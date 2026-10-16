"""The GitHub wiki sources (docs/wiki, private repo only) stay in sync with the code:
generated tables are current, the public API and run() parameters are documented,
Python examples parse and wiki links resolve."""

import ast
import importlib.util
import inspect
import re
from pathlib import Path

import pytest

import distribatch as dib

WIKI = Path(__file__).resolve().parent.parent / "docs" / "wiki"
pytestmark = pytest.mark.skipif(not WIKI.is_dir(), reason="wiki sources not in this checkout")

LINK = re.compile(r"\]\(([A-Za-z][A-Za-z0-9-]*)(?:#([a-z0-9-]+))?\)")
CODE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _pages() -> dict[str, str]:
    return {p.stem: p.read_text() for p in WIKI.glob("*.md")}


def _slug(heading: str) -> str:
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


def test_generated_sections_are_current() -> None:
    spec = importlib.util.spec_from_file_location(
        "gen_wiki", WIKI.parent.parent / "scripts" / "gen_wiki.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stale = [name for name, text in _pages().items() if module.update(text) != text]
    assert not stale, f"run scripts/gen_wiki.py (stale: {stale})"


def test_public_api_is_documented() -> None:
    reference = _pages()["API-Reference"]
    missing = [name for name in dib.__all__ if name not in reference]
    missing += [name for name in dib.models.__all__ if name not in reference]
    assert not missing


def test_run_parameters_are_documented() -> None:
    page = _pages()["Running-Jobs"]
    params = [p for p in inspect.signature(dib.BaseJob.run).parameters if p != "self"]
    assert [p for p in params if f"`{p}`" not in page] == []


@pytest.mark.parametrize("job_cls", [dib.GenerateJob, dib.EmbedJob, dib.TranscribeJob])
def test_job_fields_are_documented(job_cls: type[dib.BaseJob]) -> None:
    page = _pages()[job_cls.__name__]
    assert [f for f in job_cls.model_fields if f not in page] == []


def test_python_examples_parse() -> None:
    for name, text in _pages().items():
        for block in CODE.findall(text):
            code = "\n".join(line for line in block.splitlines() if not line.startswith("%"))
            try:
                ast.parse(code)
            except SyntaxError as exc:
                pytest.fail(f"{name}: {exc}\n{block}")


def test_links_resolve() -> None:
    pages = _pages()
    anchors = {
        name: {_slug(h) for h in re.findall(r"^#+ (.+)$", text, re.MULTILINE)}
        for name, text in pages.items()
    }
    broken = []
    for name, text in pages.items():
        for target, anchor in LINK.findall(text):
            if target not in pages:
                broken.append(f"{name} -> {target}")
            elif anchor and anchor not in anchors[target]:
                broken.append(f"{name} -> {target}#{anchor}")
    assert not broken
