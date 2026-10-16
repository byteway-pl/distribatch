import sys
import threading
import types
from pathlib import Path
from typing import Any

import pytest

import distribatch as dib
from distribatch import ConfigError
from distribatch.models import _cache
from distribatch.models.resolve import fuse_path, locate, resolve_local, stage_uc_model


def _hf_dir(path: Path, *, tokenizer: bool = True) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"w")
    if tokenizer:
        (path / "tokenizer.json").write_text("{}")
    return path


@pytest.fixture(autouse=True)
def _cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache = tmp_path / "cache"
    monkeypatch.setenv("DISTRIBATCH_CACHE_DIR", str(cache))
    return cache


def test_fuse_paths() -> None:
    assert fuse_path("dbfs:/models/x") == "/dbfs/models/x"
    assert fuse_path("dbfs:/Volumes/main/raw/a.wav") == "/Volumes/main/raw/a.wav"
    assert fuse_path("file:/tmp/x") == "/tmp/x"
    assert fuse_path("/Volumes/a/b") == "/Volumes/a/b"


def test_plain_directory(tmp_path: Path) -> None:
    model = _hf_dir(tmp_path / "m")
    local = resolve_local(dib.models.path(str(model)))
    assert (local.path, local.tokenizer, local.processor) == (str(model),) * 3
    assert local.source == str(model)


def test_missing_directory() -> None:
    with pytest.raises(ConfigError, match="does not exist"):
        resolve_local(dib.models.path("/nope/never"))


def test_copy_to_local_once_and_on_change(tmp_path: Path, _cache_dir: Path) -> None:
    model = _hf_dir(tmp_path / "volume" / "m")
    spec = dib.models.path(str(model), copy_to_local=True)
    first = resolve_local(spec)
    assert Path(first.path).parent == _cache_dir
    assert (Path(first.path) / "model.safetensors").read_bytes() == b"w"
    assert resolve_local(spec).path == first.path  # cached
    (model / "model.safetensors").write_bytes(b"new weights")
    assert resolve_local(spec).path != first.path  # weights changed -> new copy


def test_materialize_fills_once_under_contention(tmp_path: Path) -> None:
    calls: list[int] = []

    def fill(tmp: Path) -> None:
        calls.append(1)
        (tmp / "f").write_text("x")

    threads = [
        threading.Thread(target=_cache.materialize, args=("entry", fill, tmp_path))
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert calls == [1]
    assert (tmp_path / "entry" / "f").read_text() == "x"


def test_failed_fill_leaves_no_entry(tmp_path: Path) -> None:
    def boom(tmp: Path) -> None:
        (tmp / "partial").write_text("x")
        raise RuntimeError("download failed")

    with pytest.raises(RuntimeError):
        _cache.materialize("entry", boom, tmp_path)
    assert not (tmp_path / "entry").exists()
    assert not any(p.name.startswith(".entry.tmp") for p in tmp_path.iterdir())


def test_mlflow_transformers_layout(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _hf_dir(root / "model", tokenizer=False)
    (root / "components" / "tokenizer").mkdir(parents=True)
    (root / "components" / "tokenizer" / "tokenizer.json").write_text("{}")
    (root / "components" / "feature_extractor").mkdir(parents=True)
    (root / "components" / "feature_extractor" / "preprocessor_config.json").write_text("{}")
    (root / "MLmodel").write_text("flavors:\n  transformers: {}\n")
    local = locate(root, source="x")
    assert local.path == str(root / "model")
    assert local.tokenizer == str(root / "components" / "tokenizer")
    assert local.processor == str(root / "components" / "feature_extractor")


def test_mlflow_reference_only_model(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "MLmodel").write_text(
        "flavors:\n  transformers:\n    source_model_name: openai/whisper-small\n"
        "    source_model_revision: abc123\n"
    )
    local = locate(root, source="x")
    assert (local.path, local.revision) == ("openai/whisper-small", "abc123")


def test_no_model_found(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(ConfigError, match="no Hugging Face model"):
        locate(tmp_path / "empty", source="x")


class _FakeMlflow(types.ModuleType):
    """Just enough of mlflow for UC resolution."""

    def __init__(self, artifact_builder: Any) -> None:
        super().__init__("mlflow")
        self.registry_uri: str | None = None
        self.downloads: list[str] = []
        builder = artifact_builder
        fake = self

        class Client:
            def get_model_version_by_alias(self, name: str, alias: str) -> Any:
                return types.SimpleNamespace(version="7")

        def download_artifacts(artifact_uri: str, dst_path: str) -> str:
            fake.downloads.append(artifact_uri)
            builder(Path(dst_path))
            return dst_path

        self.MlflowClient = Client
        self.artifacts = types.SimpleNamespace(download_artifacts=download_artifacts)

    def set_registry_uri(self, uri: str) -> None:
        self.registry_uri = uri


def test_uc_model_by_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeMlflow(lambda dst: _hf_dir(dst / "model"))
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    spec = dib.models.uc("main.llm.tiny", alias="champion")
    local = resolve_local(spec)
    assert fake.registry_uri == "databricks-uc"
    assert fake.downloads == ["models:/main.llm.tiny/7"]
    assert local.path.endswith("model")
    assert local.source == "main.llm.tiny/7"
    resolve_local(spec)
    assert len(fake.downloads) == 1  # node cache hit


def test_uc_registry_failure_is_a_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeMlflow(lambda dst: None)

    class Broken:
        def get_model_version_by_alias(self, name: str, alias: str) -> Any:
            raise RuntimeError("401 unauthorized")

    fake.MlflowClient = Broken
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    with pytest.raises(ConfigError, match="stage_dir"):
        resolve_local(dib.models.uc("main.llm.tiny", alias="x"))


def test_stage_uc_model_on_driver(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = _FakeMlflow(lambda dst: _hf_dir(dst / "model"))
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    staged = stage_uc_model(
        dib.models.uc("main.llm.tiny", version=3, stage_dir=str(tmp_path / "volume"))
    )
    assert staged.copy_to_local
    assert staged.path.startswith(str(tmp_path / "volume"))
    assert fake.downloads == ["models:/main.llm.tiny/3"]
