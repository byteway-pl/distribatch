"""transformers engine on CPU with tiny random models (needs torch; downloads ~60 MB
into the repo's .cache/ on first run)."""

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

import distribatch as dib
from distribatch import ConfigError

pytestmark = pytest.mark.filterwarnings("ignore")


def _gen(model: Path, **kw: Any) -> dib.GenerateJob:
    return dib.GenerateJob(
        model=dib.models.path(str(model)),
        engine="transformers",
        sampling=dib.Sampling(max_tokens=5, seed=0),
        **kw,
    )


def test_generate_without_chat_template(tiny_models: dict[str, Path]) -> None:
    out = _gen(tiny_models["llm"], prompt="Hello {t}", system="sys").run(
        [{"t": "world"}, {"t": "there"}, {"t": "again"}], batch_size=2
    )
    for row in out:
        assert row["output_error"] is None
        assert isinstance(row["output"], str)
        meta = row["output_meta"]
        assert meta["prompt_tokens"] > 0
        assert 0 < meta["completion_tokens"] <= 5
        assert meta["finish_reason"] in ("stop", "length")
        assert meta["model_id"] == str(tiny_models["llm"])


def test_generate_is_deterministic_greedy(tiny_models: dict[str, Path]) -> None:
    job = _gen(tiny_models["llm"], prompt="{t}")
    first = job.run([{"t": "same prompt"}])[0]["output"]
    assert job.run([{"t": "same prompt"}])[0]["output"] == first


def test_generate_with_chat_template(tiny_models: dict[str, Path], tmp_path: Path) -> None:
    model = tmp_path / "templated"
    shutil.copytree(tiny_models["llm"], model)
    config = json.loads((model / "tokenizer_config.json").read_text())
    config["chat_template"] = (
        "{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}{% endfor %}<assistant>"
    )
    (model / "tokenizer_config.json").write_text(json.dumps(config))
    row = _gen(model, prompt="{t}").run([{"t": "hi"}])[0]
    assert row["output_error"] is None


def test_structured_output_is_prompted(tiny_models: dict[str, Path]) -> None:
    class Answer(BaseModel):
        a: int

    row = _gen(tiny_models["llm"], prompt="{t}", output_schema=Answer).run([{"t": "x"}])[0]
    # A random model won't produce valid JSON: the failure must be a row error.
    assert row["output"] is None
    assert row["output_error"].startswith("SchemaValidationError")
    assert row["output_meta"]["prompt_tokens"] > 20  # schema went into the system prompt


def test_embeddings_follow_sentence_transformers_config(tiny_models: dict[str, Path]) -> None:
    model = dib.models.path(str(tiny_models["embedder"]))
    job = dib.EmbedJob(model=model, text="{t}", engine="transformers")
    out = job.run([{"t": "a cat"}, {"t": "a dog"}, {"t": "a cat"}], batch_size=2)
    vectors = [row["output"] for row in out]
    assert len(vectors[0]) == 128
    assert vectors[0] == pytest.approx(vectors[2], abs=1e-5)  # batch-independent
    assert vectors[0] != vectors[1]
    norm = math.sqrt(sum(v * v for v in vectors[0]))
    unit = dib.EmbedJob(model=model, text="{t}", engine="transformers", normalize=True)
    unit_vector = unit.run([{"t": "a cat"}])[0]["output"]
    assert unit_vector == pytest.approx([v / norm for v in vectors[0]], abs=1e-5)


def test_pooling_override_changes_vectors(tiny_models: dict[str, Path]) -> None:
    model = dib.models.path(str(tiny_models["embedder"]))
    mean = dib.EmbedJob(model=model, text="{t}", engine="transformers")
    cls = dib.EmbedJob(
        model=model, text="{t}", engine="transformers", engine_options={"pooling": "cls"}
    )
    assert mean.run([{"t": "x y z"}])[0]["output"] != cls.run([{"t": "x y z"}])[0]["output"]


def test_unsupported_sentence_transformers_module(
    tiny_models: dict[str, Path], tmp_path: Path
) -> None:
    model = tmp_path / "dense"
    shutil.copytree(tiny_models["embedder"], model)
    modules = json.loads((model / "modules.json").read_text())
    modules.append(
        {"idx": 2, "name": "2", "path": "2_Dense", "type": "sentence_transformers.models.Dense"}
    )
    (model / "modules.json").write_text(json.dumps(modules))
    job = dib.EmbedJob(model=dib.models.path(str(model)), text="{t}", engine="transformers")
    with pytest.raises(ConfigError, match="Dense"):
        job.run([{"t": "x"}])


def test_dimensions_rejected_up_front_for_transformers(tiny_models: dict[str, Path]) -> None:
    with pytest.raises(ConfigError, match="dimensions= is not supported by engine transformers"):
        dib.EmbedJob(model="/w", text="{t}", engine="transformers", dimensions=8)
    with pytest.raises(ConfigError, match="not supported by engine transformers"):
        dib.EmbedJob(model="/w", text="{t}", dimensions=8)  # auto may pick transformers


def _speech(tmp_path: Path, text: str) -> Path:
    """Real speech from macOS `say` (AIFF, readable by soundfile)."""
    if shutil.which("say") is None:
        pytest.skip("needs macOS `say` for speech audio")
    out = tmp_path / "speech.aiff"
    subprocess.run(["say", "-o", str(out), text], check=True)
    return out


def test_transcribe_real_speech(tiny_models: dict[str, Path], tmp_path: Path) -> None:
    pytest.importorskip("soundfile")
    speech = _speech(tmp_path, "Hello world, this is a test.")
    job = dib.TranscribeJob(
        model=dib.models.path(str(tiny_models["whisper"])),
        audio_col="a",
        language="en",
        engine="transformers",
    )
    out = job.run([{"a": str(speech)}, {"a": speech.read_bytes()}, {"a": b"junk"}])
    for row in out[:2]:
        assert row["output_error"] is None
        text = row["output"].lower()  # whisper-tiny is not perfect ("hello lord")
        assert "hello" in text and "test" in text
        assert row["output_meta"]["audio_seconds"] > 1.0
    assert out[2]["output_error"].startswith("InvalidInputError: cannot decode audio")


def test_transcribe_timestamps(tiny_models: dict[str, Path], tmp_path: Path) -> None:
    pytest.importorskip("soundfile")
    speech = _speech(tmp_path, "Good morning everyone.")
    job = dib.TranscribeJob(
        model=dib.models.path(str(tiny_models["whisper"])),
        audio_col="a",
        language="en",
        engine="transformers",
        timestamps=True,
    )
    value = job.run([{"a": str(speech)}])[0]["output"]
    assert "morning" in value["text"].lower()
    assert value["language"] == "en"
    assert value["segments"]
    assert all(s["end"] >= s["start"] >= 0 for s in value["segments"])


def test_long_audio_is_chunked(tiny_models: dict[str, Path], tmp_path: Path) -> None:
    pytest.importorskip("soundfile")
    speech = _speech(tmp_path, "One. Two. Three. Four. Five. Six. Seven. Eight.")
    job = dib.TranscribeJob(
        model=dib.models.path(str(tiny_models["whisper"])),
        audio_col="a",
        language="en",
        engine="transformers",
        engine_options={"chunk_length_s": 2},  # forces the long-form path
    )
    row = job.run([{"a": str(speech)}])[0]
    assert row["output_error"] is None
    assert "three" in row["output"].lower() or "3" in row["output"]


def test_engine_options_validated_on_driver() -> None:
    with pytest.raises(ConfigError, match="micro_batch"):
        dib.GenerateJob(
            model="/w", prompt="{t}", engine="transformers", engine_options={"micro_batch": 2}
        )


def test_local_model_in_worker_processes(tiny_models: dict[str, Path]) -> None:
    model = dib.models.path(str(tiny_models["embedder"]))
    job = dib.EmbedJob(model=model, text="{t}", engine="transformers")
    local = job.run([{"t": "same"}])[0]["output"]
    out = job.run([{"t": "same"}, {"t": "other"}], backend="multiprocessing", workers=1)
    assert out[0]["output"] == pytest.approx(local, abs=1e-5)
