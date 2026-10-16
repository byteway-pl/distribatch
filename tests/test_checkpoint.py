import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import distribatch as dib
from distribatch import ConfigError
from distribatch.engines.fake import FakeEngine


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """User prompts sent to the fake engine (in-process backends only). A prompt with
    STOP raises KeyboardInterrupt, simulating an interrupted run."""
    sent: list[str] = []
    original = FakeEngine.generate

    def generate(self: FakeEngine, requests: Any) -> Any:
        for r in requests:
            text = r.messages[-1].content
            if "STOP" in text:
                raise KeyboardInterrupt
            sent.append(text)
        return original(self, requests)

    monkeypatch.setattr(FakeEngine, "generate", generate)
    return sent


def _rows(n: int, stop_at: int | None = None) -> list[dict[str, Any]]:
    return [{"id": i, "t": "STOP" if i == stop_at else f"row {i}"} for i in range(n)]


def _stored_rows(path: Path) -> int:
    return sum(len(f.read_text().splitlines()) for f in (path / "results").glob("*.jsonl"))


def test_interrupted_run_resumes(tmp_path: Path, calls: list[str]) -> None:
    job = dib.GenerateJob(model="fake", prompt="{t}")
    ckpt = str(tmp_path / "ckpt")
    with pytest.raises(KeyboardInterrupt):
        job.run(_rows(10, stop_at=7), checkpoint=ckpt, id_col="id", batch_size=2)
    assert _stored_rows(tmp_path / "ckpt") == 6  # batches before the interrupt, flushed
    calls.clear()
    out = job.run(_rows(10), checkpoint=ckpt, id_col="id", batch_size=2)
    assert [r["output"] for r in out] == [f"row {i}" for i in range(10)]
    assert calls == [f"row {i}" for i in range(6, 10)]
    calls.clear()
    job.run(_rows(10), checkpoint=ckpt, id_col="id")
    assert calls == []  # everything done


def test_failed_rows_are_retried_unless_disabled(tmp_path: Path, calls: list[str]) -> None:
    job = dib.GenerateJob(model=dib.models.fake(fail_marker="row 3"), prompt="{t}")
    ckpt = str(tmp_path / "ckpt")
    job.run(_rows(5), checkpoint=ckpt, id_col="id")
    calls.clear()
    out = job.run(_rows(5), checkpoint=ckpt, id_col="id", retry_failed=False)
    assert calls == []
    assert out[3]["output_error"] == "EngineError: fake failure"
    job.run(_rows(5), checkpoint=ckpt, id_col="id")
    assert calls == ["row 3"]


def test_success_wins_over_older_failure(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    failing = dib.GenerateJob(model=dib.models.fake(fail_marker="x"), prompt="{t}")
    # Same fingerprint requires the same job; write a failure then a success by hand.
    failing.run([{"id": "a", "t": "x"}], checkpoint=str(ckpt), id_col="id")
    (part,) = (ckpt / "results").glob("*.jsonl")
    success = {"id": "a", "value": "fixed", "error": None, "meta": {"attempts": 1}}
    (ckpt / "results" / "part-99999999999999999999-0.jsonl").write_text(json.dumps(success) + "\n")
    out = failing.run([{"id": "a", "t": "x"}], checkpoint=str(ckpt), id_col="id")
    assert out[0]["output"] == "fixed"


def test_checkpoint_is_bound_to_the_job(tmp_path: Path) -> None:
    ckpt = str(tmp_path / "ckpt")
    dib.GenerateJob(model="fake", prompt="{t}").run(_rows(2), checkpoint=ckpt, id_col="id")
    with pytest.raises(ConfigError, match="different job"):
        dib.GenerateJob(model="fake", prompt="Q: {t}").run(_rows(2), checkpoint=ckpt, id_col="id")
    manifest = json.loads((tmp_path / "ckpt" / "manifest.json").read_text())
    assert manifest["id_col"] == "id" and manifest["task"] == "generate"


def test_id_validation(tmp_path: Path) -> None:
    job = dib.GenerateJob(model="fake", prompt="{t}")
    ckpt = str(tmp_path / "c")
    with pytest.raises(ConfigError, match="duplicate"):
        job.run([{"id": 1, "t": "a"}, {"id": 1, "t": "b"}], checkpoint=ckpt, id_col="id")
    with pytest.raises(ConfigError, match="strings or integers"):
        job.run([{"id": 1.5, "t": "a"}], checkpoint=ckpt, id_col="id")
    with pytest.raises(ConfigError, match="no id column"):
        job.run([{"t": "a"}], checkpoint=ckpt, id_col="id")
    with pytest.raises(ConfigError, match="together"):
        job.run([{"t": "a"}], checkpoint=ckpt)


def test_dedupe(calls: list[str]) -> None:
    job = dib.GenerateJob(model="fake", prompt="{t}")
    out = job.run([{"t": "a"}, {"t": "b"}, {"t": "a"}, {"t": "a"}], dedupe=True, batch_size=1)
    assert [r["output"] for r in out] == ["a", "b", "a", "a"]
    assert calls == ["a", "b"]
    assert [r["output_meta"]["attempts"] for r in out] == [1, 1, 0, 0]
    assert out[2]["output_meta"]["prompt_tokens"] is None


def test_dedupe_with_checkpoint(tmp_path: Path, calls: list[str]) -> None:
    job = dib.GenerateJob(model="fake", prompt="{t}")
    rows = [{"id": i, "t": "same" if i % 2 else f"u{i}"} for i in range(6)]
    out = job.run(rows, dedupe=True, checkpoint=str(tmp_path / "c"), id_col="id")
    assert calls == ["u0", "same", "u2", "u4"]
    assert _stored_rows(tmp_path / "c") == 6  # every id stored, copies included
    assert [r["output"] for r in out] == [r["t"] for r in rows]


def test_multiprocessing_with_checkpoint(tmp_path: Path) -> None:
    job = dib.GenerateJob(model=dib.models.fake(fail_marker="row 4"), prompt="{t}")
    ckpt = str(tmp_path / "c")
    df = pd.DataFrame(_rows(9))
    out = job.run(df, backend="multiprocessing", workers=2, checkpoint=ckpt, id_col="id")
    assert out["output_error"].notna().sum() == 1
    assert _stored_rows(tmp_path / "c") == 9
    again = job.run(df, backend="multiprocessing", workers=2, checkpoint=ckpt, id_col="id")
    assert _stored_rows(tmp_path / "c") == 10  # only the failed row was retried
    assert list(again["output"].fillna("")) == list(out["output"].fillna(""))
