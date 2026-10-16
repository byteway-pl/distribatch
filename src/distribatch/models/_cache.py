"""Node-local model cache: one download/copy per node, however many workers ask.

Entries are directories under `cache_root()`, filled exactly once under an exclusive
file lock (`fcntl.flock`, so it also coordinates separate worker processes) and
published by an atomic rename, with a `.complete` marker. Readers that find the
marker never take the lock.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import logging
import os
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path

from distribatch.types import DistribatchError

_MARKER = ".complete"

log = logging.getLogger("distribatch")


def cache_root() -> Path:
    """``$DISTRIBATCH_CACHE_DIR``, else Databricks' local NVMe (``/local_disk0``), else
    ``~/.cache/distribatch``."""
    env = os.environ.get("DISTRIBATCH_CACHE_DIR")
    if env:
        return Path(env)
    local_disk = Path("/local_disk0")
    if local_disk.is_dir() and os.access(local_disk, os.W_OK):
        return local_disk / "distribatch" / "cache"
    return Path.home() / ".cache" / "distribatch"


def entry_name(*parts: str) -> str:
    """Readable, filesystem-safe and collision-free cache entry name."""
    kind, rest = parts[0], "-".join(parts[1:])
    readable = f"{kind}-{rest[-70:]}" if rest else kind  # keep the kind prefix, path tail
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in readable)
    digest = hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:12]
    return f"{safe}-{digest}"


@contextlib.contextmanager
def node_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
        except OSError:
            # Some FUSE mounts (e.g. UC Volumes) don't support flock. Callers that use
            # those (driver-side staging) are single-process anyway.
            log.warning("distribatch: no file locking on %s; continuing unlocked", path)
            yield
            return
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def materialize(name: str, fill: Callable[[Path], object], root: Path | None = None) -> Path:
    """The cache directory `name`, created by `fill(tmp_dir)` if missing."""
    root = root or cache_root()
    final = root / name
    if (final / _MARKER).exists():
        return final
    with node_lock(root / f".{name}.lock"):
        if (final / _MARKER).exists():  # another process filled it while we waited
            return final
        if final.exists():  # leftover of an interrupted fill
            shutil.rmtree(final)
        tmp = root / f".{name}.tmp-{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        try:
            fill(tmp)
            (tmp / _MARKER).touch()
            os.replace(tmp, final)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
    if not (final / _MARKER).exists():
        raise DistribatchError(f"model cache entry {final} was not created")
    return final
