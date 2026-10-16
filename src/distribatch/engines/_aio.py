"""A private asyncio loop on a daemon thread, for engines with async clients.

Engines are called synchronously from executors — possibly from a thread that already
runs an event loop (Jupyter/Databricks notebooks), where `asyncio.run` would fail.
Running coroutines on a dedicated loop avoids that and lets one HTTP client and its
connection pool live for the engine's whole lifetime.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

_T = TypeVar("_T")


class LoopThread:
    def __init__(self, name: str) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name=name, daemon=True)
        self._thread.start()

    def run(self, coro: Coroutine[Any, Any, _T]) -> _T:
        """Run `coro` on the loop and block until it finishes."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def stop(self) -> None:
        if self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        self._loop.close()

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()
