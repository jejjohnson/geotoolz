"""Drive a coroutine from synchronous code, inside or outside a running loop."""

from __future__ import annotations

import asyncio
import threading
from typing import Any


def _run_coroutine_safely(coro: Any) -> Any:
    """Drive ``coro`` to completion regardless of running-loop state.

    ``asyncio.run`` raises ``RuntimeError`` when nested under a running
    loop (Jupyter, FastAPI handler, pytest-asyncio). Detect that case
    and run on a worker thread with its own loop so the calling thread
    stays sync.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_box: dict[str, Any] = {}

    def _runner() -> None:
        loop = asyncio.new_event_loop()
        try:
            result_box["value"] = loop.run_until_complete(coro)
        except BaseException as exc:
            result_box["error"] = exc
        finally:
            loop.close()

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in result_box:
        raise result_box["error"]
    return result_box["value"]
