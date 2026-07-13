"""Retry helpers for transient remote I/O failures."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from loguru import logger as log
from rasterio.errors import RasterioIOError
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)


try:
    from urllib3.exceptions import ReadTimeoutError
except ImportError:  # pragma: no cover - urllib3 is optional at runtime
    _TRANSIENT_IO_ERRORS: tuple[type[BaseException], ...] = (RasterioIOError, OSError)
else:
    _TRANSIENT_IO_ERRORS = (RasterioIOError, OSError, ReadTimeoutError)


# OSError subclasses that are deterministic / will not heal on retry. Filter
# these out so callers don't pay 3x the wall time on a typo'd path or a perms
# misconfiguration. RasterioIOError typically wraps real remote/network blips
# (HTTP 5xx, partial reads) and stays in scope.
_FATAL_OS_ERRORS: tuple[type[BaseException], ...] = (
    FileNotFoundError,
    PermissionError,
    IsADirectoryError,
    NotADirectoryError,
    InterruptedError,
)


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, _FATAL_OS_ERRORS):
        return False
    return isinstance(exc, _TRANSIENT_IO_ERRORS)


# Exponential backoff capped at 30s per wait so a long-tail outage doesn't
# wedge a worker for minutes between attempts. Jitter is bounded to [0, 1]s
# and added on top — the effective per-wait cap is 31s, well under any
# reasonable request timeout.
_RETRY_WAIT = wait_exponential(multiplier=1, min=1, max=30) + wait_random(0, 1)


def _log_before_sleep(retry_state: RetryCallState) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome is not None else None
    sleep = (
        retry_state.next_action.sleep if retry_state.next_action is not None else 0.0
    )
    log.warning(
        "Transient I/O error on attempt {}; retrying in {:.2f}s: {}",
        retry_state.attempt_number,
        sleep,
        exc,
    )


def retry_transient_io[T](
    fn: Callable[..., T],
    *args: Any,
    retries: int,
    **kwargs: Any,
) -> T:
    """Call ``fn`` with retries for transient remote I/O exceptions.

    Args:
        fn: Callable to invoke.
        *args: Positional args forwarded to ``fn``.
        retries: Number of retries after the initial attempt. ``0`` disables
            retry/backoff (the call runs exactly once and any exception
            propagates immediately). Must be ``>= 0``.
        **kwargs: Keyword args forwarded to ``fn``.

    Returns:
        Whatever ``fn`` returns.

    Notes:
        Only `RasterioIOError`, `urllib3.exceptions.ReadTimeoutError`, and
        non-fatal `OSError` instances are retried. Fatal `OSError` subclasses
        (`FileNotFoundError`, `PermissionError`, `IsADirectoryError`,
        `NotADirectoryError`, `InterruptedError`) are re-raised on the first
        attempt — there is no benefit to waiting on them.
    """
    if retries < 0:
        raise ValueError(f"retries must be >= 0; got {retries}")
    if retries == 0:
        return fn(*args, **kwargs)

    retryer = Retrying(
        retry=retry_if_exception(_is_transient),
        stop=stop_after_attempt(retries + 1),
        wait=_RETRY_WAIT,
        before_sleep=_log_before_sleep,
        reraise=True,
    )
    return retryer(fn, *args, **kwargs)
