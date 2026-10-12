"""S3's spurious ``NoSuchBucket``, retried for every object-store call.

S3 occasionally answers ``NoSuchBucket`` for a bucket that exists (and that
the previous request listed); the same call succeeds a moment later. obstore
retries 5xx answers, dropped connections and throttling itself, but not this
error, so every geocloud call that talks to a store goes through
`with_bucket_retry` / `awith_bucket_retry`: the error is retried with
exponential backoff, and everything else fails at once.

obstore surfaces it two ways. A LIST or GET carries S3's error body, so the
message names ``NoSuchBucket``. A HEAD's 404 has no body and arrives as a
plain `FileNotFoundError`, indistinguishable from a missing key; there the
caller passes a ``probe`` — a cheap non-recursive LIST of the key's parent —
and the call is retried only when the probe fails with ``NoSuchBucket``. A
missing key (the probe succeeds) fails immediately, with no backoff.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable


__all__ = ["awith_bucket_retry", "is_no_such_bucket", "with_bucket_retry"]

#: Calls per operation, and the first backoff in seconds (doubling each time).
ATTEMPTS = 4
BASE_S = 1.0


def is_no_such_bucket(exc: BaseException) -> bool:
    """Whether ``exc`` is S3's ``NoSuchBucket`` answer.

    Examples:
        >>> is_no_such_bucket(OSError("<Code>NoSuchBucket</Code>"))
        True
        >>> is_no_such_bucket(FileNotFoundError("NoSuchKey"))
        False
    """
    return "NoSuchBucket" in str(exc)


def _sleep_s(attempt: int) -> float:
    return BASE_S * 2.0**attempt


def _retryable(exc: Exception, probe: Callable[[], object] | None) -> bool:
    if is_no_such_bucket(exc):
        return True
    if isinstance(exc, FileNotFoundError) and probe is not None:
        try:
            probe()
        except Exception as failure:
            return is_no_such_bucket(failure)
    return False


def with_bucket_retry[T](
    call: Callable[[], T], *, probe: Callable[[], object] | None = None
) -> T:
    """``call()``, retried while S3 answers a spurious ``NoSuchBucket``.

    Args:
        call: The store operation.
        probe: For a HEAD-based ``call``: a LIST that tells a bodiless 404
            (``FileNotFoundError``) for a missing bucket from a missing key.

    Returns:
        What ``call`` returns.

    Raises:
        Exception: ``call``'s error — at once unless it is ``NoSuchBucket``,
            else after ``ATTEMPTS`` calls.

    Examples:
        >>> with_bucket_retry(lambda: 42)
        42
    """
    for attempt in range(ATTEMPTS):
        try:
            return call()
        except Exception as exc:
            if attempt + 1 >= ATTEMPTS or not _retryable(exc, probe):
                raise
            time.sleep(_sleep_s(attempt))
    raise AssertionError("unreachable")  # pragma: no cover


async def awith_bucket_retry[T](
    call: Callable[[], Awaitable[T]], *, probe: Callable[[], object] | None = None
) -> T:
    """Async twin of `with_bucket_retry`: ``call`` returns a fresh awaitable.

    Examples:
        >>> async def answer() -> int:
        ...     return 42
        >>> asyncio.run(awith_bucket_retry(answer))
        42
    """
    for attempt in range(ATTEMPTS):
        try:
            return await call()
        except Exception as exc:
            if attempt + 1 >= ATTEMPTS or not _retryable(exc, probe):
                raise
            await asyncio.sleep(_sleep_s(attempt))
    raise AssertionError("unreachable")  # pragma: no cover
