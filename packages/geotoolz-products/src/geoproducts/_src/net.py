"""HTTP plumbing shared by every product reader.

- :func:`retrying` — one retry loop for both styles of client: calls that
  *raise* on failure (``urllib``) and calls that *return* a failed
  response (``requests``). A ``wait`` callback looks at each outcome and
  says how long to sleep before the next attempt, or ``None`` to stop.
- :func:`retry_after_seconds` / :func:`backoff_seconds` — how long to
  wait: the server's ``Retry-After`` when it sends one (API rate limits),
  exponential backoff otherwise, always clamped.
- :func:`stream_to_file` — write an API client's response chunks
  atomically through a sibling ``.part`` file, so a failed transfer never
  leaves a truncated file under the final name.
- :func:`bearer_headers_for` — attach a token only to the one API host it
  belongs to.

Standard library only: readers whose own client needs ``requests`` (or a
provider SDK) plug their transport into these helpers. Plain object and
URL downloads (``s3://``, ``https://``, …) go through ``geocloud.files``
instead (see ``_src/s3.py``).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from geoproducts._src.files import atomic_path


__all__ = [
    "MAX_RETRY_WAIT_S",
    "backoff_seconds",
    "bearer_headers_for",
    "retry_after_seconds",
    "retrying",
    "stream_to_file",
]

#: Upper bound on any single retry sleep, whatever a server asks for.
MAX_RETRY_WAIT_S = 300.0


def backoff_seconds(
    attempt: int, *, base_s: float = 1.0, cap_s: float = MAX_RETRY_WAIT_S
) -> float:
    """Exponential backoff ``base_s * 2**attempt``, clamped to ``[0, cap_s]``.

    Examples:
        >>> [backoff_seconds(a, base_s=5.0) for a in range(3)]
        [5.0, 10.0, 20.0]
    """
    return max(0.0, min(base_s * 2.0**attempt, cap_s))


def retry_after_seconds(
    header: str | None,
    attempt: int,
    *,
    base_s: float = 5.0,
    cap_s: float = MAX_RETRY_WAIT_S,
) -> float:
    """Seconds to wait before retrying a throttled (429 / 503) request.

    1. Honour ``Retry-After`` as delta-seconds (int or float).
    2. Else honour ``Retry-After`` as an HTTP-date.
    3. Else back off exponentially from ``base_s``.

    The result is clamped to ``[0, cap_s]``.

    Args:
        header: The ``Retry-After`` header value, or ``None``.
        attempt: Zero-based retry number (drives the fallback backoff).
        base_s: First fallback wait.
        cap_s: Longest wait honoured.

    Examples:
        >>> retry_after_seconds("2", attempt=0)
        2.0
        >>> retry_after_seconds(None, attempt=1)
        10.0
    """
    text = (header or "").strip()
    wait: float | None = None
    if text:
        try:
            wait = float(text)
        except ValueError:
            try:
                wait = (parsedate_to_datetime(text) - datetime.now(UTC)).total_seconds()
            except (TypeError, ValueError):
                wait = None
    if wait is None:
        return backoff_seconds(attempt, base_s=base_s, cap_s=cap_s)
    return max(0.0, min(wait, cap_s))


def retrying[T](
    call: Callable[[], T],
    *,
    wait: Callable[[Any, int], float | None],
    attempts: int,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Callable[[Any, int, float], None] | None = None,
) -> T:
    """Call ``call`` until it succeeds, ``wait`` says stop, or attempts run out.

    Each outcome — the returned value or the raised exception — goes to
    ``wait(outcome, attempt)``: a number of seconds means "sleep, then
    try again", ``None`` means "done" (a value is returned, an exception
    re-raised). The last attempt's outcome is final either way, so a
    client that returns failed responses gets the last one back.

    Args:
        call: Zero-argument callable doing one attempt.
        wait: Retry policy; see above.
        attempts: Total attempts (``>= 1``).
        sleep: Sleep function (injectable for tests).
        on_retry: Called as ``on_retry(outcome, attempt, seconds)`` before
            each sleep — for logging, or closing a discarded response.

    Returns:
        The final value of ``call``.

    Raises:
        ValueError: ``attempts`` is below 1.
        Exception: The final exception of ``call``, when it raised.

    Examples:
        >>> outcomes = iter([OSError("reset"), "ok"])
        >>> def flaky():
        ...     value = next(outcomes)
        ...     if isinstance(value, Exception):
        ...         raise value
        ...     return value
        >>> retrying(
        ...     flaky,
        ...     wait=lambda o, a: 0.0 if isinstance(o, OSError) else None,
        ...     attempts=3,
        ... )
        'ok'
    """
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1; got {attempts}.")
    for attempt in range(attempts):
        last = attempt == attempts - 1
        try:
            result = call()
        except Exception as exc:
            delay = None if last else wait(exc, attempt)
            if delay is None:
                raise
            outcome: Any = exc
        else:
            delay = None if last else wait(result, attempt)
            if delay is None:
                return result
            outcome = result
        if on_retry is not None:
            on_retry(outcome, attempt, delay)
        sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


def stream_to_file(chunks: Iterable[bytes], dest: Path | str) -> Path:
    """Write ``chunks`` to ``dest`` atomically through a sibling ``.part``.

    The final name only appears once every byte arrived; if iterating
    ``chunks`` fails, the partial file is removed and the error re-raised.

    Args:
        chunks: Byte chunks, e.g. ``response.iter_content(8192)``.
        dest: Destination file (its directory is created if missing).

    Returns:
        ``dest`` as a ``Path``.
    """
    dest = Path(dest)
    with atomic_path(dest) as part, part.open("wb") as out:
        for chunk in chunks:
            out.write(chunk)
    return dest


def bearer_headers_for(url: str, token: str | None, *, host: str) -> dict[str, str]:
    """The ``Authorization`` header for one request to ``url``, or ``{}``.

    The token is attached only to ``https://`` URLs on ``host``, so a
    provider token never leaks to a CDN, a redirect target or another
    provider.

    Args:
        url: The URL about to be requested.
        token: Bearer token, or ``None``.
        host: The one hostname the token belongs to.

    Returns:
        ``{"Authorization": "Bearer …"}`` or an empty dict.

    Examples:
        >>> bearer_headers_for("https://api.example.org/x", "t", host="api.example.org")
        {'Authorization': 'Bearer t'}
        >>> bearer_headers_for("https://cdn.example.org/x", "t", host="api.example.org")
        {}
    """
    parts = urlsplit(str(url))
    if not token or parts.scheme != "https" or parts.hostname != host:
        return {}
    return {"Authorization": f"Bearer {token}"}
