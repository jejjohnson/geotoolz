"""Anonymous access to public S3 buckets over plain HTTPS.

Open-data programmes (NOAA's GOES / Himawari / JPSS buckets, the AWS
Registry of Open Data) publish to S3 buckets that answer unsigned
requests. Listing (ListObjectsV2) and downloading need nothing beyond the
standard library, so readers built on this module work without an AWS SDK
or credentials.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.parse
import xml.etree.ElementTree as ET
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from geoproducts._src.net import (
    backoff_seconds,
    download_url,
    fetch_bytes,
    retrying,
    urllib_wait,
)


__all__ = ["download_object", "list_objects", "object_url", "s3_wait"]

_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def object_url(bucket: str, key: str = "") -> str:
    """Virtual-hosted HTTPS URL of ``key`` in ``bucket``.

    Examples:
        >>> object_url("noaa-goes19", "ABI-L1b-RadC/x.nc")
        'https://noaa-goes19.s3.amazonaws.com/ABI-L1b-RadC/x.nc'
    """
    return f"https://{bucket}.s3.amazonaws.com/{urllib.parse.quote(key)}"


def s3_wait(outcome: Any, attempt: int, *, base_s: float = 1.0) -> float | None:
    """Retry policy for anonymous S3: :func:`urllib_wait`, plus spurious 404s.

    S3 occasionally answers ``NoSuchBucket`` for a bucket that exists (and
    that the previous request listed); that 404 is retried. ``NoSuchKey``
    and every other client error fail at once.
    """
    if isinstance(outcome, urllib.error.HTTPError) and outcome.code == 404:
        body = outcome.read() if outcome.fp is not None else b""
        if b"NoSuchBucket" not in body:
            return None
        return backoff_seconds(attempt, base_s=base_s)
    return urllib_wait(outcome, attempt, base_s=base_s)


def list_objects(
    bucket: str,
    prefix: str,
    *,
    attempts: int = 4,
    backoff_s: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> list[tuple[str, int]]:
    """``(key, size)`` of every object under ``prefix`` (ListObjectsV2, paged).

    Args:
        bucket: Public bucket name.
        prefix: Key prefix (``"ABI-L1b-RadC/2026/280/12/"``).
        attempts: Attempts per page request.
        backoff_s: First retry wait.
        sleep: Sleep function (injectable for tests).

    Raises:
        urllib.error.URLError: A page request failed after its retries.
    """
    wait = partial(s3_wait, base_s=backoff_s)
    out: list[tuple[str, int]] = []
    token: str | None = None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        listing = f"{object_url(bucket)}?{urllib.parse.urlencode(params)}"
        body = retrying(
            partial(fetch_bytes, listing), wait=wait, attempts=attempts, sleep=sleep
        )
        root = ET.fromstring(body)
        for item in root.iter(f"{_S3_NS}Contents"):
            key = item.findtext(f"{_S3_NS}Key")
            if key:
                out.append((key, int(item.findtext(f"{_S3_NS}Size") or 0)))
        if root.findtext(f"{_S3_NS}IsTruncated") != "true":
            return out
        token = root.findtext(f"{_S3_NS}NextContinuationToken")


def download_object(
    bucket: str,
    key: str,
    dest: Path | str,
    *,
    attempts: int = 4,
    backoff_s: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Download ``key`` from ``bucket`` to the file ``dest`` (atomic, retried).

    Raises:
        urllib.error.URLError: The download failed after its retries.
    """
    return download_url(
        object_url(bucket, key),
        dest,
        attempts=attempts,
        wait=partial(s3_wait, base_s=backoff_s),
        sleep=sleep,
    )
