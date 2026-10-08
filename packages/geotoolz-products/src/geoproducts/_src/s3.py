"""Anonymous access to public S3 buckets, through `geocloud.files`.

Open-data programmes (NOAA's GOES / Himawari / JPSS buckets, the AWS
Registry of Open Data) publish to S3 buckets that answer unsigned
requests. Listing and downloads go through geotoolz-cloud's file verbs on
the shared obstore pool, unsigned (no credential lookup, no
instance-metadata probe), so they share connections with every other
cloud read in the process; obstore retries server errors and dropped
connections itself. geotoolz-cloud comes with the ``[obstore]``,
``[goes]`` and ``[himawari]`` extras.

A store mounted at the bucket's root (`geocloud.store.mount`) serves these
calls instead, which is how the tests run without the network.
"""

from __future__ import annotations

import time
import urllib.parse
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any

from geoproducts._src.extras import require
from geoproducts._src.net import backoff_seconds, retrying


__all__ = ["download_object", "list_objects", "object_url", "s3_wait"]

#: Unsigned requests to AWS itself in the buckets' home region. The endpoint
#: is explicit so ``$AWS_ENDPOINT_URL`` (pointing at MinIO or LocalStack for
#: the user's own work) never redirects these public reads.
_ANONYMOUS = {
    "skip_signature": True,
    "region": "us-east-1",
    "endpoint": "https://s3.us-east-1.amazonaws.com",
}


def _files(extra: str) -> ModuleType:
    return require("geocloud.files", "Listing and downloading public buckets", extra)


def object_url(bucket: str, key: str = "") -> str:
    """Virtual-hosted HTTPS URL of ``key`` in ``bucket``.

    Examples:
        >>> object_url("noaa-goes19", "ABI-L1b-RadC/x.nc")
        'https://noaa-goes19.s3.amazonaws.com/ABI-L1b-RadC/x.nc'
    """
    return f"https://{bucket}.s3.amazonaws.com/{urllib.parse.quote(key)}"


def s3_wait(outcome: Any, attempt: int, *, base_s: float = 1.0) -> float | None:
    """Retry policy on top of obstore's: S3's spurious ``NoSuchBucket``.

    S3 occasionally answers ``NoSuchBucket`` for a bucket that exists (and
    that the previous request listed); that error is retried with
    exponential backoff. obstore already retries 5xx answers and dropped
    connections (and paces 429 / 503 throttling with its own backoff), so
    everything else — a missing key included — fails at once.

    Examples:
        >>> s3_wait(OSError("<Code>NoSuchBucket</Code>"), 0)
        1.0
        >>> s3_wait(FileNotFoundError("NoSuchKey"), 0) is None
        True
    """
    if isinstance(outcome, BaseException) and "NoSuchBucket" in str(outcome):
        return backoff_seconds(attempt, base_s=base_s)
    return None


def list_objects(
    bucket: str,
    prefix: str,
    *,
    attempts: int = 4,
    backoff_s: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    extra: str = "obstore",
) -> list[tuple[str, int]]:
    """``(key, size)`` of every object under ``prefix``, sorted by key.

    Args:
        bucket: Public bucket name.
        prefix: Key prefix, matched by whole path segments
            (``"ABI-L1b-RadC/2026/280/12/"``).
        attempts: Attempts against a spurious ``NoSuchBucket``.
        backoff_s: First retry wait.
        sleep: Sleep function (injectable for tests).
        extra: The extra an install hint names when geotoolz-cloud is
            missing (the calling sensor's: ``"goes"``, ``"himawari"``).

    Raises:
        ImportError: geotoolz-cloud is not installed.
        Exception: The listing failed (obstore's error, after its retries).
    """
    files = _files(extra)
    root = f"s3://{bucket}/"
    listed = retrying(
        partial(files.ls, root + prefix, storage_options=_ANONYMOUS),
        wait=partial(s3_wait, base_s=backoff_s),
        attempts=attempts,
        sleep=sleep,
    )
    return [(item.uri.removeprefix(root), item.size) for item in listed]


def download_object(
    bucket: str,
    key: str,
    dest: Path | str,
    *,
    attempts: int = 4,
    backoff_s: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    extra: str = "obstore",
) -> Path:
    """Download ``key`` from ``bucket`` to the file ``dest``.

    Streamed to a hidden ``.part`` sibling, size-checked and renamed into
    place (`geocloud.files.download`), so a failed transfer never leaves a
    truncated file under ``dest``. ``extra`` is as for `list_objects`.

    Raises:
        ImportError: geotoolz-cloud is not installed.
        FileNotFoundError: No such key.
        Exception: The download failed (obstore's error, after its retries).
    """
    files = _files(extra)
    parent = key.rsplit("/", 1)[0] if "/" in key else ""

    def wait(outcome: Any, attempt: int) -> float | None:
        # A download starts with a HEAD, whose 404 has no body: a missing
        # key and S3's spurious ``NoSuchBucket`` look alike. Listing the
        # key's directory tells them apart (a LIST 404 names its code).
        if isinstance(outcome, FileNotFoundError):
            try:
                files.ls(
                    f"s3://{bucket}/{parent}",
                    recursive=False,
                    storage_options=_ANONYMOUS,
                )
            except Exception as probe:
                return s3_wait(probe, attempt, base_s=backoff_s)
            return None
        return s3_wait(outcome, attempt, base_s=backoff_s)

    return retrying(
        partial(
            files.download,
            f"s3://{bucket}/{key}",
            Path(dest),
            storage_options=_ANONYMOUS,
        ),
        wait=wait,
        attempts=attempts,
        sleep=sleep,
    )
