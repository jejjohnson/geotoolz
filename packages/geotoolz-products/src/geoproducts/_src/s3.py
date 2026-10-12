"""Anonymous access to public S3 buckets, through `geocloud.files`.

Open-data programmes (NOAA's GOES / Himawari / JPSS buckets, the AWS
Registry of Open Data) publish to S3 buckets that answer unsigned
requests. Listing and downloads go through geotoolz-cloud's file verbs on
the shared obstore pool, unsigned (no credential lookup, no
instance-metadata probe), so they share connections with every other
cloud read in the process; obstore retries server errors and dropped
connections, and geotoolz-cloud S3's spurious ``NoSuchBucket``.
geotoolz-cloud comes with the ``[goes]`` and ``[himawari]`` extras.

A store mounted at the bucket's root (`geocloud.store.mount`) serves these
calls instead, which is how the tests run without the network.
"""

from __future__ import annotations

import urllib.parse
from pathlib import Path
from types import ModuleType

from geoproducts._src.extras import require


__all__ = ["download_object", "list_objects", "object_url"]

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


def list_objects(
    bucket: str, prefix: str, *, extra: str = "obstore"
) -> list[tuple[str, int]]:
    """``(key, size)`` of every object under ``prefix``, sorted by key.

    Args:
        bucket: Public bucket name.
        prefix: Key prefix, matched by whole path segments
            (``"ABI-L1b-RadC/2026/280/12/"``).
        extra: The extra an install hint names when geotoolz-cloud is
            missing (the calling sensor's: ``"goes"``, ``"himawari"``).

    Raises:
        ImportError: geotoolz-cloud is not installed.
        Exception: The listing failed (after geotoolz-cloud's retries).
    """
    files = _files(extra)
    root = f"s3://{bucket}/"
    listed = files.ls(root + prefix, storage_options=_ANONYMOUS)
    return [(item.uri.removeprefix(root), item.size) for item in listed]


def download_object(
    bucket: str, key: str, dest: Path | str, *, extra: str = "obstore"
) -> Path:
    """Download ``key`` from ``bucket`` to the file ``dest``.

    Streamed to a hidden ``.part`` sibling, size-checked and renamed into
    place (`geocloud.files.download`), so a failed transfer never leaves a
    truncated file under ``dest``. ``extra`` is as for `list_objects`.

    Raises:
        ImportError: geotoolz-cloud is not installed.
        FileNotFoundError: No such key.
        Exception: The download failed (after geotoolz-cloud's retries).
    """
    files = _files(extra)
    return files.download(
        f"s3://{bucket}/{key}", Path(dest), storage_options=_ANONYMOUS
    )
