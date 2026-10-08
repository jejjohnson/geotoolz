"""Find and fetch ABI files in NOAA's public GOES buckets on AWS.

NOAA's Open Data Dissemination programme mirrors every GOES-R product to
public, anonymous S3 buckets (``noaa-goes16`` … ``noaa-goes19``), keyed by
product and scan-start hour::

    <product>/<year>/<day-of-year>/<hour>/OR_<product>-M6C<nn>_G<nn>_s<start>_e<end>_c<created>.nc

This module lists and downloads those objects over plain HTTPS with the
standard library only: no credentials, no AWS SDK.
"""

from __future__ import annotations

import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path


__all__ = [
    "ABIFile",
    "bucket",
    "download",
    "list_files",
    "parse_key",
    "url",
]

PRODUCTS: tuple[str, ...] = ("ABI-L1b-RadF", "ABI-L1b-RadC", "ABI-L1b-RadM")
_SATELLITES = (16, 17, 18, 19)
_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
_TIMEOUT_S = 60.0
# Transient failures (5xx, dropped connections, a spurious ``NoSuchBucket``
# for a bucket that answered the listing) are retried with exponential
# backoff; a missing key or a refused request fails at once.
_ATTEMPTS = 4
_BACKOFF_S = 1.0
_KEY_RE = re.compile(
    r"OR_(?P<product>ABI-L\w+-\w+?)(?P<sector>[12])?-M(?P<mode>\d)"
    r"C(?P<channel>\d{2})_G(?P<satellite>\d{2})"
    r"_s(?P<start>\d{14})_e(?P<end>\d{14})_c(?P<created>\d{14})\.nc$"
)


@dataclass(frozen=True)
class ABIFile:
    """One ABI object in a NOAA bucket, with its file-name fields parsed."""

    key: str
    satellite: str
    product: str
    sector: str | None
    mode: int
    channel: str
    start: datetime
    end: datetime
    created: datetime
    size: int | None = None

    @property
    def name(self) -> str:
        """The object's file name."""
        return self.key.rsplit("/", 1)[-1]

    @property
    def url(self) -> str:
        """Public HTTPS URL of the object."""
        return url(self.satellite, self.key)


def bucket(satellite: str | int) -> str:
    """The NOAA bucket of a GOES-R satellite (``"G19"`` / ``19`` → ``noaa-goes19``).

    Raises:
        ValueError: ``satellite`` is not GOES-16 … GOES-19.
    """
    return f"noaa-goes{_satellite_number(satellite)}"


def url(satellite: str | int, key: str) -> str:
    """Public HTTPS URL of ``key`` in ``satellite``'s bucket."""
    return f"https://{bucket(satellite)}.s3.amazonaws.com/{urllib.parse.quote(key)}"


def parse_key(key: str, *, size: int | None = None) -> ABIFile:
    """Parse an ABI object key (or bare file name).

    Raises:
        ValueError: ``key`` does not follow the ABI file-naming convention.
    """
    match = _KEY_RE.search(key)
    if match is None:
        raise ValueError(f"not an ABI file name: {key!r}")
    return ABIFile(
        key=key,
        satellite=f"G{match['satellite']}",
        product=match["product"],
        sector=f"M{match['sector']}" if match["sector"] else None,
        mode=int(match["mode"]),
        channel=f"C{match['channel']}",
        start=_parse_stamp(match["start"]),
        end=_parse_stamp(match["end"]),
        created=_parse_stamp(match["created"]),
        size=size,
    )


def list_files(
    *,
    satellite: str | int,
    start: datetime,
    end: datetime | None = None,
    product: str = "ABI-L1b-RadC",
    channel: int | str | None = None,
    sector: str | None = None,
) -> list[ABIFile]:
    """List the ABI files whose scan starts in ``[start, end)``.

    Args:
        satellite: ``"G16"`` … ``"G19"`` (or ``16`` … ``19``).
        start: Window start. Naive datetimes are taken as UTC.
        end: Window end (exclusive). Default: ``start`` plus one hour.
        product: Bucket product prefix, e.g. ``"ABI-L1b-RadF"`` (full disk),
            ``"ABI-L1b-RadC"`` (CONUS / PACUS) or ``"ABI-L1b-RadM"``
            (mesoscale). Default ``"ABI-L1b-RadC"``.
        channel: Keep one channel (``13`` or ``"C13"``). Default: all.
        sector: Mesoscale sector, ``"M1"`` or ``"M2"``. Default: both.

    Returns:
        The matching files, sorted by scan start then channel.

    Raises:
        ValueError: ``end`` is not after ``start``, or ``channel`` /
            ``satellite`` is out of range.
        urllib.error.URLError: The bucket listing failed (transient errors
            are retried first).
    """
    start = _as_utc(start)
    end = start + timedelta(hours=1) if end is None else _as_utc(end)
    if end <= start:
        raise ValueError(f"end ({end}) must be after start ({start}).")
    wanted_channel = None if channel is None else _channel_name(channel)
    files: list[ABIFile] = []
    hour = start.replace(minute=0, second=0, microsecond=0)
    while hour < end:
        prefix = f"{product}/{hour:%Y}/{hour:%j}/{hour:%H}/"
        for key, size in _list_objects(bucket(satellite), prefix):
            try:
                item = parse_key(key, size=size)
            except ValueError:
                continue
            if not start <= item.start < end:
                continue
            if wanted_channel is not None and item.channel != wanted_channel:
                continue
            if sector is not None and item.sector != sector:
                continue
            files.append(item)
        hour += timedelta(hours=1)
    return sorted(files, key=lambda item: (item.start, item.channel, item.key))


def download(
    file: ABIFile | str,
    dest: str | Path,
    *,
    satellite: str | int | None = None,
    overwrite: bool = False,
) -> Path:
    """Download one ABI object into the directory ``dest``.

    The file is streamed to ``<name>.part`` and renamed when complete, so
    an interrupted download never leaves a truncated ``.nc`` behind. An
    existing file is reused unless ``overwrite`` is set.

    Args:
        file: An :class:`ABIFile` from :func:`list_files`, or an object key.
        dest: Destination directory (created if missing).
        satellite: Required when ``file`` is a key whose name does not
            identify the satellite; otherwise read from the file name.
        overwrite: Re-download even when the file already exists.

    Returns:
        Path of the downloaded file.

    Raises:
        urllib.error.URLError: The download failed (transient errors are
            retried first).
    """
    item = parse_key(file) if isinstance(file, str) else file
    source = url(satellite or item.satellite, item.key)
    target = Path(dest) / item.name
    if target.exists() and not overwrite:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    part_path = target.with_name(target.name + ".part")
    try:
        _with_retries(lambda: _stream_to(source, part_path))
        part_path.replace(target)
    finally:
        part_path.unlink(missing_ok=True)
    return target


def _list_objects(bucket_name: str, prefix: str) -> list[tuple[str, int]]:
    """``(key, size)`` of every object under ``prefix`` (ListObjectsV2)."""
    out: list[tuple[str, int]] = []
    token: str | None = None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        query = urllib.parse.urlencode(params)
        listing = f"https://{bucket_name}.s3.amazonaws.com/?{query}"
        root = ET.fromstring(_with_retries(partial(_fetch, listing)))
        for item in root.iter(f"{_S3_NS}Contents"):
            key = item.findtext(f"{_S3_NS}Key")
            size = item.findtext(f"{_S3_NS}Size")
            if key:
                out.append((key, int(size or 0)))
        if root.findtext(f"{_S3_NS}IsTruncated") != "true":
            return out
        token = root.findtext(f"{_S3_NS}NextContinuationToken")


def _fetch(source: str) -> bytes:
    with urllib.request.urlopen(source, timeout=_TIMEOUT_S) as response:
        return response.read()


def _stream_to(source: str, path: Path) -> None:
    with (
        urllib.request.urlopen(source, timeout=_TIMEOUT_S) as response,
        path.open("wb") as out,
    ):
        shutil.copyfileobj(response, out, length=1 << 20)


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code >= 500 or exc.code == 429:
            return True
        if exc.code == 404:
            body = exc.read() if exc.fp is not None else b""
            return b"NoSuchBucket" in body
        return False
    return isinstance(exc, OSError)  # URLError, timeouts, connection resets


def _with_retries[T](call: Callable[[], T]) -> T:
    for attempt in range(_ATTEMPTS):
        try:
            return call()
        except OSError as exc:
            if attempt == _ATTEMPTS - 1 or not _is_transient(exc):
                raise
            time.sleep(_BACKOFF_S * 2**attempt)
    raise AssertionError("unreachable")


def _satellite_number(satellite: str | int) -> int:
    text = str(satellite).upper().removeprefix("GOES").removeprefix("-")
    number = int(text.removeprefix("G")) if text.removeprefix("G").isdigit() else -1
    if number not in _SATELLITES:
        raise ValueError(
            f"satellite must be one of G16, G17, G18, G19; got {satellite!r}."
        )
    return number


def _channel_name(channel: int | str) -> str:
    text = str(channel).upper().removeprefix("C")
    if not text.isdigit() or not 1 <= int(text) <= 16:
        raise ValueError(f"channel must be 1-16 (or 'C01'-'C16'); got {channel!r}.")
    return f"C{int(text):02d}"


def _parse_stamp(stamp: str) -> datetime:
    """``YYYYJJJHHMMSSt`` (day-of-year, tenths of a second) → UTC datetime."""
    base = datetime.strptime(stamp[:13], "%Y%j%H%M%S").replace(tzinfo=UTC)
    return base + timedelta(milliseconds=100 * int(stamp[13]))


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
