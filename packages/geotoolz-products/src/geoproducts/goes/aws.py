"""Find and fetch ABI files in NOAA's public GOES buckets on AWS.

NOAA's Open Data Dissemination programme mirrors every GOES-R product to
public, anonymous S3 buckets (``noaa-goes16`` … ``noaa-goes19``), keyed by
product and scan-start hour::

    <product>/<year>/<day-of-year>/<hour>/OR_<product>-M6C<nn>_G<nn>_s<start>_e<end>_c<created>.nc

This module maps ABI file names, satellites and time windows onto those
objects; the anonymous listing, retries and atomic downloads come from
the package's shared public-S3 client (standard library only: no
credentials, no AWS SDK).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from geoproducts._src import s3
from geoproducts._src.query import as_utc


__all__ = [
    "ABIFile",
    "bucket",
    "download",
    "list_files",
    "parse_key",
    "url",
]

# L1b radiances; L2 products follow ``ABI-L2-<code><sector letter>``
# (``ABI-L2-ACMC``, ``ABI-L2-MCMIPF``, ...).
PRODUCTS: tuple[str, ...] = ("ABI-L1b-RadF", "ABI-L1b-RadC", "ABI-L1b-RadM")
_SATELLITES = (16, 17, 18, 19)
# Transient failures (5xx, dropped connections, a spurious ``NoSuchBucket``
# for a bucket that answered the listing) are retried with exponential
# backoff; a missing key or a refused request fails at once.
_ATTEMPTS = 4
_BACKOFF_S = 1.0
_KEY_RE = re.compile(
    r"OR_(?P<product>ABI-L(?:1b|2)-[A-Za-z0-9]+?)(?P<sector>[12])?-M(?P<mode>\d)"
    r"(?:C(?P<channel>\d{2}))?_G(?P<satellite>\d{2})"
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
    channel: str | None
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
    return s3.object_url(bucket(satellite), key)


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
        channel=f"C{match['channel']}" if match["channel"] else None,
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
            ``"ABI-L1b-RadC"`` (CONUS / PACUS), ``"ABI-L1b-RadM"``
            (mesoscale), or an L2 product such as ``"ABI-L2-ACMC"`` (clear
            sky mask) or ``"ABI-L2-MCMIPF"`` (all 16 channels on one grid).
            Default ``"ABI-L1b-RadC"``.
        channel: Keep one channel (``13`` or ``"C13"``); L2 files without
            a channel are then skipped. Default: all.
        sector: Mesoscale sector, ``"M1"`` or ``"M2"``. Default: both.

    Returns:
        The matching files, sorted by scan start then channel.

    Raises:
        ValueError: ``end`` is not after ``start``, or ``channel`` /
            ``satellite`` is out of range.
        urllib.error.URLError: The bucket listing failed (transient errors
            are retried first).
    """
    start = as_utc(start)
    end = start + timedelta(hours=1) if end is None else as_utc(end)
    if end <= start:
        raise ValueError(f"end ({end}) must be after start ({start}).")
    wanted_channel = None if channel is None else _channel_name(channel)
    files: list[ABIFile] = []
    hour = start.replace(minute=0, second=0, microsecond=0)
    while hour < end:
        prefix = f"{product}/{hour:%Y}/{hour:%j}/{hour:%H}/"
        for key, size in s3.list_objects(
            bucket(satellite), prefix, attempts=_ATTEMPTS, backoff_s=_BACKOFF_S
        ):
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
    return sorted(files, key=lambda item: (item.start, item.channel or "", item.key))


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
    target = Path(dest) / item.name
    if target.exists() and not overwrite:
        return target
    return s3.download_object(
        bucket(satellite or item.satellite),
        item.key,
        target,
        attempts=_ATTEMPTS,
        backoff_s=_BACKOFF_S,
    )


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
