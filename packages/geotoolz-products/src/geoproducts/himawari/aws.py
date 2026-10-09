"""Find and fetch Himawari files in NOAA's public buckets on AWS.

NOAA mirrors the JMA Himawari-8 / -9 data to public, anonymous S3 buckets
(``noaa-himawari8`` for the 2015-2022 archive, ``noaa-himawari9``), keyed
by product and 10-minute observation slot::

    AHI-L1b-FLDK/<yyyy>/<mm>/<dd>/<hhmm>/HS_H09_<yyyymmdd>_<hhmm>_B13_FLDK_R20_S0510.DAT.bz2
    AHI-L1b-Japan/...                    HS_H09_..._B13_JP01_R20_S0101.DAT.bz2
    AHI-L1b-Target/...                   HS_H09_..._B13_R301_R20_S0101.DAT.bz2
    AHI-L2-FLDK-Clouds/...               AHI-CMSK_v1r1_h09_s<start>_e<end>_c<created>.nc

L1b files are one HSD segment of one band (the full disk in 10 segments;
the Japan and target areas in one, scanned four times per slot as
``JP01`` … ``JP04`` / ``R301`` … ``R304``). Listing and atomic downloads
go unsigned through ``geocloud.files`` on the shared obstore pool (no
credentials, no AWS SDK; geotoolz-cloud comes with the ``[himawari]``
extra).
"""

from __future__ import annotations

import bz2
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, get_args

from geoproducts._src import s3
from geoproducts._src.files import atomic_path
from geoproducts._src.query import as_utc


__all__ = [
    "L2File",
    "Sector",
    "Segment",
    "bucket",
    "download",
    "list_l2",
    "list_segments",
    "parse_key",
    "url",
]

Sector = Literal["FLDK", "Japan", "Target"]
_SATELLITES = (8, 9)
_SLOT = timedelta(minutes=10)
# Transient failures are retried with exponential backoff; a missing key
# or a refused request fails at once (see ``_src.s3.s3_wait``).
_ATTEMPTS = 4
_BACKOFF_S = 1.0
_SEGMENT_RE = re.compile(
    r"HS_H(?P<satellite>\d{2})_(?P<slot>\d{8}_\d{4})_B(?P<band>\d{2})"
    r"_(?P<area>FLDK|JP\d{2}|R\d{3})_R(?P<resolution>\d{2})"
    r"_S(?P<segment>\d{2})(?P<total>\d{2})\.DAT(?:\.bz2)?$"
)
_L2_RE = re.compile(
    r"AHI-(?P<product>[A-Z0-9]+)_v\d+r\d+_h(?P<satellite>\d{2})"
    r"_s(?P<start>\d{15})_e(?P<end>\d{15})_c(?P<created>\d{15})\.nc$"
)


@dataclass(frozen=True)
class Segment:
    """One L1b HSD segment in a NOAA bucket, with its file-name fields parsed.

    Attributes:
        key: Object key.
        satellite: ``"H08"`` / ``"H09"``.
        slot: Nominal 10-minute observation slot (UTC); the Japan / target
            sub-scans ``JP02`` … start 2.5, 5 and 7.5 minutes later.
        band: ``"B01"`` … ``"B16"``.
        area: ``"FLDK"``, ``"JP01"`` … ``"JP04"``, ``"R301"`` … ``"R304"``.
        resolution_km: ``0.5``, ``1.0`` or ``2.0``.
        segment: 1-based segment number.
        total_segments: Segments of the area (10 for the full disk).
        size: Object size (bytes), when listed.
    """

    key: str
    satellite: str
    slot: datetime
    band: str
    area: str
    resolution_km: float
    segment: int
    total_segments: int
    size: int | None = None

    @property
    def name(self) -> str:
        """The object's file name."""
        return self.key.rsplit("/", 1)[-1]

    @property
    def url(self) -> str:
        """Public HTTPS URL of the object."""
        return url(self.satellite, self.key)


@dataclass(frozen=True)
class L2File:
    """One NOAA AHI L2 object, with its file-name fields parsed."""

    key: str
    satellite: str
    product: str
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
    """The NOAA bucket of a Himawari satellite (``"H09"`` / ``9`` → ``noaa-himawari9``).

    Raises:
        ValueError: ``satellite`` is not Himawari-8 or -9.
    """
    return f"noaa-himawari{_satellite_number(satellite)}"


def url(satellite: str | int, key: str) -> str:
    """Public HTTPS URL of ``key`` in ``satellite``'s bucket."""
    return s3.object_url(bucket(satellite), key)


def parse_key(key: str, *, size: int | None = None) -> Segment | L2File:
    """Parse an L1b segment or L2 object key (or bare file name).

    Raises:
        ValueError: ``key`` follows neither naming convention.
    """
    match = _SEGMENT_RE.search(key)
    if match is not None:
        return Segment(
            key=key,
            satellite=f"H{match['satellite']}",
            slot=datetime.strptime(match["slot"], "%Y%m%d_%H%M").replace(tzinfo=UTC),
            band=f"B{match['band']}",
            area=match["area"],
            resolution_km=int(match["resolution"]) / 10.0,
            segment=int(match["segment"]),
            total_segments=int(match["total"]),
            size=size,
        )
    match = _L2_RE.search(key)
    if match is not None:
        return L2File(
            key=key,
            satellite=f"H{match['satellite']}",
            product=match["product"],
            start=_parse_stamp(match["start"]),
            end=_parse_stamp(match["end"]),
            created=_parse_stamp(match["created"]),
            size=size,
        )
    raise ValueError(f"not a Himawari HSD segment or L2 file name: {key!r}")


def list_segments(
    *,
    start: datetime,
    end: datetime | None = None,
    satellite: str | int = "H09",
    sector: Sector = "FLDK",
    band: int | str | None = None,
    area: str | None = None,
    segments: int | list[int] | None = None,
) -> list[Segment]:
    """List the L1b HSD segments of the 10-minute slots overlapping ``[start, end)``.

    Args:
        start: Window start, rounded down to its slot (``03:07`` lists the
            ``03:00`` slot). Naive datetimes are taken as UTC.
        end: Window end (exclusive). Default: ``start`` plus 10 minutes
            (one slot).
        satellite: ``"H09"`` (default) or ``"H08"`` (or ``9`` / ``8``).
        sector: ``"FLDK"`` (full disk, default), ``"Japan"`` or ``"Target"``.
        band: Keep one band (``13`` or ``"B13"``). Default: all 16.
        area: Keep one sub-scan (``"JP01"``, ``"R302"``, …). Default: all.
        segments: Keep these segment numbers (1-based), e.g. the full-disk
            segments over an area of interest. Default: all.

    Returns:
        The matching segments, sorted by slot, band, area and segment.

    Raises:
        ValueError: ``end`` is not after ``start``, or ``satellite`` /
            ``sector`` / ``band`` is out of range.
        ImportError: geotoolz-cloud is not installed (``[himawari]`` extra).
        Exception: The bucket listing failed (obstore's error; transient
            errors are retried first).
    """
    if sector not in get_args(Sector):
        raise ValueError(f"sector must be one of {get_args(Sector)}; got {sector!r}.")
    wanted_band = None if band is None else _band_name(band)
    wanted_segments = {segments} if isinstance(segments, int) else segments
    out: list[Segment] = []
    for key, size in _list_slots(satellite, f"AHI-L1b-{sector}", start, end):
        try:
            item = parse_key(key, size=size)
        except ValueError:
            continue
        if not isinstance(item, Segment):
            continue
        if wanted_band is not None and item.band != wanted_band:
            continue
        if area is not None and item.area != area:
            continue
        if wanted_segments is not None and item.segment not in wanted_segments:
            continue
        out.append(item)
    return sorted(out, key=lambda s: (s.slot, s.band, s.area, s.segment))


def list_l2(
    *,
    start: datetime,
    end: datetime | None = None,
    satellite: str | int = "H09",
    product: str = "CMSK",
    group: str = "Clouds",
) -> list[L2File]:
    """List NOAA AHI L2 full-disk files whose scan starts in ``[start, end)``.

    Args:
        start: Window start. Naive datetimes are taken as UTC.
        end: Window end (exclusive). Default: ``start`` plus 10 minutes.
        satellite: ``"H09"`` (default) or ``"H08"``.
        product: Product code: ``"CMSK"`` (cloud mask, default), ``"CHGT"``
            (cloud-top height), ``"CPHS"`` (cloud phase).
        group: Bucket product group (``AHI-L2-FLDK-<group>``). Default
            ``"Clouds"``.

    Returns:
        The matching files, sorted by scan start.

    Raises:
        ValueError: ``end`` is not after ``start``, or ``satellite`` is out
            of range.
        ImportError: geotoolz-cloud is not installed (``[himawari]`` extra).
        Exception: The bucket listing failed (obstore's error).
    """
    lo = as_utc(start)
    hi = lo + _SLOT if end is None else as_utc(end)
    out: list[L2File] = []
    for key, size in _list_slots(satellite, f"AHI-L2-FLDK-{group}", lo, hi):
        try:
            item = parse_key(key, size=size)
        except ValueError:
            continue
        if (
            isinstance(item, L2File)
            and item.product == product
            and lo <= item.start < hi
        ):
            out.append(item)
    return sorted(out, key=lambda f: (f.start, f.key))


def download(
    file: Segment | L2File | str,
    dest: str | Path,
    *,
    satellite: str | int | None = None,
    decompress: bool = False,
    overwrite: bool = False,
) -> Path:
    """Download one object into the directory ``dest``.

    The file is streamed to ``<name>.part`` and renamed when complete, so
    an interrupted download never leaves a truncated file behind. An
    existing file is reused unless ``overwrite`` is set.

    Args:
        file: A :class:`Segment` / :class:`L2File` from the listings, or an
            object key.
        dest: Destination directory (created if missing).
        satellite: Required when ``file`` is a key whose name does not
            identify the satellite; otherwise read from the file name.
        decompress: Store ``.DAT.bz2`` segments decompressed as ``.DAT``
            (about 3-4x larger, but memory-mapped by
            :class:`~geoproducts.himawari.Reader`, so windows read without
            decompressing the segment). A ``.bz2`` already in ``dest`` is
            decompressed without downloading it again, and kept.
        overwrite: Re-download even when the file already exists.

    Returns:
        Path of the downloaded (or decompressed) file.

    Raises:
        ImportError: geotoolz-cloud is not installed (``[himawari]`` extra).
        FileNotFoundError: No such object.
        Exception: The download failed (obstore's error; transient errors
            are retried first).
    """
    item = parse_key(file) if isinstance(file, str) else file
    name = item.name
    unpack = decompress and name.endswith(".bz2")
    target = Path(dest) / (name.removesuffix(".bz2") if unpack else name)
    if target.exists() and not overwrite:
        return target
    local = Path(dest) / name
    # Decompress a ``.bz2`` already on disk (and keep it); otherwise fetch
    # into a hidden temporary file that is removed once decompressed.
    fetch = not unpack or overwrite or not local.exists()
    compressed = local.with_name(f".{name}.download") if unpack and fetch else local
    if fetch:
        s3.download_object(
            bucket(satellite or item.satellite),
            item.key,
            compressed,
            attempts=_ATTEMPTS,
            backoff_s=_BACKOFF_S,
            extra="himawari",
        )
    if not unpack:
        return target
    try:
        with (
            bz2.open(compressed, "rb") as src,
            atomic_path(target) as tmp,
            tmp.open("wb") as out,
        ):
            shutil.copyfileobj(src, out, length=1 << 20)
    finally:
        if fetch:
            compressed.unlink(missing_ok=True)
    return target


def _list_slots(
    satellite: str | int, product: str, start: datetime, end: datetime | None
) -> list[tuple[str, int]]:
    """``(key, size)`` of every object in the 10-minute slots of ``[start, end)``.

    ``start`` rounds down to its slot; the default ``end`` is one slot later.
    """
    start = as_utc(start)
    first = start.replace(
        minute=start.minute - start.minute % 10, second=0, microsecond=0
    )
    end = first + _SLOT if end is None else as_utc(end)
    if end <= start:
        raise ValueError(f"end ({end}) must be after start ({start}).")
    name = bucket(satellite)
    slot = first
    keys: list[tuple[str, int]] = []
    while slot < end:
        prefix = f"{product}/{slot:%Y/%m/%d/%H%M}/"
        keys.extend(
            s3.list_objects(
                name, prefix, attempts=_ATTEMPTS, backoff_s=_BACKOFF_S, extra="himawari"
            )
        )
        slot += _SLOT
    return keys


def _satellite_number(satellite: str | int) -> int:
    text = str(satellite).upper().removeprefix("HIMAWARI").removeprefix("-")
    text = text.removeprefix("H")
    number = int(text) if text.isdigit() else -1
    if number not in _SATELLITES:
        raise ValueError(f"satellite must be H08 or H09; got {satellite!r}.")
    return number


def _band_name(band: int | str) -> str:
    text = str(band).upper().removeprefix("B")
    if not text.isdigit() or not 1 <= int(text) <= 16:
        raise ValueError(f"band must be 1-16 (or 'B01'-'B16'); got {band!r}.")
    return f"B{int(text):02d}"


def _parse_stamp(stamp: str) -> datetime:
    """``YYYYMMDDHHMMSSt`` (tenths of a second) → UTC datetime."""
    base = datetime.strptime(stamp[:14], "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    return base + timedelta(milliseconds=100 * int(stamp[14]))
