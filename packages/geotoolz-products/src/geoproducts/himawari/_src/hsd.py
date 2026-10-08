"""Himawari Standard Data (HSD): the binary format of AHI L1b segments.

Written from JMA's *Himawari Standard Data User's Guide*. One HSD file is
one segment (a band of lines) of one band of one scan. It starts with a
header of numbered blocks — each opens with its block number (``uint8``)
and length (``uint16``; ``uint32`` for block 10) — followed by the image
as ``uint16`` counts, ``lines x columns``. Files are often distributed
``bzip2``-compressed (``.DAT.bz2``).

Only the blocks a reader needs are decoded:

1. basic information — satellite, observation area, times, header length
2. data information — bits per pixel, columns, lines
3. projection — sub-satellite longitude, ``CFAC`` / ``LFAC`` / ``COFF`` /
   ``LOFF``, Earth radii and satellite distance
5. calibration — count → radiance, plus radiance → albedo (bands 1-6) or
   radiance → brightness temperature (bands 7-16)
7. segment — total segments, sequence number, first line
8. navigation correction — image rotation and per-line shifts (zero in
   JMA's operational files, which are navigated on the ground)
"""

from __future__ import annotations

import bz2
import os
import struct
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import IO, Any

import numpy as np


__all__ = ["HSDHeader", "read_header", "read_lines"]

Source = str | os.PathLike[str]

_MJD_EPOCH = datetime(1858, 11, 17, tzinfo=UTC)
# Generous upper bound on the header (JMA files use ~1.5 KB).
_HEADER_PROBE = 16_384


def _open(path: Source) -> IO[bytes]:
    """Open an HSD file, transparently decompressing ``.bz2``."""
    path = Path(path)
    if path.suffix == ".bz2":
        return bz2.open(path, "rb")
    return path.open("rb")


def _mjd(value: float) -> datetime:
    return _MJD_EPOCH + timedelta(days=value)


def _text(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("ascii", errors="replace").strip()


@dataclass(frozen=True)
class HSDHeader:
    """The decoded header fields of one HSD segment.

    Attributes:
        satellite: ``"Himawari-9"`` / ``"Himawari-8"``.
        area: Observation area (``"FLDK"``, ``"JP01"`` … ``"JP04"``,
            ``"R301"`` …).
        start_time / end_time: Observation start / end (UTC).
        header_length: Bytes before the image data.
        bits_per_pixel: Storage bits per pixel (16).
        columns / lines: Segment image size.
        sub_lon: Sub-satellite longitude of the projection (degrees east).
        cfac / lfac: Column / line scaling factors (CGMS convention).
        coff / loff: Column / line offsets, 1-based, of the area image.
        distance_km: Distance from the Earth's centre to the satellite.
        equatorial_radius_km / polar_radius_km: Earth ellipsoid.
        band: Band number (1-16).
        wavelength_um: Central wavelength (µm).
        valid_bits: Significant bits of the counts.
        error_count / outside_count: Count values marking error pixels and
            pixels outside the scan area.
        gain / offset: Count → radiance (W m-2 sr-1 µm-1).
        albedo_coefficient: Radiance → albedo (bands 1-6; ``NaN`` otherwise).
        updated_gain / updated_offset: Updated count → radiance coefficients
            of bands 1-6, ``NaN`` when absent.
        tb_coefficients: ``(c0, c1, c2)`` effective → brightness temperature
            (bands 7-16).
        planck_constants: ``(c, h, k)`` — speed of light, Planck and
            Boltzmann constants as written in the file (bands 7-16).
        total_segments / segment: Segment count and 1-based sequence number.
        first_line: 1-based line of the area image this segment starts at.
        timeline: Observation timeline ``HHMM`` of the scan (block 1): the
            nominal 10-minute slot every segment of one scan shares.
        rotation_urad: Navigation rotation correction (block 8, µrad).
        max_shift_px: Largest per-line column / line shift (block 8, pixels).
    """

    satellite: str
    area: str
    start_time: datetime
    end_time: datetime
    header_length: int
    bits_per_pixel: int
    columns: int
    lines: int
    sub_lon: float
    cfac: int
    lfac: int
    coff: float
    loff: float
    distance_km: float
    equatorial_radius_km: float
    polar_radius_km: float
    band: int
    wavelength_um: float
    valid_bits: int
    error_count: int
    outside_count: int
    gain: float
    offset: float
    albedo_coefficient: float
    updated_gain: float
    updated_offset: float
    tb_coefficients: tuple[float, float, float]
    planck_constants: tuple[float, float, float]
    total_segments: int
    segment: int
    first_line: int
    timeline: int = 0
    rotation_urad: float = 0.0
    max_shift_px: float = 0.0

    @property
    def area_lines(self) -> int:
        """Lines of the whole area image (all segments)."""
        return self.lines * self.total_segments

    @property
    def is_emissive(self) -> bool:
        """Bands 7-16 (thermal); bands 1-6 are reflective."""
        return self.band >= 7


def _blocks(raw: bytes) -> dict[int, bytes]:
    """Split the header into ``{block number: block bytes}``."""
    blocks: dict[int, bytes] = {}
    off = 0
    while off < len(raw):
        number = raw[off]
        fmt = "<I" if number == 10 else "<H"
        if off + 1 + struct.calcsize(fmt) > len(raw):
            raise ValueError(f"truncated or invalid HSD header at block {number}.")
        (length,) = struct.unpack_from(fmt, raw, off + 1)
        if length <= 0 or off + length > len(raw):
            raise ValueError(f"truncated or invalid HSD header at block {number}.")
        blocks[number] = raw[off : off + length]
        off += length
        if number == 11:
            break
    return blocks


def _parse(raw: bytes) -> HSDHeader:
    if len(raw) < 11 or raw[0] != 1:
        raise ValueError("not a Himawari Standard Data file (block 1 missing).")
    if raw[5] != 0:  # JMA distributes little-endian files only
        raise ValueError("big-endian HSD files are not supported.")
    blocks = _blocks(raw)
    missing = {1, 2, 3, 5, 7} - set(blocks)
    if missing:
        raise ValueError(f"HSD header lacks blocks {sorted(missing)}.")
    b1, b2, b3, b5, b7 = (blocks[n] for n in (1, 2, 3, 5, 7))
    start, end, _created = struct.unpack_from("<ddd", b1, 46)
    (header_length,) = struct.unpack_from("<I", b1, 70)
    bits, columns, lines, compression = struct.unpack_from("<HHHB", b2, 3)
    if compression != 0:
        raise ValueError(
            f"HSD data blocks compressed in-file (flag {compression}) are not "
            "supported; JMA / NOAA distribute uncompressed blocks (whole-file "
            ".bz2 is fine)."
        )
    (timeline,) = struct.unpack_from("<H", b1, 44)
    sub_lon, cfac, lfac, coff, loff, distance, re_km, rp_km = struct.unpack_from(
        "<dIIffddd", b3, 3
    )
    band, wavelength, valid_bits, err, outside, gain, offset = struct.unpack_from(
        "<HdHHHdd", b5, 3
    )
    tail = 3 + struct.calcsize("<HdHHHdd")
    nan = float("nan")
    albedo = upd_gain = upd_offset = nan
    tb: tuple[float, float, float] = (nan, nan, nan)
    planck: tuple[float, float, float] = (nan, nan, nan)
    if band >= 7:
        c0, c1, c2, _i0, _i1, _i2, c, h, k = struct.unpack_from("<9d", b5, tail)
        tb, planck = (c0, c1, c2), (c, h, k)
    else:
        albedo, _update_time, upd_gain, upd_offset = struct.unpack_from(
            "<dddd", b5, tail
        )
    total_segments, segment, first_line = struct.unpack_from("<BBH", b7, 3)
    rotation, max_shift = _navigation_correction(blocks.get(8))
    return HSDHeader(
        satellite=_text(b1[6:22]),
        area=_text(b1[38:42]),
        start_time=_mjd(start),
        end_time=_mjd(end),
        header_length=header_length,
        bits_per_pixel=bits,
        columns=columns,
        lines=lines,
        sub_lon=sub_lon,
        cfac=cfac,
        lfac=lfac,
        coff=coff,
        loff=loff,
        distance_km=distance,
        equatorial_radius_km=re_km,
        polar_radius_km=rp_km,
        band=band,
        wavelength_um=wavelength,
        valid_bits=valid_bits,
        error_count=err,
        outside_count=outside,
        gain=gain,
        offset=offset,
        albedo_coefficient=albedo,
        updated_gain=upd_gain if upd_gain > 0 else nan,
        updated_offset=upd_offset if upd_gain > 0 else nan,
        tb_coefficients=tb,
        planck_constants=planck,
        total_segments=total_segments,
        segment=segment,
        first_line=first_line,
        timeline=timeline,
        rotation_urad=rotation,
        max_shift_px=max_shift,
    )


def _navigation_correction(b8: bytes | None) -> tuple[float, float]:
    """Block 8: ``(rotation µrad, largest per-line shift in pixels)``."""
    if b8 is None or len(b8) < 21:
        return 0.0, 0.0
    (rotation,) = struct.unpack_from("<d", b8, 11)
    (count,) = struct.unpack_from("<H", b8, 19)
    shifts = [
        max(abs(col), abs(line))
        for _, col, line in (
            struct.unpack_from("<Hff", b8, 21 + 10 * i)
            for i in range(count)
            if 21 + 10 * (i + 1) <= len(b8)
        )
    ]
    return float(rotation), float(max(shifts, default=0.0))


def read_header(path: Source) -> HSDHeader:
    """Decode the header of an HSD segment (``.DAT`` or ``.DAT.bz2``).

    Only the first few kilobytes are read (and decompressed).

    Raises:
        ValueError: The file is not a valid little-endian HSD segment.
    """
    with _open(path) as f:
        return _parse(f.read(_HEADER_PROBE))


def read_lines(path: Source, header: HSDHeader, rows: slice, cols: slice) -> np.ndarray:
    """Counts of segment lines ``rows`` and columns ``cols`` (``uint16``).

    Uncompressed files are memory-mapped, so only the touched lines are
    read; ``.bz2`` files are decompressed up to the last requested line.

    Args:
        path: The segment file.
        header: Its decoded header.
        rows: Lines within the segment (0-based, step 1).
        cols: Columns (0-based, step 1).
    """
    start, stop = rows.start or 0, rows.stop if rows.stop is not None else header.lines
    row_bytes = header.columns * 2
    if Path(path).suffix != ".bz2":
        data: Any = np.memmap(
            path,
            dtype="<u2",
            mode="r",
            offset=header.header_length,
            shape=(header.lines, header.columns),
        )
        return np.array(data[start:stop, cols])
    with _open(path) as f:
        f.seek(header.header_length + start * row_bytes)
        buf = f.read((stop - start) * row_bytes)
    counts = np.frombuffer(buf, dtype="<u2").reshape(stop - start, header.columns)
    return counts[:, cols].copy()
