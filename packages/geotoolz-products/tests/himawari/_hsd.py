"""Write synthetic Himawari Standard Data (HSD) segments for the tests.

The blocks have the lengths of JMA's files (1523-byte header); only the
fields the reader decodes are filled, the rest are zeros. The default grid
is a 100 x 100 full disk — the 2 km disk scaled down 55x — so the Earth's
limb, and the space around it, fall inside the image.
"""

from __future__ import annotations

import bz2
import struct
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


# The 2 km full disk is 5500 pixels for CFAC 20466275; scaled to SIZE.
SIZE = 100
CFAC = round(20_466_275 * SIZE / 5500)
OFFSET = SIZE / 2 + 0.5
START = datetime(2026, 10, 7, 3, 0, 20, tzinfo=UTC)
# Band 13 / band 3 calibration of a real Himawari-9 file.
IR_CALIBRATION = {
    "wavelength_um": 10.4073,
    "gain": -0.0018914,
    "offset": 15.484,
    "tb": (-0.1385, 1.0006, -2.2e-6),
    "planck": (2.99792458e8, 6.62606957e-34, 1.3806488e-23),
}
VIS_CALIBRATION = {
    "wavelength_um": 0.6399,
    "gain": 0.3051037,
    "offset": -6.1020741,
    "albedo": 0.0019279848,
    "updated": (0.30901666, -6.1803331),
}
_MJD_EPOCH = datetime(1858, 11, 17, tzinfo=UTC)


def _mjd(when: datetime) -> float:
    return (when - _MJD_EPOCH).total_seconds() / 86_400.0


def _block(number: int, length: int, body: bytes) -> bytes:
    head = struct.pack("<BI" if number == 10 else "<BH", number, length)
    out = head + body
    assert len(out) <= length, (number, len(out), length)
    return out + bytes(length - len(out))


def write_hsd(
    path: Path,
    counts: np.ndarray,
    *,
    band: int = 13,
    segment: int = 1,
    total_segments: int = 1,
    area: str = "FLDK",
    satellite: str = "Himawari-9",
    columns: int | None = None,
    cfac: int = CFAC,
    coff: float = OFFSET,
    loff: float = OFFSET,
    start: datetime = START,
    updated_vis: bool = True,
    byte_order: int = 0,
    timeline: int = 300,
    compression: int = 0,
    rotation_urad: float = 0.0,
    shift_px: float = 0.0,
    gain: float | None = None,
    error_block_length: int = 47,
    compress: bool | None = None,
) -> Path:
    """Write one segment; ``counts`` is its ``(lines, columns)`` image.

    ``first_line`` follows from ``segment`` (equal-height segments).
    Compressed when the name ends in ``.bz2`` (or ``compress=True``).
    """
    counts = np.asarray(counts, dtype="<u2")
    lines, width = counts.shape
    columns = width if columns is None else columns
    first_line = (segment - 1) * lines + 1
    header_length = 1523 - 47 + error_block_length
    b1 = _block(
        1,
        282,
        struct.pack("<HB", 11, byte_order)
        + satellite.encode().ljust(16, b"\0")
        + b"MSC".ljust(16, b"\0")
        + area.encode().ljust(4, b"\0")
        + bytes(2)
        + struct.pack("<H", timeline)
        + struct.pack("<ddd", _mjd(start), _mjd(start) + 0.0005, _mjd(start) + 0.003)
        + struct.pack("<II", header_length, counts.nbytes),
    )
    b2 = _block(2, 50, struct.pack("<HHHB", 16, columns, lines, compression))
    b3 = _block(
        3,
        127,
        struct.pack(
            "<dIIffddd", 140.7, cfac, cfac, coff, loff, 42164.0, 6378.137, 6356.7523
        ),
    )
    b4 = _block(4, 139, b"")
    if band >= 7:
        c = IR_CALIBRATION
        body5 = struct.pack(
            "<HdHHHdd",
            band,
            c["wavelength_um"],
            12,
            65535,
            65534,
            c["gain"] if gain is None else gain,
            c["offset"],
        ) + struct.pack("<9d", *c["tb"], 0.0, 0.0, 0.0, *c["planck"])
    else:
        c = VIS_CALIBRATION
        updated = c["updated"] if updated_vis else (0.0, 0.0)
        body5 = struct.pack(
            "<HdHHHdd",
            band,
            c["wavelength_um"],
            11,
            65535,
            65534,
            c["gain"] if gain is None else gain,
            c["offset"],
        ) + struct.pack("<dddd", c["albedo"], _mjd(start), *updated)
    b5 = _block(5, 147, body5)
    b6 = _block(6, 259, b"")
    b7 = _block(7, 47, struct.pack("<BBH", total_segments, segment, first_line))
    b8 = _block(
        8,
        81,
        struct.pack("<ffdH", 1.0, 1.0, rotation_urad, 2)
        + struct.pack("<Hff", first_line, shift_px, 0.0)
        + struct.pack("<Hff", first_line + lines - 1, 0.0, 0.0),
    )
    b9 = _block(9, 85, b"")
    b10 = _block(10, error_block_length, b"")
    b11 = _block(11, 259, b"")
    header = b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9 + b10 + b11
    assert len(header) == header_length
    payload = header + counts.tobytes()
    if compress if compress is not None else path.suffix == ".bz2":
        payload = bz2.compress(payload)
    path.write_bytes(payload)
    return path


def segment_name(band: int, segment: int, total: int, *, area: str = "FLDK") -> str:
    """A JMA-style file name for a synthetic segment."""
    return f"HS_H09_20261007_0300_B{band:02d}_{area}_R20_S{segment:02d}{total:02d}.DAT"
