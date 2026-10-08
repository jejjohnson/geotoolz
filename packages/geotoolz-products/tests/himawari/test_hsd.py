"""The HSD header decoder and segment line reader."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest
from _hsd import CFAC, OFFSET, START, VIS_CALIBRATION

from geoproducts.himawari import HSDHeader, read_header
from geoproducts.himawari._src.hsd import read_lines


def test_header_fields(hsd) -> None:
    counts = np.arange(25 * 100, dtype=np.uint16).reshape(25, 100)
    header = read_header(hsd(counts, segment=3, total_segments=4))
    assert isinstance(header, HSDHeader)
    assert header.satellite == "Himawari-9"
    assert header.area == "FLDK"
    assert header.band == 13
    assert header.is_emissive
    assert (header.columns, header.lines, header.area_lines) == (100, 25, 100)
    assert (header.segment, header.total_segments, header.first_line) == (3, 4, 51)
    assert header.header_length == 1523
    assert header.bits_per_pixel == 16
    assert (header.cfac, header.lfac) == (CFAC, CFAC)
    assert header.coff == header.loff == pytest.approx(OFFSET)
    assert header.sub_lon == 140.7
    assert header.distance_km == 42164.0
    assert (header.error_count, header.outside_count) == (65535, 65534)
    assert abs(header.start_time - START) < timedelta(milliseconds=1)
    assert header.end_time > header.start_time
    assert header.tb_coefficients == pytest.approx((-0.1385, 1.0006, -2.2e-6))
    assert np.isnan(header.albedo_coefficient)


def test_visible_header_carries_the_updated_calibration(hsd) -> None:
    counts = np.zeros((4, 100), dtype=np.uint16)
    header = read_header(hsd(counts, band=3))
    assert not header.is_emissive
    assert header.albedo_coefficient == pytest.approx(VIS_CALIBRATION["albedo"])
    assert (header.updated_gain, header.updated_offset) == pytest.approx(
        VIS_CALIBRATION["updated"]
    )
    assert all(np.isnan(header.tb_coefficients))
    plain = read_header(hsd(counts, band=3, updated_vis=False, name="plain.DAT"))
    assert np.isnan(plain.updated_gain) and np.isnan(plain.updated_offset)


@pytest.mark.parametrize("suffix", [".DAT", ".DAT.bz2"])
def test_read_lines_from_plain_and_compressed_files(hsd, suffix: str) -> None:
    counts = np.arange(25 * 100, dtype=np.uint16).reshape(25, 100)
    path = hsd(counts, name=f"seg{suffix}")
    header = read_header(path)
    out = read_lines(path, header, slice(3, 9), slice(10, 40))
    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, counts[3:9, 10:40])
    np.testing.assert_array_equal(
        read_lines(path, header, slice(None), slice(None)), counts
    )


def test_not_hsd(tmp_path) -> None:
    path = tmp_path / "x.DAT"
    path.write_bytes(b"\x02" + bytes(100))
    with pytest.raises(ValueError, match="not a Himawari Standard Data file"):
        read_header(path)


def test_big_endian_is_refused(hsd) -> None:
    with pytest.raises(ValueError, match="big-endian"):
        read_header(hsd(np.zeros((2, 100), np.uint16), byte_order=1))


def test_truncated_header(hsd, tmp_path) -> None:
    path = hsd(np.zeros((2, 100), np.uint16))
    cut = tmp_path / "cut.DAT"
    cut.write_bytes(path.read_bytes()[:600])
    with pytest.raises(ValueError, match="truncated or invalid HSD header"):
        read_header(cut)


def test_missing_blocks(hsd, tmp_path) -> None:
    raw = bytearray(hsd(np.zeros((2, 100), np.uint16)).read_bytes())
    # Renumber block 7 (segment information) so it is not found.
    offset = 282 + 50 + 127 + 139 + 147 + 259
    assert raw[offset] == 7
    raw[offset] = 12
    path = tmp_path / "renumbered.DAT"
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match=r"lacks blocks \[7\]"):
        read_header(path)
