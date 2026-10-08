"""End-to-end checks against real HSD files from NOAA's public AWS bucket.

The bucket is anonymous and free, so these need only network access: they
list one archived 10-minute slot and download two small files (~3 MB, one
full-disk B13 segment; ~1 MB, the Japan-area B13 scan). Marked
``integration``: they run from the "Extended Tests" workflow or
``make test-slow``, never in the automatic CI tier::

    pytest -m integration tests/himawari
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from rasterio.warp import transform as warp_transform

from geoproducts import himawari
from geoproducts.himawari import aws


pytestmark = pytest.mark.integration

# A fixed, archived slot: the bucket keeps the full record.
SLOT = datetime(2026, 10, 7, 3, 0)


@pytest.fixture(scope="module")
def downloads(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    dest = tmp_path_factory.mktemp("himawari")
    (segment,) = aws.list_segments(start=SLOT, band=13, segments=5)
    (japan,) = aws.list_segments(start=SLOT, sector="Japan", band=13, area="JP01")
    return {
        "fldk": aws.download(segment, dest),
        "japan": aws.download(japan, dest),
        "japan_dat": aws.download(japan, dest, decompress=True),
    }


def test_full_disk_slot_has_ten_segments_per_band() -> None:
    files = aws.list_segments(start=SLOT, band="B13")
    assert [f.segment for f in files] == list(range(1, 11))
    assert all(f.total_segments == 10 and f.resolution_km == 2.0 for f in files)
    assert all(f.slot == SLOT.replace(tzinfo=UTC) and f.size for f in files)


def test_cloud_products_are_listed() -> None:
    (mask,) = aws.list_l2(start=SLOT)
    assert mask.product == "CMSK" and mask.satellite == "H09"
    assert mask.size and mask.size > 100e6


def test_segment_header_matches_the_file_name(downloads: dict[str, Path]) -> None:
    reader = himawari.Reader(downloads["fldk"], calibration="brightness_temperature")
    assert (reader.band, reader.area, reader.segments) == ("B13", "FLDK", (5,))
    assert reader.satellite == "Himawari-9"
    assert reader.shape == (1, 5500, 5500)
    assert reader.res == pytest.approx((2000.0, 2000.0), rel=1e-6)
    assert reader.satellite_lon_deg == pytest.approx(140.7)
    # Each segment carries its own scan time, within the 10-minute slot.
    slot = SLOT.replace(tzinfo=UTC)
    assert slot < reader.start_time < reader.end_time < slot + timedelta(minutes=10)


def test_brightness_temperature_and_the_disk(downloads: dict[str, Path]) -> None:
    rows = slice(2200, 2750)  # segment 5 of 10
    counts = himawari.Reader(downloads["fldk"], calibration="counts")
    raw = np.asarray(counts.load())[0, rows]
    on_disk = counts._grid.on_earth(rows, slice(0, 5500))
    # Georeferencing: every pixel the scan skipped lies off the computed
    # disk, and every pixel on it was scanned.
    outside = raw == counts.header.outside_count
    assert outside.any() and not (outside & on_disk).any()
    bt = np.asarray(
        himawari.Reader(downloads["fldk"], calibration="brightness_temperature").load()
    )[0, rows]
    assert np.isfinite(bt[on_disk]).all()
    assert np.isnan(bt[~on_disk]).all()
    assert 150.0 < np.nanmin(bt) < np.nanmax(bt) < 340.0


def test_japan_area_sits_over_japan(downloads: dict[str, Path]) -> None:
    reader = himawari.Reader(downloads["japan"])
    assert reader.area == "JP01"
    assert reader.shape == (1, 1200, 1500)
    left, bottom, right, top = reader.bounds
    lon, lat = warp_transform(
        reader.crs, "EPSG:4326", [(left + right) / 2], [(bottom + top) / 2]
    )
    assert 125.0 < lon[0] < 150.0 and 25.0 < lat[0] < 45.0


def test_decompressed_segment_reads_the_same(downloads: dict[str, Path]) -> None:
    assert downloads["japan_dat"].suffix == ".DAT"
    packed = himawari.Reader(downloads["japan"], calibration="brightness_temperature")
    plain = himawari.Reader(
        downloads["japan_dat"], calibration="brightness_temperature"
    )
    np.testing.assert_array_equal(np.asarray(plain.load()), np.asarray(packed.load()))
    # About a degree around Tokyo, read by lon/lat bounds.
    chip = plain.read_from_bounds((139.2, 35.2, 140.2, 36.2), crs_bounds="EPSG:4326")
    assert chip.crs == plain.crs
    assert 40 < chip.shape[-1] < 80  # ~1 degree of 2 km pixels
    assert np.isfinite(np.asarray(chip)).all()
