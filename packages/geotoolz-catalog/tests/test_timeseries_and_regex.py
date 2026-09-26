"""`load_raster_timeseries` step days and filename-regex date parsing (#219)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_bounds

from geocatalog import GeoSlice, build_raster_catalog, load_raster_timeseries
from geocatalog._src._timeutil import filename_interval
from geocatalog._src.raster import _timeseries_days


BOUNDS = (500_000.0, 4_000_000.0, 500_200.0, 4_000_200.0)


def _tif(path: Path, value: int) -> Path:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=20,
        width=20,
        count=1,
        dtype="uint16",
        crs="EPSG:32629",
        transform=from_bounds(*BOUNDS, 20, 20),
        nodata=0,
    ) as dst:
        dst.write(np.full((1, 20, 20), value, dtype=np.uint16))
    return path


def _iv(a: str, b: str) -> pd.Interval:
    return pd.Interval(pd.Timestamp(a), pd.Timestamp(b), closed="both")


def _slice(a: str, b: str) -> GeoSlice:
    return GeoSlice(BOUNDS, _iv(a, b), (10.0, 10.0), "EPSG:32629")


class TestTimeseriesDays:
    def test_sentinel_rows_add_no_step(self) -> None:
        idx = pd.IntervalIndex(
            [_iv("1900-01-01", "2100-01-01"), _iv("2024-06-02", "2024-06-02 23:59")],
            closed="both",
        )
        assert _timeseries_days(idx, _iv("2024-06-01", "2024-06-30")) == [
            pd.Timestamp("2024-06-02")
        ]

    def test_only_sentinel_rows_gives_one_step_at_slice_start(self) -> None:
        idx = pd.IntervalIndex([_iv("1900-01-01", "2100-01-01")], closed="both")
        assert _timeseries_days(idx, _iv("2024-06-05 12:00", "2024-06-30")) == [
            pd.Timestamp("2024-06-05")
        ]

    def test_row_starting_before_slice_counts_from_slice_start(self) -> None:
        idx = pd.IntervalIndex([_iv("2024-05-20", "2024-06-10")], closed="both")
        assert _timeseries_days(idx, _iv("2024-06-01", "2024-06-30")) == [
            pd.Timestamp("2024-06-01")
        ]


def test_timeseries_has_no_phantom_step_for_static_file(tmp_path: Path) -> None:
    static = _tif(tmp_path / "dem.tif", 7)
    d1 = _tif(tmp_path / "S2_20240602.tif", 1)
    d2 = _tif(tmp_path / "S2_20240605.tif", 2)
    dated = build_raster_catalog([d1, d2], filename_regex=r"S2_(?P<date>\d{8})\.tif")
    undated = build_raster_catalog([static])
    cat = dated.union(undated)

    out = load_raster_timeseries(cat, _slice("2024-06-01", "2024-06-30"), n_workers=1)
    assert np.asarray(out).shape[0] == 2


def test_sub_day_slice_is_not_widened(tmp_path: Path) -> None:
    """Two acquisitions on one day; a morning-only slice sees only the first."""
    a = _tif(tmp_path / "S2_20240602T0900.tif", 1)
    b = _tif(tmp_path / "S2_20240602T1500.tif", 2)
    rows = []
    for path, hour in ((a, 9), (b, 15)):
        cat = build_raster_catalog([path])
        t = pd.Timestamp("2024-06-02") + pd.Timedelta(hours=hour)
        cat.gdf.index = pd.IntervalIndex([pd.Interval(t, t, closed="both")])
        rows.append(cat)
    cat = rows[0].union(rows[1])

    out = load_raster_timeseries(
        cat, _slice("2024-06-02 08:00", "2024-06-02 12:00"), n_workers=1
    )
    values = np.asarray(out)
    assert values.shape[0] == 1
    assert (values == 1).all()


class TestFilenameInterval:
    PATTERN = r"(?P<date>\d{8})\.tif|(?P<start>\d{8})_(?P<stop>\d{8})\.tif"

    def test_start_stop_branch_of_alternation(self) -> None:
        import re

        groups = re.search(self.PATTERN, "x_20240601_20240603.tif").groupdict()
        assert groups["date"] is None
        start, end = filename_interval(groups, "%Y%m%d")
        assert start == pd.Timestamp("2024-06-01")
        assert end.floor("D") == pd.Timestamp("2024-06-03")

    def test_date_branch_of_alternation(self) -> None:
        import re

        groups = re.search(self.PATTERN, "x_20240601.tif").groupdict()
        start, end = filename_interval(groups, "%Y%m%d")
        assert (start, end.floor("D")) == (pd.Timestamp("2024-06-01"),) * 2

    def test_nothing_matched_raises(self) -> None:
        with pytest.raises(ValueError, match="matched groups"):
            filename_interval(
                {"date": None, "start": "20240601", "stop": None}, "%Y%m%d"
            )

    def test_builder_rows_have_real_intervals(self, tmp_path: Path) -> None:
        path = _tif(tmp_path / "S2_20240601_20240603.tif", 1)
        cat = build_raster_catalog(
            [path],
            filename_regex=r"S2_(?:(?P<date>\d{8})|(?P<start>\d{8})_(?P<stop>\d{8}))\.tif",
        )
        iv = cat.gdf.index[0]
        assert not pd.isna(iv.left) and not pd.isna(iv.right)
        assert iv.left == pd.Timestamp("2024-06-01")
