"""Tests for `build_raster_catalog`, `load_raster`, `load_raster_timeseries`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geotoolz.catalog import (
    build_raster_catalog,
    load_raster,
    load_raster_timeseries,
)
from geotoolz.types import GeoSlice


REGEX = r"S2_T29SND_(?P<date>\d{8}).*\.tif"


class TestBuildRasterCatalog:
    def test_one_file(self, utm29_tile_factory) -> None:
        path = utm29_tile_factory((500_000, 4_000_000, 500_320, 4_000_320), "20240115")
        catalog = build_raster_catalog([path], filename_regex=REGEX)
        assert len(catalog) == 1
        assert catalog.backend == "raster"
        bounds = catalog.total_bounds
        np.testing.assert_allclose(bounds, (500_000, 4_000_000, 500_320, 4_000_320))

    def test_two_files_one_catalog(self, utm29_tile_factory) -> None:
        path_a = utm29_tile_factory(
            (500_000, 4_000_000, 500_320, 4_000_320), "20240115"
        )
        path_b = utm29_tile_factory(
            (500_320, 4_000_000, 500_640, 4_000_320), "20240116"
        )
        catalog = build_raster_catalog([path_a, path_b], filename_regex=REGEX)
        assert len(catalog) == 2

    def test_missing_regex_match_skipped(
        self, utm29_tile_factory, tmp_path, caplog
    ) -> None:
        good = utm29_tile_factory((500_000, 4_000_000, 500_320, 4_000_320), "20240115")
        # Drop a file that doesn't match the regex.
        bogus = tmp_path / "nomatch.tif"
        bogus.write_bytes(good.read_bytes())
        catalog = build_raster_catalog([good, bogus], filename_regex=REGEX)
        assert len(catalog) == 1

    def test_no_regex_uses_sentinel_interval(self, utm29_tile_factory) -> None:
        path = utm29_tile_factory((500_000, 4_000_000, 500_320, 4_000_320), "20240115")
        catalog = build_raster_catalog([path], filename_regex=None)
        interval = catalog.gdf.index[0]
        # Sentinel interval is [1900, 2100), narrower than Timestamp.min/max.
        assert interval.left == pd.Timestamp("1900-01-01")
        assert interval.right == pd.Timestamp("2100-01-01")


class TestLoadRaster:
    def test_returns_geotensor_with_correct_shape(self, utm29_tile_factory) -> None:
        path = utm29_tile_factory(
            (500_000, 4_000_000, 500_320, 4_000_320), "20240115", value=42
        )
        catalog = build_raster_catalog([path], filename_regex=REGEX)
        sl = GeoSlice(
            bounds=(500_000, 4_000_000, 500_320, 4_000_320),
            interval=pd.Interval(
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-31"),
                closed="both",
            ),
            resolution=(10.0, 10.0),
            crs="EPSG:32629",
        )
        tensor = load_raster(catalog, sl)
        assert tensor.values.shape == (3, 32, 32)
        # All pixels are 42 in our synthetic file.
        np.testing.assert_array_equal(tensor.values, 42)

    def test_rejects_invalid_merge_method(self, utm29_tile_factory) -> None:
        path = utm29_tile_factory((500_000, 4_000_000, 500_320, 4_000_320), "20240115")
        catalog = build_raster_catalog([path], filename_regex=REGEX)
        sl = GeoSlice(
            bounds=(500_000, 4_000_000, 500_320, 4_000_320),
            interval=pd.Interval(
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-31"),
                closed="both",
            ),
            resolution=(10.0, 10.0),
            crs="EPSG:32629",
        )
        with pytest.raises(ValueError, match="merge_method"):
            load_raster(catalog, sl, merge_method="count")  # type: ignore[arg-type]

    def test_band_indexes(self, utm29_tile_factory) -> None:
        path = utm29_tile_factory(
            (500_000, 4_000_000, 500_320, 4_000_320), "20240115", n_bands=4
        )
        catalog = build_raster_catalog([path], filename_regex=REGEX)
        sl = GeoSlice(
            bounds=(500_000, 4_000_000, 500_320, 4_000_320),
            interval=pd.Interval(
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-31"),
                closed="both",
            ),
            resolution=(10.0, 10.0),
            crs="EPSG:32629",
        )
        tensor = load_raster(catalog, sl, band_indexes=[1, 3])
        assert tensor.values.shape[0] == 2


class TestLoadRasterTimeseries:
    def test_stacks_distinct_days(self, utm29_tile_factory) -> None:
        path_a = utm29_tile_factory(
            (500_000, 4_000_000, 500_320, 4_000_320), "20240115", value=1
        )
        path_b = utm29_tile_factory(
            (500_000, 4_000_000, 500_320, 4_000_320), "20240116", value=2
        )
        catalog = build_raster_catalog([path_a, path_b], filename_regex=REGEX)
        sl = GeoSlice(
            bounds=(500_000, 4_000_000, 500_320, 4_000_320),
            interval=pd.Interval(
                pd.Timestamp("2024-01-15"),
                pd.Timestamp("2024-01-17"),
                closed="both",
            ),
            resolution=(10.0, 10.0),
            crs="EPSG:32629",
        )
        tensor = load_raster_timeseries(catalog, sl)
        # Two days, 3 bands, 32x32.
        assert tensor.values.shape == (2, 3, 32, 32)
