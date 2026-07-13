"""Tests for `build_raster_catalog`, `load_raster`, `load_raster_timeseries`."""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import pytest

import geocatalog._src.raster as raster_module
from geocatalog import (
    GeoSlice,
    build_raster_catalog,
    load_raster,
    load_raster_timeseries,
)


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
    def test_stacks_distinct_days_in_time_order(
        self, utm29_tile_factory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
        original_load_raster = raster_module.load_raster
        # Deterministic out-of-order completion: both workers rendezvous at
        # the barrier so they are concurrently in-flight, then the later day
        # (2024-01-16) returns first while the earlier day (2024-01-15) only
        # proceeds once the later one has signaled completion.
        barrier = threading.Barrier(2)
        later_done = threading.Event()

        def delay_first_day_load(catalog, slice_, **kwargs):
            barrier.wait()
            if slice_.interval.left == pd.Timestamp("2024-01-15"):
                later_done.wait()
            result = original_load_raster(catalog, slice_, **kwargs)
            if slice_.interval.left == pd.Timestamp("2024-01-16"):
                later_done.set()
            return result

        monkeypatch.setattr(raster_module, "load_raster", delay_first_day_load)

        tensor = load_raster_timeseries(catalog, sl, n_workers=2)
        # Two days, 3 bands, 32x32.
        assert tensor.values.shape == (2, 3, 32, 32)
        np.testing.assert_array_equal(tensor.values[0], 1)
        np.testing.assert_array_equal(tensor.values[1], 2)

    def test_missing_day_skip_and_raise(
        self, utm29_tile_factory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path_a = utm29_tile_factory(
            (500_000, 4_000_000, 500_320, 4_000_320), "20240115", value=1
        )
        path_b = utm29_tile_factory(
            (500_000, 4_000_000, 500_320, 4_000_320), "20240116", value=2
        )
        path_c = utm29_tile_factory(
            (500_000, 4_000_000, 500_320, 4_000_320), "20240117", value=3
        )
        catalog = build_raster_catalog([path_a, path_b, path_c], filename_regex=REGEX)
        sl = GeoSlice(
            bounds=(500_000, 4_000_000, 500_320, 4_000_320),
            interval=pd.Interval(
                pd.Timestamp("2024-01-15"),
                pd.Timestamp("2024-01-18"),
                closed="both",
            ),
            resolution=(10.0, 10.0),
            crs="EPSG:32629",
        )
        original_load_raster = raster_module.load_raster

        def raise_on_middle_day(catalog, slice_, **kwargs):
            if slice_.interval.left == pd.Timestamp("2024-01-16"):
                raise ValueError("missing middle day")
            return original_load_raster(catalog, slice_, **kwargs)

        monkeypatch.setattr(raster_module, "load_raster", raise_on_middle_day)

        tensor = load_raster_timeseries(catalog, sl, n_workers=2)
        assert tensor.values.shape == (2, 3, 32, 32)
        np.testing.assert_array_equal(tensor.values[0], 1)
        np.testing.assert_array_equal(tensor.values[1], 3)
        with pytest.raises(ValueError, match="missing middle day"):
            load_raster_timeseries(catalog, sl, n_workers=2, on_missing_day="raise")

    def test_n_workers_one_matches_concurrent(self, utm29_tile_factory) -> None:
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

        serial = load_raster_timeseries(catalog, sl, n_workers=1)
        concurrent = load_raster_timeseries(catalog, sl, n_workers=2)
        np.testing.assert_array_equal(serial.values, concurrent.values)


class TestLoadRasterConcurrentOpen:
    """The file-open phase runs in a thread pool (gh #3)."""

    @staticmethod
    def _four_tile_setup(utm29_tile_factory):
        paths = [
            utm29_tile_factory((500_000, 4_000_000, 500_320, 4_000_320), "20240115"),
            utm29_tile_factory((500_320, 4_000_000, 500_640, 4_000_320), "20240115"),
            utm29_tile_factory((500_000, 4_000_320, 500_320, 4_000_640), "20240115"),
            utm29_tile_factory((500_320, 4_000_320, 500_640, 4_000_640), "20240115"),
        ]
        catalog = build_raster_catalog(paths, filename_regex=REGEX)
        sl = GeoSlice(
            bounds=(500_000, 4_000_000, 500_640, 4_000_640),
            interval=pd.Interval(
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-31"),
                closed="both",
            ),
            resolution=(10.0, 10.0),
            crs="EPSG:32629",
        )
        return catalog, sl

    def test_parallel_matches_serial(self, utm29_tile_factory) -> None:
        catalog, sl = self._four_tile_setup(utm29_tile_factory)
        serial = load_raster(catalog, sl, max_open_workers=1)
        parallel = load_raster(catalog, sl, max_open_workers=8)
        np.testing.assert_array_equal(serial.values, parallel.values)
        assert serial.transform == parallel.transform

    def test_opens_actually_run_concurrently(
        self, utm29_tile_factory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        catalog, sl = self._four_tile_setup(utm29_tile_factory)
        seen_threads: set[int] = set()
        original_open = raster_module.rasterio.open

        def tracking_open(*args, **kwargs):
            seen_threads.add(threading.get_ident())
            return original_open(*args, **kwargs)

        monkeypatch.setattr(raster_module.rasterio, "open", tracking_open)
        load_raster(catalog, sl, max_open_workers=4)
        assert len(seen_threads) > 1

    def test_failed_open_closes_other_handles(
        self, utm29_tile_factory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        catalog, sl = self._four_tile_setup(utm29_tile_factory)
        opened: list = []
        original_open = raster_module.rasterio.open
        calls = {"n": 0}

        def flaky_open(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("synthetic open failure")
            handle = original_open(*args, **kwargs)
            opened.append(handle)
            return handle

        monkeypatch.setattr(raster_module.rasterio, "open", flaky_open)
        with pytest.raises(OSError, match="synthetic open failure"):
            load_raster(catalog, sl, max_open_workers=4, retries=0)
        assert opened  # some files did open before the failure...
        assert all(h.closed for h in opened)  # ...and all were closed

    def test_aload_raster_parity(self, utm29_tile_factory) -> None:
        import asyncio

        from geocatalog import aload_raster

        catalog, sl = self._four_tile_setup(utm29_tile_factory)
        sync_tensor = load_raster(catalog, sl)
        async_tensor = asyncio.run(aload_raster(catalog, sl, concurrency=4))
        np.testing.assert_array_equal(sync_tensor.values, async_tensor.values)
        assert sync_tensor.transform == async_tensor.transform
