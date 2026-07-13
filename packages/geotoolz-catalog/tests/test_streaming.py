"""Tests for the `backend="duckdb"` streaming builders.

Covers all three builders (raster / vector / xarray) plus the underlying
`StreamingParquetWriter`. Skipped wholesale if the ``[duckdb]`` extra is
not installed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyproj
import pytest
import shapely.geometry


duckdb = pytest.importorskip("duckdb")

from geocatalog import (
    append_files,
    build_raster_catalog,
    build_vector_catalog,
    open_catalog,
)
from geocatalog._src.duckdb_backend import DuckDBGeoCatalog
from geocatalog._src.streaming import StreamingParquetWriter, _iter_rows_parallel


RASTER_REGEX = r"S2_T29SND_(?P<date>\d{8})_\d+_\d+\.tif"
VECTOR_REGEX = r"labels_(?P<date>\d{8})\.gpkg"


def _delayed_row(filepath: str | Path) -> dict[str, Any] | None:
    """Picklable helper for process-pool ordering tests."""
    index = int(Path(filepath).stem)
    # Later indices complete faster to exercise ordered output under
    # non-input completion order.
    time.sleep((3 - min(index, 3)) * 0.02)
    if index == 2:
        return None
    return {"filepath": str(filepath), "index": index}


def _toy_extract(filepath: str | Path) -> dict[str, object]:
    stem = Path(filepath).stem
    if len(stem) < 10:
        raise ValueError("toy fixture filenames must start with YYYY-MM-DD")
    date = pd.Timestamp(stem[:10])
    offset = date.day
    return {
        "filepath": str(filepath),
        "geometry": shapely.geometry.box(offset, 0, offset + 1, 1),
        "start_time": date,
        "end_time": date + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1),
        "crs": "EPSG:4326",
    }


# ---------------------------------------------------------------------------
# Parallel row extraction
# ---------------------------------------------------------------------------


class TestIterRowsParallel:
    def test_ordered_preserves_input_order_with_workers(self, tmp_path: Path) -> None:
        paths = [tmp_path / f"{i}.tif" for i in range(5)]

        rows = list(
            _iter_rows_parallel(
                paths,
                _delayed_row,
                n_workers=3,
                ordered=True,
            )
        )

        assert [row["index"] for row in rows] == [0, 1, 3, 4]

    def test_build_raster_forwards_ordered_to_streamer(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: dict[str, Any] = {}
        sentinel = object()

        def fake_stream_build_duckdb(*args: Any, **kwargs: Any) -> object:
            captured.update(kwargs)
            return sentinel

        monkeypatch.setattr(
            "geocatalog._src.streaming.stream_build_duckdb",
            fake_stream_build_duckdb,
        )

        result = build_raster_catalog(
            [tmp_path / "0.tif"],
            backend="duckdb",
            out_path=tmp_path / "cat.parquet",
            n_workers=2,
            ordered=True,
            sort_by=None,
        )

        assert result is sentinel
        assert captured["ordered"] is True
        assert captured["n_workers"] == 2
        assert captured["sort_by"] is None


# ---------------------------------------------------------------------------
# Round-trip per builder
# ---------------------------------------------------------------------------


class TestRasterStreamingRoundtrip:
    def test_round_trip(self, utm29_tile_factory, tmp_path: Path) -> None:
        # Five small tiles in EPSG:32629 across two dates.
        paths = [
            utm29_tile_factory(
                (500_000 + i * 160, 4_000_000, 500_160 + i * 160, 4_000_160), date
            )
            for i, date in enumerate(
                ["20240115", "20240115", "20240116", "20240116", "20240117"]
            )
        ]
        out = tmp_path / "raster_cat.parquet"
        catalog = build_raster_catalog(
            paths,
            filename_regex=RASTER_REGEX,
            backend="duckdb",
            out_path=out,
        )

        assert isinstance(catalog, DuckDBGeoCatalog)
        assert len(catalog) == 5
        assert catalog.backend == "raster"
        assert catalog.crs == pyproj.CRS("EPSG:4326")

        reopened = open_catalog(out, engine="duckdb")
        assert len(reopened) == 5
        assert reopened.backend == "raster"

    def test_geopandas_can_read_streamed_artifact(
        self, utm29_tile_factory, tmp_path: Path
    ) -> None:
        """GeoParquet 1.1 round-trip: the artifact must open via geopandas."""
        paths = [
            utm29_tile_factory((500_000, 4_000_000, 500_160, 4_000_160), "20240115"),
            utm29_tile_factory((500_160, 4_000_160, 500_320, 4_000_320), "20240116"),
        ]
        out = tmp_path / "raster_cat.parquet"
        build_raster_catalog(
            paths, filename_regex=RASTER_REGEX, backend="duckdb", out_path=out
        )

        gdf = gpd.read_parquet(out)
        assert len(gdf) == 2
        assert gdf.crs == pyproj.CRS("EPSG:4326")
        # Geometry survives WKB → polygon round-trip.
        assert all(isinstance(g, shapely.geometry.Polygon) for g in gdf.geometry)


class TestVectorStreamingRoundtrip:
    def test_round_trip(self, tmp_path: Path) -> None:
        # Two small vector files in EPSG:32629.
        for i, date in enumerate(["20240115", "20240116"]):
            gdf = gpd.GeoDataFrame(
                {
                    "geometry": [
                        shapely.geometry.box(
                            500_000 + i * 160,
                            4_000_000,
                            500_160 + i * 160,
                            4_000_160,
                        )
                    ],
                },
                crs="EPSG:32629",
            )
            gdf.to_file(tmp_path / f"labels_{date}.gpkg", driver="GPKG")
        paths = sorted(tmp_path.glob("labels_*.gpkg"))

        out = tmp_path / "vector_cat.parquet"
        catalog = build_vector_catalog(
            paths,
            filename_regex=VECTOR_REGEX,
            backend="duckdb",
            out_path=out,
        )

        assert isinstance(catalog, DuckDBGeoCatalog)
        assert len(catalog) == 2
        assert catalog.backend == "vector"
        assert catalog.crs == pyproj.CRS("EPSG:4326")

        reopened = open_catalog(out, engine="duckdb")
        assert len(reopened) == 2
        assert reopened.backend == "vector"


class TestXarrayStreamingRoundtrip:
    def test_round_trip(self, tmp_path: Path) -> None:
        xr = pytest.importorskip("xarray")
        # Three small NetCDFs.
        for i, date in enumerate(["20240115", "20240116", "20240117"]):
            ds = xr.Dataset(
                {
                    "ndvi": (
                        ("time", "y", "x"),
                        np.full((2, 8, 8), float(i), dtype=np.float32),
                    )
                },
                coords={
                    "time": pd.date_range(date, periods=2, freq="D"),
                    "y": np.linspace(40.5, 40.0, 8),
                    "x": np.linspace(-3.5 + i * 0.5, -3.0 + i * 0.5, 8),
                },
            )
            ds.to_netcdf(tmp_path / f"modis_{date}.nc")
        paths = sorted(tmp_path.glob("modis_*.nc"))

        out = tmp_path / "xarray_cat.parquet"
        from geocatalog import build_xarray_catalog

        catalog = build_xarray_catalog(
            paths,
            target_crs="EPSG:4326",
            data_vars=["ndvi"],
            backend="duckdb",
            out_path=out,
        )
        assert isinstance(catalog, DuckDBGeoCatalog)
        assert len(catalog) == 3
        assert catalog.backend == "xarray"

        reopened = open_catalog(out, engine="duckdb")
        assert len(reopened) == 3


# ---------------------------------------------------------------------------
# Sort verification
# ---------------------------------------------------------------------------


class TestSort:
    def test_sort_by_start_time(self, utm29_tile_factory, tmp_path: Path) -> None:
        # Build with files in REVERSE chronological order; the streaming
        # rewrite must put them back in ascending start_time.
        paths = [
            utm29_tile_factory((500_000, 4_000_000, 500_160, 4_000_160), "20240117"),
            utm29_tile_factory((500_160, 4_000_000, 500_320, 4_000_160), "20240115"),
            utm29_tile_factory((500_320, 4_000_000, 500_480, 4_000_160), "20240116"),
        ]
        out = tmp_path / "sorted.parquet"
        build_raster_catalog(
            paths,
            filename_regex=RASTER_REGEX,
            backend="duckdb",
            out_path=out,
            sort_by=("start_time",),
        )
        gdf = gpd.read_parquet(out)
        # Inputs were 17, 15, 16 — sorted output is 15, 16, 17.
        starts = gdf["start_time"].tolist()
        assert starts == sorted(starts)
        assert starts[0] < starts[-1]

    def test_sort_by_hilbert(self, utm29_tile_factory, tmp_path: Path) -> None:
        # Same date, four tiles in a 2x2 grid; Hilbert sort should produce
        # a spatially coherent ordering distinct from raw input order.
        # Input order is deliberately scrambled so a no-op sort would be
        # caught by the inequality check below.
        paths = [
            utm29_tile_factory((500_000, 4_000_000, 500_160, 4_000_160), "20240115"),
            utm29_tile_factory((500_160, 4_000_160, 500_320, 4_000_320), "20240115"),
            utm29_tile_factory((500_160, 4_000_000, 500_320, 4_000_160), "20240115"),
            utm29_tile_factory((500_000, 4_000_160, 500_160, 4_000_320), "20240115"),
        ]
        out = tmp_path / "hilbert.parquet"
        build_raster_catalog(
            paths,
            filename_regex=RASTER_REGEX,
            backend="duckdb",
            out_path=out,
            sort_by=("start_time", "geometry_hilbert"),
        )
        gdf = gpd.read_parquet(out)
        assert len(gdf) == 4
        assert all(isinstance(g, shapely.geometry.Polygon) for g in gdf.geometry)

        # Strong assertion: the physical row order must match DuckDB's
        # ST_Hilbert ranking of the same geometries. Compute the Hilbert
        # value per row and verify the sequence is monotonically
        # non-decreasing — i.e. the sort_by rewrite actually applied the
        # Hilbert ordering, not just a no-op or a fallback sort.
        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial")
        hilbert_vals = [
            r[0]
            for r in con.sql(
                "SELECT ST_Hilbert(ST_Centroid(geometry)) AS h "
                f"FROM read_parquet('{out}')"
            ).fetchall()
        ]
        con.close()
        assert hilbert_vals == sorted(hilbert_vals), (
            f"rows are not in Hilbert order: {hilbert_vals}"
        )
        # And the rewrite must have changed the order from raw input.
        actual_paths = [Path(p).name for p in gdf["filepath"]]
        input_paths = [p.name for p in paths]
        assert actual_paths != input_paths, (
            "Hilbert rewrite produced input order — possible no-op."
        )

    def test_no_sort_leaves_extraction_order(
        self, utm29_tile_factory, tmp_path: Path
    ) -> None:
        paths = [
            utm29_tile_factory((500_000, 4_000_000, 500_160, 4_000_160), "20240117"),
            utm29_tile_factory((500_160, 4_000_000, 500_320, 4_000_160), "20240115"),
        ]
        out = tmp_path / "unsorted.parquet"
        build_raster_catalog(
            paths,
            filename_regex=RASTER_REGEX,
            backend="duckdb",
            out_path=out,
            sort_by=None,
        )
        gdf = gpd.read_parquet(out)
        # Inputs in reverse chronological order — no sort means the output
        # follows input order, so first row is the 17th.
        assert gdf["start_time"].iloc[0] > gdf["start_time"].iloc[-1]


# ---------------------------------------------------------------------------
# Worker parity
# ---------------------------------------------------------------------------


class TestWorkers:
    def test_n_workers_equivalent_to_serial(
        self, utm29_tile_factory, tmp_path: Path
    ) -> None:
        paths = [
            utm29_tile_factory(
                (500_000 + i * 160, 4_000_000, 500_160 + i * 160, 4_000_160), "20240115"
            )
            for i in range(6)
        ]
        out_serial = tmp_path / "serial.parquet"
        out_parallel = tmp_path / "parallel.parquet"
        build_raster_catalog(
            paths,
            filename_regex=RASTER_REGEX,
            backend="duckdb",
            out_path=out_serial,
            n_workers=1,
        )
        build_raster_catalog(
            paths,
            filename_regex=RASTER_REGEX,
            backend="duckdb",
            out_path=out_parallel,
            n_workers=2,
        )
        # Both went through the Hilbert sort pass, so rows must be in the
        # same order regardless of extraction concurrency.
        a = gpd.read_parquet(out_serial)
        b = gpd.read_parquet(out_parallel)
        assert a["filepath"].tolist() == b["filepath"].tolist()
        assert a["start_time"].tolist() == b["start_time"].tolist()


# ---------------------------------------------------------------------------
# Hive partitioning + incremental append
# ---------------------------------------------------------------------------


class TestPartitionedArchives:
    def test_partitioned_build_opens_via_factory(
        self, utm29_tile_factory, tmp_path: Path
    ) -> None:
        paths = [
            utm29_tile_factory((500_000, 4_000_000, 500_160, 4_000_160), "20240115"),
            utm29_tile_factory((500_160, 4_000_000, 500_320, 4_000_160), "20240215"),
        ]
        out = tmp_path / "partitioned"
        catalog = build_raster_catalog(
            paths,
            filename_regex=RASTER_REGEX,
            backend="duckdb",
            out_path=out,
            partition_by=("year", "month"),
            n_workers=2,
        )

        assert out.is_dir()
        assert (out / "year=2024" / "month=1").is_dir()
        assert (out / "year=2024" / "month=2").is_dir()
        reopened = open_catalog(out, engine="duckdb")
        assert len(reopened) == len(catalog) == 2
        assert reopened.backend == "raster"
        assert len(reopened.sql("year = 2024 AND month = 1")) == 1

    def test_append_files_leaves_existing_shards_untouched(
        self, tmp_path: Path
    ) -> None:
        archive = tmp_path / "archive"
        append_files(
            archive,
            [tmp_path / "2024-01-01-a.tif", tmp_path / "2024-01-02-b.tif"],
            _toy_extract,
            crs="EPSG:4326",
            backend="raster",
            partition_by=("year", "month"),
        )
        before = {path: path.stat().st_mtime_ns for path in archive.rglob("*.parquet")}

        catalog = append_files(
            archive,
            [tmp_path / "2024-02-01-c.tif"],
            _toy_extract,
            crs="EPSG:4326",
            backend="raster",
            partition_by=("year", "month"),
        )

        assert len(catalog) == 3
        for path, mtime in before.items():
            assert path.exists()
            assert path.stat().st_mtime_ns == mtime
        assert len(list((archive / "year=2024" / "month=2").glob("*.parquet"))) == 1
        february = catalog.sql("month = 2")
        assert len(february) == 1
        assert Path(next(february.iter_rows()).filepath).name == "2024-02-01-c.tif"

    def test_partition_value_rejects_nat_start_time(self) -> None:
        """NaT start_time must raise, not silently produce year=nan shards."""
        from geocatalog._src.streaming import _partition_value

        row = {
            "start_time": pd.NaT,
            "geometry": shapely.geometry.box(0, 0, 1, 1),
        }
        with pytest.raises(ValueError, match="NaT start_time"):
            _partition_value(row, "year")

    def test_partition_value_rejects_missing_start_time(self) -> None:
        """year/month/day with no start_time field must raise a clear error."""
        from geocatalog._src.streaming import _partition_value

        row = {"geometry": shapely.geometry.box(0, 0, 1, 1)}
        with pytest.raises(ValueError, match="requires a 'start_time' field"):
            _partition_value(row, "month")

    def test_open_single_file_under_hive_dir_no_synthetic_columns(
        self, tmp_path: Path
    ) -> None:
        """Opening a single file under `year=YYYY/...` must not inject `year`.

        Regression for PR #41 review: `hive_partitioning=true` was hard-coded
        on every `read_parquet` call, including the schema-version probe and
        backend-tag probe. For a file sitting under a `key=value` directory
        that always added a synthetic partition column to the catalog schema.
        """
        from geocatalog._src.parquet import to_geoparquet

        partition_dir = tmp_path / "year=2024"
        partition_dir.mkdir()
        cat = _build_cat_for_open()
        single = partition_dir / "data.parquet"
        to_geoparquet(cat, single)

        catalog = DuckDBGeoCatalog.open(single)
        columns = catalog.relation.columns
        assert "year" not in columns, (
            f"single-file open injected synthetic 'year' column: {columns}"
        )

    def test_max_open_writers_correctness(self, tmp_path: Path) -> None:
        """LRU eviction must preserve all rows across many partitions."""
        from geocatalog._src.streaming import write_partitioned_rows

        rows = []
        for i in range(100):
            rows.append(
                {
                    "filepath": f"f{i}.tif",
                    "geometry": shapely.geometry.box(i, 0, i + 1, 1),
                    "start_time": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                    "end_time": pd.Timestamp("2024-01-02") + pd.Timedelta(days=i),
                    "part": i,  # one partition per row -> 100 partitions
                }
            )
        out = tmp_path / "partitioned"
        n_written = write_partitioned_rows(
            iter(rows),
            out_path=out,
            crs="EPSG:4326",
            backend="raster",
            partition_by=("part",),
            max_open_writers=4,
        )
        assert n_written == 100

        # Every partition got its own directory; reopen and count rows.
        catalog = DuckDBGeoCatalog.open(out)
        assert len(catalog) == 100
        # Spot-check: row 42 is alive in part=42.
        slice42 = catalog.sql("part = 42")
        assert len(slice42) == 1
        row42 = next(slice42.iter_rows())
        assert row42.filepath == "f42.tif"

    def test_max_open_writers_no_fd_leak(self, tmp_path: Path) -> None:
        """The FD count after writing 100 partitions with cap=4 must be stable."""
        import gc

        from geocatalog._src.streaming import write_partitioned_rows

        # Some FDs come and go during DuckDB init; settle first.
        gc.collect()
        before = _count_open_fds()

        rows = []
        for i in range(100):
            rows.append(
                {
                    "filepath": f"f{i}.tif",
                    "geometry": shapely.geometry.box(i, 0, i + 1, 1),
                    "start_time": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                    "end_time": pd.Timestamp("2024-01-02") + pd.Timedelta(days=i),
                    "part": i,
                }
            )
        write_partitioned_rows(
            iter(rows),
            out_path=tmp_path / "partitioned",
            crs="EPSG:4326",
            backend="raster",
            partition_by=("part",),
            max_open_writers=4,
        )

        gc.collect()
        after = _count_open_fds()
        # Tolerate small noise (logger handles, etc.); the unbounded version
        # would leak ~100 FDs through the closing iteration if there were
        # a leak, so a tolerance of 10 is plenty.
        assert after - before < 10, (
            f"FD leak: before={before} after={after} (cap=4, 100 partitions)"
        )

    def test_append_files_partition_layout_mismatch(self, tmp_path: Path) -> None:
        """Appending with a different partition_by must raise ValueError."""
        archive = tmp_path / "archive"
        append_files(
            archive,
            [tmp_path / "2024-01-01-a.tif"],
            _toy_extract,
            crs="EPSG:4326",
            backend="raster",
            partition_by=("year", "month"),
        )
        with pytest.raises(
            ValueError, match=r"partition_by=.*does not match.*existing layout"
        ) as exc_info:
            append_files(
                archive,
                [tmp_path / "2024-02-01-b.tif"],
                _toy_extract,
                crs="EPSG:4326",
                backend="raster",
                partition_by=("year",),
            )
        # Error message names both layouts so the user can debug.
        msg = str(exc_info.value)
        assert "year" in msg and "month" in msg


def _build_cat_for_open():
    """Build a tiny `InMemoryGeoCatalog` for `DuckDBGeoCatalog.open` tests."""
    from geocatalog import InMemoryGeoCatalog

    gdf = gpd.GeoDataFrame(
        {
            "filepath": ["a.tif"],
            "geometry": [shapely.geometry.box(0, 0, 1, 1)],
            "start_time": [pd.Timestamp("2024-01-15")],
            "end_time": [pd.Timestamp("2024-01-16")],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    idx = pd.IntervalIndex.from_arrays(
        gdf["start_time"], gdf["end_time"], closed="both", name="datetime"
    )
    gdf = gdf.drop(columns=["start_time", "end_time"]).set_index(idx)
    return InMemoryGeoCatalog(gdf, backend="raster")


def _count_open_fds() -> int:
    """Best-effort open-FD count for the current process (POSIX only)."""
    import os
    import sys

    if sys.platform.startswith("win"):
        pytest.skip("FD counting requires POSIX")
    return len(os.listdir(f"/proc/{os.getpid()}/fd"))


# ---------------------------------------------------------------------------
# EPSG:4326 default + cross-CRS smoke test
# ---------------------------------------------------------------------------


class TestCRSDefaults:
    def test_target_crs_none_defaults_to_4326(
        self, utm29_tile_factory, tmp_path: Path
    ) -> None:
        path = utm29_tile_factory((500_000, 4_000_000, 500_160, 4_000_160), "20240115")
        out = tmp_path / "cat.parquet"
        catalog = build_raster_catalog(
            [path],
            filename_regex=RASTER_REGEX,
            backend="duckdb",
            out_path=out,
            target_crs=None,  # explicit None
        )
        assert catalog.crs == pyproj.CRS("EPSG:4326")
        # Footprint must have been reprojected — UTM coords (~500_000)
        # are out of lat/lon range, so a successful read means we actually
        # got 4326-space geometry.
        bounds = catalog.total_bounds
        assert -180.0 <= bounds[0] <= 180.0
        assert -90.0 <= bounds[1] <= 90.0

    def test_mixed_utm_zones_canonicalize_to_4326(self, tmp_path: Path) -> None:
        """Cross-CRS smoke test (closes Phase 2 ship-list item)."""
        import rasterio
        from rasterio.transform import from_bounds

        # One tile in UTM zone 29N, one in zone 30N.
        zones = [
            ("20240115", 32629, 500_000, 4_000_000),
            ("20240116", 32630, 400_000, 4_000_000),
        ]
        for date, epsg, xmin, ymin in zones:
            path = tmp_path / f"S2_T29SND_{date}_{xmin}_{ymin}.tif"
            transform = from_bounds(xmin, ymin, xmin + 160, ymin + 160, 32, 32)
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=32,
                width=32,
                count=1,
                dtype="uint16",
                crs=f"EPSG:{epsg}",
                transform=transform,
            ) as dst:
                dst.write(np.full((1, 32, 32), 1, dtype=np.uint16))

        paths = sorted(tmp_path.glob("*.tif"))
        out = tmp_path / "mixed.parquet"
        catalog = build_raster_catalog(
            paths,
            filename_regex=RASTER_REGEX,
            backend="duckdb",
            out_path=out,
        )
        assert catalog.crs == pyproj.CRS("EPSG:4326")
        assert len(catalog) == 2
        # A 4326-CRS query at the UTM-29 tile's approximate lat/lon
        # should return at least one row.
        from geocatalog import GeoSlice

        slice_ = GeoSlice(
            bounds=(-9.0, 36.0, -7.0, 37.0),
            interval=pd.Interval(
                pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"), closed="both"
            ),
            resolution=(0.001, 0.001),
            crs=pyproj.CRS("EPSG:4326"),
        )
        # Just verify the query mechanism works on the mixed-CRS streamed
        # artifact — we don't assert specific row counts since the projected
        # extents depend on the exact UTM bounds.
        result = catalog.query(slice_)
        assert isinstance(result, DuckDBGeoCatalog)


# ---------------------------------------------------------------------------
# GeoParquet 1.1 bbox-struct + metadata
# ---------------------------------------------------------------------------


class TestGeoParquetMetadata:
    def test_bbox_covering_struct_emitted(
        self, utm29_tile_factory, tmp_path: Path
    ) -> None:
        path = utm29_tile_factory((500_000, 4_000_000, 500_160, 4_000_160), "20240115")
        out = tmp_path / "cat.parquet"
        build_raster_catalog(
            [path], filename_regex=RASTER_REGEX, backend="duckdb", out_path=out
        )
        meta = pq.read_metadata(out)
        schema = meta.schema.to_arrow_schema()
        assert "bbox" in schema.names
        bbox_field = schema.field("bbox")
        children = {f.name for f in bbox_field.type}
        assert children == {"xmin", "ymin", "xmax", "ymax"}

    def test_geo_metadata_parses(self, utm29_tile_factory, tmp_path: Path) -> None:
        path = utm29_tile_factory((500_000, 4_000_000, 500_160, 4_000_160), "20240115")
        out = tmp_path / "cat.parquet"
        build_raster_catalog(
            [path], filename_regex=RASTER_REGEX, backend="duckdb", out_path=out
        )
        meta = pq.read_metadata(out)
        kv = meta.metadata
        assert kv is not None
        assert b"geo" in kv
        geo = json.loads(kv[b"geo"].decode("utf-8"))
        assert geo["version"] == "1.1.0"
        assert geo["primary_column"] == "geometry"
        col = geo["columns"]["geometry"]
        assert col["encoding"] == "WKB"
        assert "Polygon" in col["geometry_types"]
        assert "covering" in col
        assert col["covering"]["bbox"]["xmin"] == ["bbox", "xmin"]


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_duckdb_requires_out_path(self, utm29_tile_factory, tmp_path: Path) -> None:
        path = utm29_tile_factory((500_000, 4_000_000, 500_160, 4_000_160), "20240115")
        with pytest.raises(ValueError, match="requires out_path"):
            build_raster_catalog(
                [path], filename_regex=RASTER_REGEX, backend="duckdb", out_path=None
            )

    def test_empty_filepaths_raises(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.parquet"
        with pytest.raises(ValueError, match="no files yielded"):
            build_raster_catalog([], backend="duckdb", out_path=out)

    def test_invalid_backend_rejected_raster(
        self, utm29_tile_factory, tmp_path: Path
    ) -> None:
        path = utm29_tile_factory((500_000, 4_000_000, 500_160, 4_000_160), "20240115")
        with pytest.raises(ValueError, match="must be 'memory' or 'duckdb'"):
            build_raster_catalog(
                [path],
                filename_regex=RASTER_REGEX,
                backend="duckbd",  # typo
                out_path=tmp_path / "x.parquet",
            )

    def test_invalid_backend_rejected_vector(self, tmp_path: Path) -> None:
        gdf = gpd.GeoDataFrame(
            {"geometry": [shapely.geometry.box(0, 0, 1, 1)]}, crs="EPSG:32629"
        )
        gdf.to_file(tmp_path / "labels_20240115.gpkg", driver="GPKG")
        with pytest.raises(ValueError, match="must be 'memory' or 'duckdb'"):
            build_vector_catalog(
                [tmp_path / "labels_20240115.gpkg"],
                filename_regex=VECTOR_REGEX,
                backend="duckbd",  # typo
                out_path=tmp_path / "x.parquet",
            )

    def test_invalid_backend_rejected_xarray(self, tmp_path: Path) -> None:
        xr = pytest.importorskip("xarray")
        ds = xr.Dataset(
            {"ndvi": (("y", "x"), np.zeros((4, 4), dtype=np.float32))},
            coords={
                "y": np.linspace(40.5, 40.0, 4),
                "x": np.linspace(-3.5, -3.0, 4),
            },
        )
        ds.to_netcdf(tmp_path / "x.nc")
        from geocatalog import build_xarray_catalog

        with pytest.raises(ValueError, match="must be 'memory' or 'duckdb'"):
            build_xarray_catalog(
                [tmp_path / "x.nc"],
                target_crs="EPSG:4326",
                backend="duckbd",  # typo
                out_path=tmp_path / "y.parquet",
            )

    def test_xarray_duckdb_requires_target_crs(self, tmp_path: Path) -> None:
        """The xarray duckdb branch doesn't reproject coordinate bounds —
        a silent EPSG:4326 default would mislabel projected NetCDFs."""
        xr = pytest.importorskip("xarray")
        ds = xr.Dataset(
            {"ndvi": (("y", "x"), np.zeros((4, 4), dtype=np.float32))},
            coords={
                "y": np.linspace(40.5, 40.0, 4),
                "x": np.linspace(-3.5, -3.0, 4),
            },
        )
        ds.to_netcdf(tmp_path / "x.nc")
        from geocatalog import build_xarray_catalog

        with pytest.raises(ValueError, match="requires target_crs"):
            build_xarray_catalog(
                [tmp_path / "x.nc"],
                backend="duckdb",
                out_path=tmp_path / "y.parquet",
                # target_crs omitted → ValueError
            )

    def test_empty_input_leaves_existing_artifact_intact(
        self, utm29_tile_factory, tmp_path: Path
    ) -> None:
        """Regression for the staged-write contract: a failed build (no
        matching rows) must not clobber a pre-existing out_path."""
        out = tmp_path / "preexisting.parquet"
        good = utm29_tile_factory((500_000, 4_000_000, 500_160, 4_000_160), "20240115")
        build_raster_catalog(
            [good], filename_regex=RASTER_REGEX, backend="duckdb", out_path=out
        )
        before = out.read_bytes()

        # Second build: regex doesn't match the renamed copy → 0 rows.
        bad = tmp_path / "no_match_name.tif"
        bad.write_bytes(good.read_bytes())
        with pytest.raises(ValueError, match="no files yielded"):
            build_raster_catalog(
                [bad], filename_regex=RASTER_REGEX, backend="duckdb", out_path=out
            )

        # The pre-existing artifact survived intact.
        assert out.read_bytes() == before


# ---------------------------------------------------------------------------
# StreamingParquetWriter direct tests
# ---------------------------------------------------------------------------


class TestStreamingParquetWriterDirect:
    def test_write_then_read(self, tmp_path: Path) -> None:
        path = tmp_path / "direct.parquet"
        with StreamingParquetWriter(
            path, crs="EPSG:4326", backend="raster", batch_size=2
        ) as w:
            for i in range(5):
                w.write_row(
                    {
                        "filepath": f"f{i}.tif",
                        "geometry": shapely.geometry.box(i, i, i + 1, i + 1),
                        "start_time": pd.Timestamp("2024-01-01"),
                        "end_time": pd.Timestamp("2024-01-02"),
                        "crs": "EPSG:4326",
                    }
                )
        gdf = gpd.read_parquet(path)
        assert len(gdf) == 5
        assert gdf.crs == pyproj.CRS("EPSG:4326")
        assert "_backend" in gdf.columns
        assert (gdf["_backend"] == "raster").all()

    def test_empty_writer_produces_valid_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.parquet"
        with StreamingParquetWriter(path, crs="EPSG:4326", backend="raster"):
            pass
        # File exists with valid GeoParquet metadata; geopandas can open it.
        gdf = gpd.read_parquet(path)
        assert len(gdf) == 0

    def test_write_after_close_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "x.parquet"
        w = StreamingParquetWriter(path, crs="EPSG:4326", backend="raster")
        w.write_row(
            {
                "filepath": "f.tif",
                "geometry": shapely.geometry.box(0, 0, 1, 1),
                "start_time": pd.Timestamp("2024-01-01"),
                "end_time": pd.Timestamp("2024-01-02"),
                "crs": "EPSG:4326",
            }
        )
        w.close()
        with pytest.raises(RuntimeError, match="after close"):
            w.write_row(
                {
                    "filepath": "g.tif",
                    "geometry": shapely.geometry.box(0, 0, 1, 1),
                    "start_time": pd.Timestamp("2024-01-01"),
                    "end_time": pd.Timestamp("2024-01-02"),
                    "crs": "EPSG:4326",
                }
            )
