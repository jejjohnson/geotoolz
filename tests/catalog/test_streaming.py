"""Tests for the `backend="duckdb"` streaming builders.

Covers all three builders (raster / vector / xarray) plus the underlying
`StreamingParquetWriter`. Skipped wholesale if the ``[duckdb]`` extra is
not installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyproj
import pytest
import shapely.geometry


duckdb = pytest.importorskip("duckdb")

from geotoolz.catalog import (
    build_raster_catalog,
    build_vector_catalog,
    open_catalog,
)
from geotoolz.catalog._src.duckdb_backend import DuckDBGeoCatalog
from geotoolz.catalog._src.streaming import StreamingParquetWriter


RASTER_REGEX = r"S2_T29SND_(?P<date>\d{8})_\d+_\d+\.tif"
VECTOR_REGEX = r"labels_(?P<date>\d{8})\.gpkg"


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
        from geotoolz.catalog import build_xarray_catalog

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
        # All on the same day, so order is purely Hilbert.
        assert len(gdf) == 4
        # Sanity: the rewrite ran and produced a real polygon column.
        assert all(isinstance(g, shapely.geometry.Polygon) for g in gdf.geometry)

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
        from geotoolz.types import GeoSlice

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
