"""Strict-mode + warning behaviour for catalog metadata readers (gh #13, #26).

External GeoParquet artifacts (written by GDAL / DuckDB / geopandas
directly) lack the reserved ``_backend`` column. The default behaviour is
warn-and-fall-back to ``backend="raster"``; ``strict=True`` raises
`CatalogMetadataError`; an explicit ``backend=`` override bypasses the
check entirely.
"""

from __future__ import annotations

import io
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import shapely.geometry
from loguru import logger

from geocatalog import (
    CatalogMetadataError,
    from_geoparquet,
    open_catalog,
    to_geoparquet,
)
from geocatalog._src.memory import InMemoryGeoCatalog


@pytest.fixture
def loguru_sink():
    """In-memory loguru sink with the geocatalog namespace enabled."""
    buf = io.StringIO()
    handler_id = logger.add(buf, level="TRACE", format="{name} | {message}")
    logger.enable("geocatalog")
    yield buf
    logger.disable("geocatalog")
    logger.remove(handler_id)


def _external_parquet(tmp_path: Path) -> Path:
    """A GeoParquet written by geopandas directly — no ``_backend`` column."""
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [shapely.geometry.box(0, 0, 100, 100)],
            "start_time": [pd.Timestamp("2024-01-01")],
            "end_time": [pd.Timestamp("2024-01-02")],
            "filepath": ["A.tif"],
        },
        geometry="geometry",
        crs="EPSG:32629",
    )
    path = tmp_path / "external.parquet"
    gdf.to_parquet(path)
    return path


def _geocatalog_parquet(tmp_path: Path) -> Path:
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [shapely.geometry.box(0, 0, 100, 100)],
            "start_time": [pd.Timestamp("2024-01-01")],
            "end_time": [pd.Timestamp("2024-01-02")],
            "filepath": ["A.tif"],
        },
        geometry="geometry",
        crs="EPSG:32629",
    )
    path = tmp_path / "native.parquet"
    to_geoparquet(InMemoryGeoCatalog(gdf, backend="vector"), path)
    return path


class TestFromGeoparquet:
    def test_missing_backend_warns_and_defaults(
        self, tmp_path: Path, loguru_sink: io.StringIO
    ) -> None:
        cat = from_geoparquet(_external_parquet(tmp_path))
        assert cat.backend == "raster"
        assert "no _backend column" in loguru_sink.getvalue()

    def test_missing_backend_strict_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CatalogMetadataError, match="_backend"):
            from_geoparquet(_external_parquet(tmp_path), strict=True)

    def test_explicit_backend_bypasses_even_strict(
        self, tmp_path: Path, loguru_sink: io.StringIO
    ) -> None:
        cat = from_geoparquet(
            _external_parquet(tmp_path), backend="vector", strict=True
        )
        assert cat.backend == "vector"
        assert "no _backend column" not in loguru_sink.getvalue()

    def test_native_artifact_never_warns(
        self, tmp_path: Path, loguru_sink: io.StringIO
    ) -> None:
        cat = from_geoparquet(_geocatalog_parquet(tmp_path), strict=True)
        assert cat.backend == "vector"
        assert "no _backend column" not in loguru_sink.getvalue()


class TestOpenCatalogFactory:
    def test_memory_engine_strict_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CatalogMetadataError, match="_backend"):
            open_catalog(_external_parquet(tmp_path), engine="memory", strict=True)

    def test_memory_engine_override_honoured(self, tmp_path: Path) -> None:
        cat = open_catalog(
            _external_parquet(tmp_path),
            engine="memory",
            backend="vector",
            strict=True,
        )
        assert cat.backend == "vector"


class TestDuckDBOpen:
    def test_missing_backend_warns_and_defaults(
        self, tmp_path: Path, loguru_sink: io.StringIO
    ) -> None:
        pytest.importorskip("duckdb")
        from geocatalog import DuckDBGeoCatalog

        cat = DuckDBGeoCatalog.open(_external_parquet(tmp_path))
        assert cat.backend == "raster"
        assert "no _backend column" in loguru_sink.getvalue()

    def test_missing_backend_strict_raises(self, tmp_path: Path) -> None:
        pytest.importorskip("duckdb")
        from geocatalog import DuckDBGeoCatalog

        with pytest.raises(CatalogMetadataError, match="_backend"):
            DuckDBGeoCatalog.open(_external_parquet(tmp_path), strict=True)

    def test_explicit_backend_bypasses_even_strict(self, tmp_path: Path) -> None:
        pytest.importorskip("duckdb")
        from geocatalog import DuckDBGeoCatalog

        cat = DuckDBGeoCatalog.open(
            _external_parquet(tmp_path), backend="vector", strict=True
        )
        assert cat.backend == "vector"

    def test_native_artifact_no_warning(
        self, tmp_path: Path, loguru_sink: io.StringIO
    ) -> None:
        pytest.importorskip("duckdb")
        from geocatalog import DuckDBGeoCatalog

        cat = DuckDBGeoCatalog.open(_geocatalog_parquet(tmp_path), strict=True)
        assert cat.backend == "vector"
        assert "no _backend column" not in loguru_sink.getvalue()


class TestCrsMetadataReader:
    def test_corrupt_file_warns_and_defaults(
        self, tmp_path: Path, loguru_sink: io.StringIO
    ) -> None:
        pytest.importorskip("duckdb")
        from geocatalog._src.duckdb_backend import _read_geoparquet_crs

        bad = tmp_path / "corrupt.parquet"
        bad.write_bytes(b"this is not parquet")
        assert _read_geoparquet_crs(bad, default="EPSG:4326") == "EPSG:4326"
        assert "could not read Parquet metadata" in loguru_sink.getvalue()

    def test_corrupt_file_strict_raises(self, tmp_path: Path) -> None:
        pytest.importorskip("duckdb")
        from geocatalog._src.duckdb_backend import _read_geoparquet_crs

        bad = tmp_path / "corrupt.parquet"
        bad.write_bytes(b"this is not parquet")
        with pytest.raises(CatalogMetadataError, match="Parquet metadata"):
            _read_geoparquet_crs(bad, default="EPSG:4326", strict=True)


def _tagged_parquet(tmp_path: Path, tag: object) -> Path:
    """A parquet whose ``_backend`` column carries an arbitrary tag value."""
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [shapely.geometry.box(0, 0, 100, 100)],
            "start_time": [pd.Timestamp("2024-01-01")],
            "end_time": [pd.Timestamp("2024-01-02")],
            "filepath": ["A.tif"],
            "_backend": [tag],
        },
        geometry="geometry",
        crs="EPSG:32629",
    )
    path = tmp_path / "tagged.parquet"
    gdf.to_parquet(path)
    return path


class TestCorruptBackendTag:
    def test_unrecognised_tag_warns_and_defaults(
        self, tmp_path: Path, loguru_sink: io.StringIO
    ) -> None:
        pytest.importorskip("duckdb")
        from geocatalog import DuckDBGeoCatalog

        cat = DuckDBGeoCatalog.open(_tagged_parquet(tmp_path, "vecotr"))
        assert cat.backend == "raster"
        assert "unrecognised _backend tag 'vecotr'" in loguru_sink.getvalue()

    def test_unrecognised_tag_strict_raises(self, tmp_path: Path) -> None:
        pytest.importorskip("duckdb")
        from geocatalog import DuckDBGeoCatalog

        with pytest.raises(CatalogMetadataError, match="unrecognised _backend tag"):
            DuckDBGeoCatalog.open(_tagged_parquet(tmp_path, "vecotr"), strict=True)

    def test_null_tag_strict_raises(self, tmp_path: Path) -> None:
        pytest.importorskip("duckdb")
        from geocatalog import DuckDBGeoCatalog

        with pytest.raises(CatalogMetadataError, match="no readable value"):
            DuckDBGeoCatalog.open(_tagged_parquet(tmp_path, None), strict=True)

    def test_bad_tag_with_explicit_backend_bypasses(self, tmp_path: Path) -> None:
        pytest.importorskip("duckdb")
        from geocatalog import DuckDBGeoCatalog

        cat = DuckDBGeoCatalog.open(
            _tagged_parquet(tmp_path, "vecotr"), backend="vector", strict=True
        )
        assert cat.backend == "vector"


class TestMalformedGeoMetadataShapes:
    @staticmethod
    def _parquet_with_geo(tmp_path: Path, geo_bytes: bytes) -> Path:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table({"x": [1, 2]})
        table = table.replace_schema_metadata({b"geo": geo_bytes})
        path = tmp_path / "weird_geo.parquet"
        pq.write_table(table, path)
        return path

    def test_non_mapping_geo_json_warns_and_defaults(
        self, tmp_path: Path, loguru_sink: io.StringIO
    ) -> None:
        pytest.importorskip("duckdb")
        from geocatalog._src.duckdb_backend import _read_geoparquet_crs

        path = self._parquet_with_geo(tmp_path, b"[1, 2, 3]")
        assert _read_geoparquet_crs(path, default="EPSG:4326") == "EPSG:4326"
        assert "malformed GeoParquet 'geo' metadata" in loguru_sink.getvalue()

    def test_non_mapping_columns_entry_strict_raises(self, tmp_path: Path) -> None:
        pytest.importorskip("duckdb")
        from geocatalog._src.duckdb_backend import _read_geoparquet_crs

        path = self._parquet_with_geo(
            tmp_path, b'{"primary_column": "geometry", "columns": ["geometry"]}'
        )
        with pytest.raises(CatalogMetadataError, match="malformed GeoParquet"):
            _read_geoparquet_crs(path, default="EPSG:4326", strict=True)

    def test_invalid_utf8_geo_warns_and_defaults(
        self, tmp_path: Path, loguru_sink: io.StringIO
    ) -> None:
        pytest.importorskip("duckdb")
        from geocatalog._src.duckdb_backend import _read_geoparquet_crs

        path = self._parquet_with_geo(tmp_path, b"\xff\xfe{not json}")
        assert _read_geoparquet_crs(path, default="EPSG:4326") == "EPSG:4326"
        assert "malformed GeoParquet 'geo' metadata" in loguru_sink.getvalue()
