"""Tests for `geocatalog.staging.field_for` — the geopatcher bridge.

The whole module is skipped when geopatcher isn't installed so a
plain `pip install geocatalog` (without the `[patch]` extra) still
passes CI. With geopatcher present, we build a small staged
catalog of real GeoTIFFs and verify the returned `RasterField`s
read the bytes back correctly through `select(window)`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from shapely.geometry import box

from tests.conftest import catalog_from_rows


geopatcher = pytest.importorskip("geopatcher")

from geocatalog._src.memory import InMemoryGeoCatalog
from geocatalog._src.staging import field_for, stage


@pytest.fixture
def asset_catalog(tmp_path: Path, utm29_tile_factory) -> InMemoryGeoCatalog:
    """A two-row catalog with a JSON `assets` map per row, staged to local TIFs."""
    red0 = utm29_tile_factory(
        (500_000, 4_000_000, 500_320, 4_000_320), "20240115", value=10
    )
    nir0 = utm29_tile_factory(
        (500_000, 4_000_000, 500_320, 4_000_320), "20240116", value=20
    )
    red1 = utm29_tile_factory(
        (500_320, 4_000_000, 500_640, 4_000_320), "20240117", value=30
    )
    nir1 = utm29_tile_factory(
        (500_320, 4_000_000, 500_640, 4_000_320), "20240118", value=40
    )
    cat = catalog_from_rows(
        rows=[
            {
                "geometry": box(500_000, 4_000_000, 500_320, 4_000_320),
                "start_time": pd.Timestamp("2024-01-15"),
                "end_time": pd.Timestamp("2024-01-15"),
                "filepath": str(red0),
                "assets": json.dumps({"red": str(red0), "nir": str(nir0)}),
            },
            {
                "geometry": box(500_320, 4_000_000, 500_640, 4_000_320),
                "start_time": pd.Timestamp("2024-01-17"),
                "end_time": pd.Timestamp("2024-01-17"),
                "filepath": str(red1),
                "assets": json.dumps({"red": str(red1), "nir": str(nir1)}),
            },
        ],
        crs="EPSG:32629",
    )
    return stage(cat, dest=tmp_path / "cache")


@pytest.fixture
def legacy_catalog(tmp_path: Path, utm29_tile_factory) -> InMemoryGeoCatalog:
    """Single-row catalog with only `filepath` (the `build_raster_catalog` shape)."""
    path = utm29_tile_factory(
        (500_000, 4_000_000, 500_320, 4_000_320), "20240115", value=7
    )
    return catalog_from_rows(
        rows=[
            {
                "geometry": box(500_000, 4_000_000, 500_320, 4_000_320),
                "start_time": pd.Timestamp("2024-01-15"),
                "end_time": pd.Timestamp("2024-01-15"),
                "filepath": str(path),
            }
        ],
        crs="EPSG:32629",
    )


class TestFieldForAsset:
    def test_returns_one_field_per_row(self, asset_catalog: InMemoryGeoCatalog) -> None:
        fields = field_for(asset_catalog, "red")
        assert len(fields) == len(asset_catalog)
        for f in fields:
            assert isinstance(f, geopatcher.Field)
            assert isinstance(f, geopatcher.RasterField)

    def test_domain_carries_crs_and_bounds(
        self, asset_catalog: InMemoryGeoCatalog
    ) -> None:
        # The Field's domain duck-types georeader.GeoDataBase, which
        # exposes crs/bounds/shape — what samplers consult.
        fields = field_for(asset_catalog, "red")
        domain = fields[0].domain
        assert str(domain.crs).endswith("32629")
        xmin, ymin, xmax, ymax = domain.bounds
        assert (xmin, ymin, xmax, ymax) == (500_000, 4_000_000, 500_320, 4_000_320)

    def test_select_reads_pixels(self, asset_catalog: InMemoryGeoCatalog) -> None:
        import rasterio

        fields = field_for(asset_catalog, "red")
        # Read the full window — should round-trip the seeded constant.
        window = rasterio.windows.Window(col_off=0, row_off=0, width=32, height=32)
        tensor = fields[0].select(window)
        # `utm29_tile_factory` writes 3 bands of the requested value.
        assert tensor.values.shape == (3, 32, 32)
        assert int(tensor.values[0, 0, 0]) == 10  # red0 was seeded with value=10

    def test_nir_asset_selects_other_path(
        self, asset_catalog: InMemoryGeoCatalog
    ) -> None:
        import rasterio

        fields = field_for(asset_catalog, "nir")
        window = rasterio.windows.Window(col_off=0, row_off=0, width=32, height=32)
        # nir0 was seeded with value=20.
        assert int(fields[0].select(window).values[0, 0, 0]) == 20


class TestFieldForFilepathFallback:
    def test_asset_none_uses_filepath(self, legacy_catalog: InMemoryGeoCatalog) -> None:
        import rasterio

        fields = field_for(legacy_catalog)
        assert len(fields) == 1
        window = rasterio.windows.Window(col_off=0, row_off=0, width=32, height=32)
        assert int(fields[0].select(window).values[0, 0, 0]) == 7


class TestFieldForErrors:
    def test_missing_asset_key_raises_keyerror(
        self, asset_catalog: InMemoryGeoCatalog
    ) -> None:
        with pytest.raises(KeyError, match="scl"):
            field_for(asset_catalog, "scl")

    def test_empty_catalog_raises_valueerror(
        self, asset_catalog: InMemoryGeoCatalog
    ) -> None:
        empty = asset_catalog.query(
            bounds=(0, 0, 1, 1),
            time=(pd.Timestamp("1900-01-01"), pd.Timestamp("1900-01-02")),
        )
        with pytest.raises(ValueError, match="empty"):
            field_for(empty, "red")

    def test_unsupported_mode_rejected(self, asset_catalog: InMemoryGeoCatalog) -> None:
        with pytest.raises(ValueError, match="raster"):
            field_for(asset_catalog, "red", mode="vector")

    def test_asset_named_but_no_assets_column_raises(
        self, legacy_catalog: InMemoryGeoCatalog
    ) -> None:
        # Legacy catalogs (build_raster_catalog) have no `assets` column;
        # asking for a named asset must fail loudly rather than silently
        # falling back.
        with pytest.raises(KeyError, match="assets"):
            field_for(legacy_catalog, "red")

    def test_non_raster_backend_rejected(
        self, legacy_catalog: InMemoryGeoCatalog
    ) -> None:
        # Catalogs whose `backend` doesn't match `mode` must fail
        # upfront — a vector-backed catalog would otherwise reach
        # RasterioReader and explode with a less helpful error.
        vector_cat = InMemoryGeoCatalog(legacy_catalog.gdf, backend="vector")
        with pytest.raises(ValueError, match="backend='vector'"):
            field_for(vector_cat)

    def test_windows_drive_path_treated_as_local(self) -> None:
        # `urlparse("C:/data/tile.tif").scheme` returns "c" on every
        # platform — without the drive-letter special-case, the
        # local-path classifier would reject valid Windows paths.
        from geocatalog._src.staging._field_for import _is_local_path

        assert _is_local_path("C:/data/tile.tif") is True
        assert _is_local_path("c:/data/tile.tif") is True
        assert _is_local_path("D:/some/long/path.tif") is True
        # Real remote schemes are still rejected.
        assert _is_local_path("https://example.com/tile.tif") is False
        assert _is_local_path("s3://bucket/tile.tif") is False
        assert _is_local_path("file:///tmp/tile.tif") is True
        assert _is_local_path("/tmp/tile.tif") is True

    def test_non_local_uri_in_asset_map_raises_keyerror(
        self, tmp_path: Path, utm29_tile_factory
    ) -> None:
        # `stage(on_error="skip")` leaves the original URI in the asset
        # map for failed rows. `field_for` must surface that as a clear
        # KeyError rather than silently building a RasterField pointing
        # at a remote object.
        good = utm29_tile_factory(
            (500_000, 4_000_000, 500_320, 4_000_320), "20240115", value=10
        )
        cat = catalog_from_rows(
            rows=[
                {
                    "geometry": box(500_000, 4_000_000, 500_320, 4_000_320),
                    "start_time": pd.Timestamp("2024-01-15"),
                    "end_time": pd.Timestamp("2024-01-15"),
                    "filepath": str(good),
                    "assets": json.dumps(
                        {"red": str(good), "nir": "https://nope.example/never.tif"}
                    ),
                }
            ],
            crs="EPSG:32629",
        )
        with pytest.raises(KeyError, match=r"non-local URIs"):
            field_for(cat, "nir")


class TestFieldForImportGuard:
    def test_missing_geopatcher_raises_clear_importerror(
        self, asset_catalog: InMemoryGeoCatalog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Simulate geopatcher absence by short-circuiting the import.
        # We have to hide both `geopatcher` itself and any cached
        # submodules so the top-level `from geopatcher import ...`
        # raises.
        for name in list(sys.modules):
            if name == "geopatcher" or name.startswith("geopatcher."):
                monkeypatch.delitem(sys.modules, name, raising=False)
        monkeypatch.setitem(sys.modules, "geopatcher", None)
        with pytest.raises(ImportError, match=r"geocatalog\[patch\]"):
            field_for(asset_catalog, "red")
