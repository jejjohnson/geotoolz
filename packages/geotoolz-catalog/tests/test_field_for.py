"""Tests for `geocatalog.staging.field_for` — the geopatcher bridge (#244).

The whole module is skipped when geopatcher isn't installed so a
plain install (without the `[patch]` extra) still passes CI. With
geopatcher present, small staged catalogs of real GeoTIFFs are turned
into `RasterField`s on a slice grid and round-tripped through a
`SpatialPatcher`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds, transform_geom
from shapely.geometry import box, mapping, shape

from tests.conftest import catalog_from_rows


geopatcher = pytest.importorskip("geopatcher")

from georeader.geotensor import GeoTensor

from geocatalog._src.geoslice import GeoSlice
from geocatalog._src.memory import InMemoryGeoCatalog
from geocatalog._src.staging import field_for, stage


LEFT = (500_000, 4_000_000, 500_320, 4_000_320)
RIGHT = (500_320, 4_000_000, 500_640, 4_000_320)
WINTER = (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-31"))


def _slice(
    bounds: tuple[float, float, float, float] = (*LEFT[:2], *RIGHT[2:]),
) -> GeoSlice:
    return GeoSlice(
        bounds=bounds,
        interval=pd.Interval(*WINTER, closed="both"),
        resolution=(10.0, 10.0),
        crs="EPSG:32629",
    )


def _row(bounds: tuple[float, ...], day: str, **extras: object) -> dict:
    return {
        "geometry": box(*bounds),
        "start_time": pd.Timestamp(day),
        "end_time": pd.Timestamp(day),
        **extras,
    }


@pytest.fixture
def asset_catalog(tmp_path: Path, utm29_tile_factory) -> InMemoryGeoCatalog:
    """Two side-by-side rows with `red` / `nir` assets, staged to local TIFs."""
    red0 = utm29_tile_factory(LEFT, "20240115", value=10)
    nir0 = utm29_tile_factory(LEFT, "20240116", value=20)
    red1 = utm29_tile_factory(RIGHT, "20240117", value=30)
    nir1 = utm29_tile_factory(RIGHT, "20240118", value=40)
    cat = catalog_from_rows(
        rows=[
            _row(
                LEFT,
                "2024-01-15",
                filepath=str(red0),
                assets=json.dumps({"red": str(red0), "nir": str(nir0)}),
            ),
            _row(
                RIGHT,
                "2024-01-17",
                filepath=str(red1),
                assets=json.dumps({"red": str(red1), "nir": str(nir1)}),
            ),
        ],
        crs="EPSG:32629",
    )
    return stage(cat, dest=tmp_path / "cache")


@pytest.fixture
def legacy_catalog(utm29_tile_factory) -> InMemoryGeoCatalog:
    """Single-row catalog with only `filepath` (the `build_raster_catalog` shape)."""
    path = utm29_tile_factory(LEFT, "20240115", value=7)
    return catalog_from_rows(
        rows=[_row(LEFT, "2024-01-15", filepath=str(path))], crs="EPSG:32629"
    )


@pytest.fixture
def cross_crs_catalog(tmp_path: Path, utm29_tile_factory) -> InMemoryGeoCatalog:
    """Left tile in UTM 29N; right tile written in UTM 30N, indexed in 29N."""
    left = utm29_tile_factory(LEFT, "20240115", n_bands=1, value=10)
    # The 30N file covers the right tile's footprint (plus a margin).
    xmin, ymin, xmax, ymax = transform_bounds("EPSG:32629", "EPSG:32630", *RIGHT)
    pad = 50.0
    bounds30 = (xmin - pad, ymin - pad, xmax + pad, ymax + pad)
    right = tmp_path / "right_utm30.tif"
    with rasterio.open(
        right,
        "w",
        driver="GTiff",
        height=40,
        width=40,
        count=1,
        dtype="uint16",
        crs="EPSG:32630",
        transform=from_bounds(*bounds30, 40, 40),
    ) as dst:
        dst.write(np.full((1, 40, 40), 30, dtype=np.uint16))
    footprint = shape(
        transform_geom("EPSG:32630", "EPSG:32629", mapping(box(*bounds30)))
    )
    return catalog_from_rows(
        rows=[
            _row(LEFT, "2024-01-15", filepath=str(left)),
            {**_row(RIGHT, "2024-01-17", filepath=str(right)), "geometry": footprint},
        ],
        crs="EPSG:32629",
    )


def _patcher() -> geopatcher.SpatialPatcher:
    return geopatcher.SpatialPatcher(
        geometry=geopatcher.SpatialRectangular(size=(8, 8), boundary="pad"),
        sampler=geopatcher.SpatialRegularStride(step=8),
        window=geopatcher.SpatialBoxcar(),
        aggregation=geopatcher.SpatialOverlapAdd(),
    )


# ---------------------------------------------------------------------------
# One field on the slice grid
# ---------------------------------------------------------------------------


class TestFieldOnTheSliceGrid:
    def test_mosaics_the_rows_onto_the_slice(
        self, asset_catalog: InMemoryGeoCatalog
    ) -> None:
        field = field_for(asset_catalog, _slice(), asset="red")
        assert isinstance(field, geopatcher.RasterField)
        tensor = field.domain
        assert isinstance(tensor, GeoTensor)
        assert str(tensor.crs).endswith("32629")
        assert tensor.shape == (3, 32, 64)
        assert int(tensor.values[0, 0, 0]) == 10  # red0
        assert int(tensor.values[0, 0, -1]) == 30  # red1

    def test_asset_selects_the_band_file(
        self, asset_catalog: InMemoryGeoCatalog
    ) -> None:
        tensor = field_for(asset_catalog, _slice(), asset="nir").domain
        assert (int(tensor.values[0, 0, 0]), int(tensor.values[0, 0, -1])) == (20, 40)

    def test_band_indexes_are_forwarded(
        self, asset_catalog: InMemoryGeoCatalog
    ) -> None:
        field = field_for(asset_catalog, _slice(), asset="red", band_indexes=[2])
        assert field.domain.shape == (1, 32, 64)

    def test_no_slice_covers_the_catalog_at_native_resolution(
        self, legacy_catalog: InMemoryGeoCatalog
    ) -> None:
        tensor = field_for(legacy_catalog).domain
        assert tensor.shape == (3, 32, 32)
        assert tuple(tensor.bounds) == LEFT
        assert int(tensor.values[0, 5, 5]) == 7

    def test_rows_outside_the_slice_need_not_carry_the_asset(
        self, asset_catalog: InMemoryGeoCatalog
    ) -> None:
        gdf = asset_catalog.gdf.copy()
        gdf.iloc[1, gdf.columns.get_loc("assets")] = json.dumps({"other": "x.tif"})
        cat = InMemoryGeoCatalog(gdf, kind="raster")
        # Clear of the shared edge, which `query` counts as intersecting.
        field = field_for(cat, _slice((*LEFT[:2], 500_300, LEFT[3])), asset="red")
        assert int(field.domain.values[0, 0, 0]) == 10


class TestCrossCrsRoundTrip:
    def test_field_is_in_the_catalog_crs(
        self, cross_crs_catalog: InMemoryGeoCatalog
    ) -> None:
        tensor = field_for(cross_crs_catalog, _slice()).domain
        assert str(tensor.crs).endswith("32629")
        assert tensor.shape == (1, 32, 64)
        assert int(tensor.values[0, 16, 8]) == 10
        assert int(tensor.values[0, 16, 56]) == 30  # warped from UTM 30N

    def test_split_then_merge_reassembles_the_field(
        self, cross_crs_catalog: InMemoryGeoCatalog
    ) -> None:
        field = field_for(cross_crs_catalog, _slice())
        patcher = _patcher()
        patches = list(patcher.split(field))
        assert len(patches) == (32 // 8) * (64 // 8)
        merged = np.asarray(patcher.merge(patches, field.domain))
        np.testing.assert_array_equal(merged, np.asarray(field.domain.values))


class TestLazyFields:
    def test_materialize_false_gives_one_lazy_field_per_row(
        self, cross_crs_catalog: InMemoryGeoCatalog
    ) -> None:
        fields = field_for(cross_crs_catalog, _slice(), materialize=False)
        assert len(fields) == 2
        # Each lazy field stays in its file's CRS; nothing is warped.
        crs = [str(f.domain.crs) for f in fields]
        assert crs[0].endswith("32629") and crs[1].endswith("32630")
        window = rasterio.windows.Window(0, 0, 4, 4)
        assert int(fields[1].select(window).values[0, 0, 0]) == 30


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TestFieldForErrors:
    def test_missing_asset_key_raises_keyerror(
        self, asset_catalog: InMemoryGeoCatalog
    ) -> None:
        with pytest.raises(KeyError, match="scl"):
            field_for(asset_catalog, _slice(), asset="scl")

    def test_asset_passed_positionally_is_a_type_error(
        self, asset_catalog: InMemoryGeoCatalog
    ) -> None:
        with pytest.raises(TypeError, match=r"asset='red'"):
            field_for(asset_catalog, "red")  # type: ignore[call-overload]

    def test_empty_catalog_raises_valueerror(
        self, asset_catalog: InMemoryGeoCatalog
    ) -> None:
        empty = asset_catalog.query(
            bounds=(0, 0, 1, 1),
            time=(pd.Timestamp("1900-01-01"), pd.Timestamp("1900-01-02")),
        )
        with pytest.raises(ValueError, match="empty"):
            field_for(empty, asset="red")

    def test_slice_selecting_nothing_raises(
        self, asset_catalog: InMemoryGeoCatalog
    ) -> None:
        with pytest.raises(ValueError, match="no catalog row"):
            field_for(asset_catalog, _slice((0, 0, 100, 100)), asset="red")

    def test_grid_cannot_be_inferred_from_a_foreign_crs_file(
        self, cross_crs_catalog: InMemoryGeoCatalog
    ) -> None:
        gdf = cross_crs_catalog.gdf.iloc[::-1]  # the UTM 30N file first
        with pytest.raises(ValueError, match="slice_="):
            field_for(InMemoryGeoCatalog(gdf, kind="raster"))

    def test_unsupported_mode_rejected(self, asset_catalog: InMemoryGeoCatalog) -> None:
        with pytest.raises(ValueError, match="raster"):
            field_for(asset_catalog, asset="red", mode="vector")

    def test_asset_named_but_no_assets_column_raises(
        self, legacy_catalog: InMemoryGeoCatalog
    ) -> None:
        with pytest.raises(KeyError, match="assets"):
            field_for(legacy_catalog, asset="red")

    def test_non_raster_kind_rejected(self, legacy_catalog: InMemoryGeoCatalog) -> None:
        vector_cat = InMemoryGeoCatalog(legacy_catalog.gdf, kind="vector")
        with pytest.raises(ValueError, match="kind='vector'"):
            field_for(vector_cat)

    def test_windows_drive_path_treated_as_local(self) -> None:
        from geocatalog._src.staging._field_for import _is_local_path

        assert _is_local_path("C:/data/tile.tif") is True
        assert _is_local_path("c:/data/tile.tif") is True
        assert _is_local_path("D:/some/long/path.tif") is True
        assert _is_local_path("https://example.com/tile.tif") is False
        assert _is_local_path("s3://bucket/tile.tif") is False
        assert _is_local_path("file:///tmp/tile.tif") is True
        assert _is_local_path("/tmp/tile.tif") is True

    def test_non_local_uri_in_asset_map_raises_keyerror(
        self, utm29_tile_factory
    ) -> None:
        # `stage(on_error="skip")` leaves the original URI in the asset map.
        good = utm29_tile_factory(LEFT, "20240115", value=10)
        cat = catalog_from_rows(
            rows=[
                _row(
                    LEFT,
                    "2024-01-15",
                    filepath=str(good),
                    assets=json.dumps(
                        {"red": str(good), "nir": "https://nope.example/never.tif"}
                    ),
                )
            ],
            crs="EPSG:32629",
        )
        with pytest.raises(KeyError, match=r"non-local URIs"):
            field_for(cat, _slice(LEFT), asset="nir")


class TestImportGuards:
    def test_missing_geopatcher_names_the_patch_extra(
        self, asset_catalog: InMemoryGeoCatalog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in list(sys.modules):
            if name == "geopatcher" or name.startswith("geopatcher."):
                monkeypatch.delitem(sys.modules, name, raising=False)
        monkeypatch.setitem(sys.modules, "geopatcher", None)
        with pytest.raises(ImportError, match=r"geotoolz-catalog\[patch\]"):
            field_for(asset_catalog, asset="red")

    def test_missing_georeader_names_georeader(
        self, asset_catalog: InMemoryGeoCatalog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "georeader.rasterio_reader", None)
        with pytest.raises(ImportError, match="georeader-spaceml") as info:
            field_for(asset_catalog, asset="red")
        assert "patch" not in str(info.value)


def test_dict_asset_maps_from_stage_are_readable(
    tmp_path: Path, utm29_tile_factory
) -> None:
    red = utm29_tile_factory(LEFT, "20240115", value=10)
    cat = catalog_from_rows(
        rows=[_row(LEFT, "2024-01-15", filepath=str(red), assets="placeholder")],
        crs="EPSG:32629",
    )
    cat.gdf["assets"] = [{"red": str(red)}]
    staged = stage(cat, dest=tmp_path / "cache")
    field = field_for(staged, _slice(LEFT), asset="red")
    assert int(field.domain.values[0, 0, 0]) == 10


# ---------------------------------------------------------------------------
# Review follow-ups (#366)
# ---------------------------------------------------------------------------


def test_lazy_fields_without_a_slice_need_no_grid(
    cross_crs_catalog: InMemoryGeoCatalog,
) -> None:
    gdf = cross_crs_catalog.gdf.iloc[::-1]  # the UTM 30N file first
    fields = field_for(InMemoryGeoCatalog(gdf, kind="raster"), materialize=False)
    assert len(fields) == 2


def test_lazy_fields_honour_band_indexes(
    asset_catalog: InMemoryGeoCatalog,
) -> None:
    import rasterio

    fields = field_for(asset_catalog, asset="red", band_indexes=[2], materialize=False)
    window = rasterio.windows.Window(0, 0, 4, 4)
    assert fields[0].select(window).shape == (1, 4, 4)


def test_inferred_grid_covers_a_partial_last_pixel(utm29_tile_factory) -> None:
    # A second 10 m tile shifted 4 m east: the union is 32.4 pixels wide.
    first = utm29_tile_factory(LEFT, "20240115", value=5)
    shifted = (500_004, 4_000_000, 500_324, 4_000_320)
    second = utm29_tile_factory(shifted, "20240116", value=9)
    cat = catalog_from_rows(
        rows=[
            _row(LEFT, "2024-01-15", filepath=str(first)),
            _row(shifted, "2024-01-16", filepath=str(second)),
        ],
        crs="EPSG:32629",
    )
    tensor = field_for(cat).domain
    xmin, _, xmax, _ = tensor.bounds
    assert (xmin, xmax) == (500_000, 500_330)  # whole pixels on the first grid
    assert tensor.shape[-1] == 33
    assert int(tensor.values[0, 0, -2]) == 9  # the later row wins the overlap
