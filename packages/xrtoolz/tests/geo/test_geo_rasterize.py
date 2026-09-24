"""Tests for ``xrtoolz.geo.rasterize`` / ``rasterize_like`` (georeader oracle)."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import rasterio.windows
import xarray as xr
from shapely.geometry import LineString, Polygon, box

from xrtoolz.geo import (
    Rasterize,
    footprint,
    rasterize,
    rasterize_like,
    transform_polygon,
    vectorize,
)

from ._georeader_oracle import (
    CRS,
    TRANSFORM,
    as_dataarray,
    as_geotensor,
)


georeader_rasterize = pytest.importorskip("georeader.rasterize")

SHAPE = (40, 50)
# A skewed quadrilateral well inside the 500 m × 400 m grid.
POLY_UTM = Polygon(
    [(440_052.0, 4_474_930.0), (440_390.0, 4_474_880.0), (440_310.0, 4_474_700.0),
     (440_090.0, 4_474_660.0)]
)  # fmt: skip


@pytest.fixture
def like() -> xr.DataArray:
    return as_dataarray(np.zeros(SHAPE, dtype=np.float32))


@pytest.fixture
def like_gt():
    return as_geotensor(np.zeros(SHAPE, dtype=np.float32))


class TestRasterizeLike:
    @pytest.mark.parametrize("all_touched", [False, True])
    def test_matches_georeader_geometry_like(self, like, like_gt, all_touched):
        out = rasterize_like(POLY_UTM, like, all_touched=all_touched, dtype="uint8")
        ref = georeader_rasterize.rasterize_geometry_like(
            POLY_UTM, like_gt, all_touched=all_touched, return_only_data=True
        )
        np.testing.assert_array_equal(out.values, ref)
        assert out.values.sum() > 0

    def test_reprojects_geometry_like_georeader(self, like, like_gt):
        # Geometry in lon/lat onto a UTM grid — the CRS path.
        poly_ll = transform_polygon(POLY_UTM, CRS, "EPSG:4326")
        out = rasterize_like(poly_ll, like, geometries_crs="EPSG:4326", dtype="uint8")
        ref = georeader_rasterize.rasterize_geometry_like(
            poly_ll, like_gt, crs_geometry="EPSG:4326", return_only_data=True
        )
        np.testing.assert_array_equal(out.values, ref)
        assert out.rio.crs.to_epsg() == 32630

    def test_geodataframe_column_matches_georeader(self, like, like_gt):
        gdf = gpd.GeoDataFrame(
            {"cls": np.array([3, 7], dtype=np.int32)},
            geometry=[POLY_UTM, box(440_400.0, 4_474_700.0, 440_480.0, 4_474_900.0)],
            crs=CRS,
        ).to_crs("EPSG:4326")  # rasterize_like must bring it back to UTM
        out = rasterize_like(gdf, like, column="cls", dtype="int32")
        ref = georeader_rasterize.rasterize_geopandas_like(
            gdf, like_gt, column="cls", return_only_data=True
        )
        np.testing.assert_array_equal(out.values, ref)
        assert set(np.unique(out.values)) == {0, 3, 7}
        assert out.name == "cls"

    def test_output_is_on_like_grid(self, like):
        out = rasterize_like([POLY_UTM], like, fill=-1.0, value=5.0)
        assert out.dims == ("y", "x")
        assert out.dtype == np.float32
        xr.testing.assert_equal(out["x"], like["x"])
        xr.testing.assert_equal(out["y"], like["y"])
        assert out.rio.crs == like.rio.crs
        assert out.rio.transform() == TRANSFORM
        assert set(np.unique(out.values)) == {-1.0, 5.0}

    def test_line_and_empty_inputs(self, like):
        line = LineString([(440_005.0, 4_474_995.0), (440_495.0, 4_474_605.0)])
        assert rasterize_like(line, like).values.sum() > 0
        empty = rasterize_like([], like, fill=2.0)
        assert (empty.values == 2.0).all()

    def test_column_requires_geodataframe(self, like):
        with pytest.raises(ValueError, match="GeoDataFrame"):
            rasterize_like(POLY_UTM, like, column="cls")


class TestRasterize:
    def test_matches_georeader_from_geometry(self):
        out = rasterize(
            POLY_UTM, transform=TRANSFORM, out_shape=SHAPE, crs=CRS, dtype="uint8"
        )
        ref = georeader_rasterize.rasterize_from_geometry(
            POLY_UTM,
            transform=TRANSFORM,
            window_out=rasterio.windows.Window(0, 0, SHAPE[1], SHAPE[0]),
            return_only_data=True,
        )
        np.testing.assert_array_equal(out.values, ref)

    def test_grid_matches_equivalent_like(self, like):
        out = rasterize(POLY_UTM, transform=TRANSFORM, out_shape=SHAPE, crs=CRS)
        np.testing.assert_allclose(out["x"], like["x"])
        np.testing.assert_allclose(out["y"], like["y"])
        assert out.rio.crs.to_epsg() == 32630
        assert out.rio.transform() == TRANSFORM
        xr.testing.assert_equal(out, rasterize_like(POLY_UTM, like))


class TestRoundTrip:
    def test_rasterize_then_vectorize_recovers_polygon(self, like):
        mask = rasterize_like(POLY_UTM, like, dtype="uint8")
        gdf = vectorize(mask)
        assert len(gdf) == 1
        recovered = gdf.geometry.iloc[0]
        pixel = abs(TRANSFORM.a)
        # Staircase edges sit within one pixel diagonal of the true edge …
        assert recovered.hausdorff_distance(POLY_UTM) <= pixel * np.sqrt(2)
        # … and the area error is bounded by a half-pixel band on the rim.
        assert recovered.symmetric_difference(POLY_UTM).area <= (
            POLY_UTM.length * pixel / 2
        )

    def test_vectorize_then_rasterize_is_exact(self, like):
        mask = rasterize_like(POLY_UTM, like, dtype="uint8")
        again = rasterize_like(vectorize(mask), like, column="value", dtype="uint8")
        np.testing.assert_array_equal(again.values, mask.values)


class TestRasterizeOperator:
    def test_eager_and_config(self, like):
        op = Rasterize(POLY_UTM, value=2.0, dtype="uint8")
        out = op(like)
        assert out.values.max() == 2
        cfg = op.get_config()
        assert cfg["geometries"] == "<Polygon>"
        assert cfg["dtype"] == "uint8"
        assert Rasterize.forbid_in_yaml

    def test_datatree_two_leaves(self, like):
        # Second leaf is a coarser grid over the same area.
        coarse = like.coarsen(x=2, y=2).mean().rio.write_crs(CRS)
        tree = xr.DataTree.from_dict(
            {"fine": like.to_dataset(name="v"), "coarse": coarse.to_dataset(name="v")}
        )
        out = Rasterize(POLY_UTM, dtype="uint8")(tree)
        assert sorted(out.children) == ["coarse", "fine"]
        fine = out["fine"].to_dataset()["rasterized"]
        np.testing.assert_array_equal(
            fine.values, rasterize_like(POLY_UTM, like, dtype="uint8").values
        )
        assert out["coarse"].to_dataset()["rasterized"].shape == (20, 25)
        # Both leaves rasterized over the same footprint.
        assert footprint(coarse).equals(footprint(like))
