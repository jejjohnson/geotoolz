"""Tests for ``xrtoolz.geo.vectorize`` / ``transform_polygon`` (georeader oracle)."""

from __future__ import annotations

import sys

import numpy as np
import pytest
import xarray as xr
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

from xrtoolz.geo import Vectorize, transform_polygon, vectorize

from ._georeader_oracle import CRS, as_dataarray, as_geotensor


georeader_vectorize = pytest.importorskip("georeader.vectorize")
window_utils = pytest.importorskip("georeader.window_utils")


def _blobs() -> np.ndarray:
    """Two 4-connected blobs of different labels plus a single-pixel speck."""
    m = np.zeros((12, 12), dtype=np.uint8)
    m[1:5, 1:6] = 1
    m[3:5, 6:8] = 1  # L-shaped extension
    m[7:11, 4:10] = 2
    m[9, 1] = 1  # 1-pixel speck, area 100 m²
    return m


def _symdiff_area(a, b) -> float:
    return a.symmetric_difference(b).area


class TestVectorize:
    def test_matches_georeader_get_polygons(self):
        mask = _blobs()
        gdf = vectorize(as_dataarray(mask))
        # georeader with filtering and simplification disabled is the raw
        # rasterio.features.shapes output — the same algorithm we wrap.
        ref = georeader_vectorize.get_polygons(
            as_geotensor(mask), min_area=0, tolerance=0
        )
        assert len(gdf) == len(ref) == 3
        assert _symdiff_area(unary_union(list(gdf.geometry)), unary_union(ref)) == 0

    def test_labels_crs_and_dtype(self):
        gdf = vectorize(as_dataarray(_blobs()), attr_name="label")
        assert gdf.crs.to_epsg() == 32630
        assert gdf["label"].dtype == np.uint8
        assert sorted(gdf["label"].tolist()) == [1, 1, 2]
        # Label 2 is a 4 × 6 pixel rectangle of 10 m pixels.
        (area2,) = gdf.loc[gdf["label"] == 2].area
        assert area2 == pytest.approx(24 * 100.0)

    def test_min_area_is_in_crs_units(self):
        gdf = vectorize(as_dataarray(_blobs()), min_area=150.0)  # > 1 pixel
        assert len(gdf) == 2
        assert gdf.area.min() > 150.0

    def test_connectivity_8_merges_diagonals(self):
        m = np.zeros((4, 4), dtype=bool)
        m[0, 0] = m[1, 1] = True
        assert len(vectorize(as_dataarray(m), connectivity=4)) == 2
        assert len(vectorize(as_dataarray(m), connectivity=8)) == 1

    def test_bool_float_and_nan(self):
        m = np.array([[np.nan, 1.5], [0.0, 1.5]])
        gdf = vectorize(as_dataarray(m))
        assert gdf["value"].tolist() == [1.5]

    def test_empty_mask(self):
        gdf = vectorize(as_dataarray(np.zeros((3, 3), dtype=np.int32)))
        assert len(gdf) == 0
        assert gdf.crs.to_epsg() == 32630

    def test_single_variable_dataset_and_size1_dims(self):
        da = as_dataarray(_blobs()[None])  # (d0=1, y, x)
        assert len(vectorize(da)) == 3
        ds = da.to_dataset(name="mask")
        assert len(vectorize(ds)) == 3

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError, match="2-D"):
            vectorize(as_dataarray(np.ones((2, 3, 3), dtype=np.uint8)))
        with pytest.raises(ValueError, match="connectivity"):
            vectorize(as_dataarray(_blobs()), connectivity=6)
        ds = xr.Dataset({"a": as_dataarray(_blobs()), "b": as_dataarray(_blobs())})
        with pytest.raises(TypeError, match="single-variable"):
            vectorize(ds)

    def test_missing_extra_points_at_vector_extra(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "geopandas", None)
        with pytest.raises(ImportError, match=r"xrtoolz\[vector\]"):
            vectorize(as_dataarray(_blobs()))


class TestTransformPolygon:
    def test_matches_georeader_polygon_to_crs(self):
        poly = box(-3.71, 40.41, -3.69, 40.42)
        ours = transform_polygon(poly, "EPSG:4326", CRS)
        ref = window_utils.polygon_to_crs(poly, "EPSG:4326", CRS)
        assert ours.equals_exact(ref, tolerance=1e-6)

    def test_round_trip(self):
        poly = Polygon([(-3.7, 40.4), (-3.6, 40.4), (-3.65, 40.5)])
        there = transform_polygon(poly, "EPSG:4326", "EPSG:3857")
        assert there.bounds[0] < -400_000  # metres, not degrees
        back = transform_polygon(there, "EPSG:3857", "EPSG:4326")
        assert back.equals_exact(poly, tolerance=1e-9)

    def test_same_crs_is_identity_and_multipolygon_kept(self):
        multi = MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)])
        assert transform_polygon(multi, "EPSG:4326", "epsg:4326") is multi
        out = transform_polygon(multi, "EPSG:4326", CRS)
        assert isinstance(out, MultiPolygon)
        assert len(out.geoms) == 2


class TestVectorizeOperator:
    def test_eager_and_config(self):
        op = Vectorize(min_area=150.0, connectivity=8, attr_name="label")
        gdf = op(as_dataarray(_blobs()))
        assert len(gdf) == 2
        assert op.get_config() == {
            "min_area": 150.0,
            "connectivity": 8,
            "attr_name": "label",
        }

    def test_datatree_is_terminal(self):
        leaf = as_dataarray(_blobs()).to_dataset(name="mask")
        tree = xr.DataTree.from_dict({"a": leaf, "b": leaf})
        with pytest.raises(TypeError, match="DataTree"):
            Vectorize()(tree)
