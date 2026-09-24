"""Tests for ``xrtoolz.geo.footprint`` / ``valid_footprint`` (georeader oracle)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from affine import Affine
from shapely.geometry import MultiPolygon, Polygon, box

from xrtoolz.geo import Footprint, ValidFootprint, footprint, valid_footprint

from ._georeader_oracle import CRS, as_dataarray, as_geotensor


NODATA = -9999.0


def _bordered(shape=(30, 40), border=3) -> np.ndarray:
    data = np.full(shape, NODATA)
    data[border:-border, border:-border] = 1.0
    return data


class TestFootprint:
    def test_regular_grid_equals_bounds_polygon(self):
        da = as_dataarray(np.ones((30, 40)))
        fp = footprint(da)
        assert isinstance(fp, Polygon)
        assert fp.equals(box(*da.rio.bounds()))
        assert fp.bounds == (440_000.0, 4_474_700.0, 440_400.0, 4_475_000.0)

    @pytest.mark.parametrize("crs", [None, "EPSG:4326", "EPSG:3857"])
    def test_matches_georeader(self, crs):
        values = np.ones((30, 40))
        ours = footprint(as_dataarray(values), crs=crs)
        ref = as_geotensor(values).footprint(crs=crs)
        assert ours.equals_exact(ref, tolerance=1e-9)

    def test_rotated_transform_matches_georeader(self):
        # rioxarray cannot build 1-D coords for a rotated grid, so go via
        # write_transform on a coordinate-free array.
        transform = Affine.rotation(20.0) * Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)
        transform = Affine.translation(440_000.0, 4_475_000.0) * transform
        values = np.ones((30, 40))
        da = (
            xr.DataArray(values, dims=("y", "x"))
            .rio.write_crs(CRS)
            .rio.write_transform(transform)
        )
        ref = as_geotensor(values, transform=transform).footprint()
        assert footprint(da).equals_exact(ref, tolerance=1e-6)

    def test_dataset_and_round_trip(self):
        ds = as_dataarray(np.ones((30, 40))).to_dataset(name="v")
        ll = footprint(ds, crs="EPSG:4326")
        assert -4.0 < ll.bounds[0] < -3.0  # near Madrid, in degrees
        from xrtoolz.geo import transform_polygon

        assert transform_polygon(ll, "EPSG:4326", CRS).equals_exact(
            footprint(ds), tolerance=1e-6
        )

    def test_reproject_without_crs_raises(self):
        da = as_dataarray(np.ones((3, 3)), crs=None)
        assert footprint(da).area == pytest.approx(900.0)
        with pytest.raises(ValueError, match="no CRS"):
            footprint(da, crs="EPSG:4326")


class TestValidFootprint:
    def test_excludes_nodata_border(self):
        da = as_dataarray(_bordered(), nodata=NODATA)
        vfp = valid_footprint(da)
        full = footprint(da)
        assert full.contains(vfp)
        # 3-pixel (30 m) border trimmed from every side.
        assert vfp.equals(box(440_030.0, 4_474_730.0, 440_370.0, 4_474_970.0))

    @pytest.mark.parametrize("crs", [None, "EPSG:4326"])
    def test_matches_georeader(self, crs):
        data = _bordered()
        data[10:20, 20:30] = NODATA  # a hole
        ours = valid_footprint(as_dataarray(data, nodata=NODATA), crs=crs)
        ref = as_geotensor(data, nodata=NODATA).valid_footprint(crs=crs)
        # georeader simplifies at 1 px; on axis-aligned edges that only
        # drops collinear vertices, so the geometries coincide.
        assert ours.symmetric_difference(ref).area == pytest.approx(0.0, abs=1e-12)
        assert len(ours.interiors) == 1

    @pytest.mark.parametrize("method", ["all", "any"])
    def test_multiband_method_matches_georeader(self, method):
        bands = np.stack([_bordered(border=3), _bordered(border=6)])
        ours = valid_footprint(as_dataarray(bands, nodata=NODATA), method=method)
        ref = as_geotensor(bands, nodata=NODATA).valid_footprint(method=method)
        assert ours.equals(ref)
        border = 6 if method == "all" else 3
        assert ours.equals(
            footprint(as_dataarray(_bordered(border=border))).buffer(
                -10.0 * border, join_style="mitre"
            )
        )

    def test_nan_and_explicit_nodata_override(self):
        data = _bordered()
        data[data == NODATA] = np.nan
        data[0:5, :] = 0.0  # zeros are only invalid when nodata=0
        nan_only = valid_footprint(as_dataarray(data))
        with_zero = valid_footprint(as_dataarray(data), nodata=0.0)
        assert with_zero.area < nan_only.area

    def test_disjoint_regions_give_multipolygon(self):
        data = np.full((10, 10), NODATA)
        data[1:3, 1:3] = 1.0
        data[6:9, 6:9] = 1.0
        vfp = valid_footprint(as_dataarray(data, nodata=NODATA))
        assert isinstance(vfp, MultiPolygon)
        assert len(vfp.geoms) == 2

    def test_dataset_combines_variables(self):
        ds = xr.Dataset(
            {
                "a": as_dataarray(_bordered(border=2), nodata=NODATA),
                "b": as_dataarray(_bordered(border=5), nodata=NODATA),
                "scalar": 1.0,  # no spatial dims — ignored
            }
        )
        assert valid_footprint(ds, method="all").equals(valid_footprint(ds["b"]))
        assert valid_footprint(ds, method="any").equals(valid_footprint(ds["a"]))

    def test_errors(self):
        with pytest.raises(ValueError, match="no valid pixels"):
            valid_footprint(as_dataarray(np.full((3, 3), np.nan)))
        with pytest.raises(ValueError, match="method"):
            valid_footprint(as_dataarray(np.ones((3, 3))), method="most")
        with pytest.raises(ValueError, match="spatial dims"):
            valid_footprint(
                xr.Dataset({"s": 1.0}, coords={"x": [0.5, 1.5], "y": [1.5, 0.5]})
            )


class TestFootprintOperators:
    def test_eager_and_config(self):
        da = as_dataarray(_bordered(), nodata=NODATA)
        assert Footprint()(da).equals(footprint(da))
        op = ValidFootprint(crs="EPSG:4326", method="any")
        assert op(da).equals(valid_footprint(da, crs="EPSG:4326"))
        assert Footprint(crs="EPSG:4326").get_config() == {"crs": "EPSG:4326"}
        assert op.get_config() == {
            "crs": "EPSG:4326",
            "nodata": None,
            "method": "any",
            "connectivity": 4,
        }

    @pytest.mark.parametrize("op", [Footprint(), ValidFootprint()])
    def test_datatree_is_terminal(self, op):
        leaf = as_dataarray(_bordered(), nodata=NODATA).to_dataset(name="v")
        tree = xr.DataTree.from_dict({"a": leaf, "b": leaf})
        with pytest.raises(TypeError, match="DataTree"):
            op(tree)
