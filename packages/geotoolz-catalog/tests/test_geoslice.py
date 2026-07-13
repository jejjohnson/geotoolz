"""Tests for `geocatalog.GeoSlice`."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pyproj
import pytest

from geocatalog import GeoSlice, slice_to_window, window_to_slice


@pytest.fixture
def slice_4326() -> GeoSlice:
    return GeoSlice(
        bounds=(-10.0, 40.0, -8.0, 42.0),
        interval=pd.Interval(
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-31"),
            closed="both",
        ),
        resolution=(0.01, 0.01),
        crs="EPSG:4326",
    )


class TestGeoSliceInvariants:
    def test_bounds_must_be_ordered(self) -> None:
        with pytest.raises(ValueError, match="xmin < xmax"):
            GeoSlice(
                bounds=(1.0, 0.0, 0.0, 1.0),  # xmin > xmax
                interval=pd.Interval(0, 1, closed="both"),
                resolution=(0.1, 0.1),
                crs="EPSG:4326",
            )

    def test_interval_must_be_closed_both(self) -> None:
        with pytest.raises(ValueError, match="closed='both'"):
            GeoSlice(
                bounds=(0.0, 0.0, 1.0, 1.0),
                interval=pd.Interval(0, 1, closed="left"),
                resolution=(0.1, 0.1),
                crs="EPSG:4326",
            )

    def test_interval_must_be_pd_interval(self) -> None:
        with pytest.raises(TypeError, match=r"pd\.Interval"):
            GeoSlice(
                bounds=(0.0, 0.0, 1.0, 1.0),
                interval=(0, 1),  # type: ignore[arg-type]
                resolution=(0.1, 0.1),
                crs="EPSG:4326",
            )

    def test_resolution_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            GeoSlice(
                bounds=(0.0, 0.0, 1.0, 1.0),
                interval=pd.Interval(0, 1, closed="both"),
                resolution=(0.1, -0.1),
                crs="EPSG:4326",
            )

    def test_crs_string_coerced_to_pyproj(self, slice_4326: GeoSlice) -> None:
        assert isinstance(slice_4326.crs, pyproj.CRS)
        assert slice_4326.crs == pyproj.CRS.from_epsg(4326)


class TestGeoSliceFrozen:
    def test_frozen(self, slice_4326: GeoSlice) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            slice_4326.bounds = (0.0, 0.0, 1.0, 1.0)  # type: ignore[misc]

    def test_hashable(self, slice_4326: GeoSlice) -> None:
        # Frozen + hashable = can be used as a dict key.
        d = {slice_4326: "value"}
        assert d[slice_4326] == "value"

    def test_replace(self, slice_4326: GeoSlice) -> None:
        replaced = dataclasses.replace(slice_4326, resolution=(0.005, 0.005))
        assert replaced.resolution == (0.005, 0.005)
        assert replaced.bounds == slice_4326.bounds


class TestGeoSliceDerived:
    def test_shape(self, slice_4326: GeoSlice) -> None:
        assert slice_4326.shape == (200, 200)
        assert slice_4326.height == 200
        assert slice_4326.width == 200

    def test_transform_north_up(self, slice_4326: GeoSlice) -> None:
        t = slice_4326.transform
        # Affine(a, b, c, d, e, f) = (x_res, 0, x_origin, 0, -y_res, y_origin)
        assert t.a == pytest.approx(0.01)
        assert t.e == pytest.approx(-0.01)
        assert t.c == pytest.approx(-10.0)
        assert t.f == pytest.approx(42.0)


class TestGeoSliceToCRS:
    def test_identity(self, slice_4326: GeoSlice) -> None:
        out = slice_4326.to_crs("EPSG:4326")
        assert out is slice_4326

    def test_reproject_preserves_shape_roughly(self, slice_4326: GeoSlice) -> None:
        # 4326 bbox in Iberia → UTM zone 29N.
        out = slice_4326.to_crs("EPSG:32629")
        assert out.crs == pyproj.CRS.from_epsg(32629)
        # Shape should be within ±1 pixel of original (resolution rescaled).
        np.testing.assert_allclose(out.shape, slice_4326.shape, atol=2)


class TestSliceWindowRoundtrip:
    def test_roundtrip(self, slice_4326: GeoSlice) -> None:
        w = slice_to_window(slice_4326, slice_4326.transform)
        recovered = window_to_slice(
            w,
            slice_4326.transform,
            slice_4326.crs,
            slice_4326.interval,
            slice_4326.resolution,
        )
        np.testing.assert_allclose(recovered.bounds, slice_4326.bounds, atol=1e-9)


class TestToCrsDegenerateGuards:
    """`to_crs` refuses pathological reprojections loudly (gh #16)."""

    def test_out_of_domain_reprojection_raises(self) -> None:
        # An AOI on the invisible hemisphere of a geostationary grid
        # (GOES / Meteosat style) reprojects to (inf, inf, inf, inf).
        sl = GeoSlice(
            bounds=(150.0, -10.0, 170.0, 10.0),
            interval=pd.Interval(
                pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), closed="both"
            ),
            resolution=(0.1, 0.1),
            crs="EPSG:4326",
        )
        with pytest.raises(NotImplementedError, match="degenerate box"):
            sl.to_crs("+proj=geos +h=35785831 +lon_0=0")

    def test_antimeridian_crossing_bounds_rejected_at_construction(self) -> None:
        # The other half of gh #16: wrap-around bounds can't even be
        # constructed, so to_crs never sees them.
        with pytest.raises(ValueError, match="xmin < xmax"):
            GeoSlice(
                bounds=(170.0, -10.0, -170.0, 10.0),
                interval=pd.Interval(
                    pd.Timestamp("2024-01-01"),
                    pd.Timestamp("2024-01-02"),
                    closed="both",
                ),
                resolution=(0.1, 0.1),
                crs="EPSG:4326",
            )

    def test_happy_path_reprojection_unchanged(self) -> None:
        sl = GeoSlice(
            bounds=(-3.8, 40.3, -3.6, 40.5),
            interval=pd.Interval(
                pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), closed="both"
            ),
            resolution=(0.001, 0.001),
            crs="EPSG:4326",
        )
        out = sl.to_crs("EPSG:3857")
        assert out.crs.to_epsg() == 3857
        assert out.bounds[0] < out.bounds[2] and out.bounds[1] < out.bounds[3]
