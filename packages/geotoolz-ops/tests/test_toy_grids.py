"""Smoke tests for the ``_helpers.TOY_GRIDS`` fixture axis.

Each named grid must actually exhibit the property it exists to probe;
otherwise the family tests that use it would pass vacuously.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from _helpers import DEFAULT_CRS, DEFAULT_TRANSFORM, TOY_GRIDS, toy_geotensor
from pyproj import CRS


def _pixel_sizes(t: rasterio.Affine) -> tuple[float, float]:
    """Ground length of one column step and one row step."""
    return float(np.hypot(t.a, t.d)), float(np.hypot(t.b, t.e))


def test_default_grid_is_unchanged() -> None:
    gt = toy_geotensor(np.zeros((2, 3, 3)))
    assert gt.transform == DEFAULT_TRANSFORM == TOY_GRIDS["utm"][0]
    assert gt.crs == DEFAULT_CRS


@pytest.mark.parametrize("grid", sorted(TOY_GRIDS))
def test_grid_sets_transform_and_crs(grid: str) -> None:
    transform, crs = TOY_GRIDS[grid]
    gt = toy_geotensor(np.zeros((2, 3, 3)), grid=grid)
    assert gt.transform == transform
    assert CRS.from_user_input(gt.crs) == CRS.from_user_input(crs)
    assert transform.determinant != 0.0


def test_explicit_transform_and_crs_override_the_grid() -> None:
    other = rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    gt = toy_geotensor(
        np.zeros((3, 3)), grid="geographic", transform=other, crs="EPSG:32630"
    )
    assert gt.transform == other
    assert gt.crs == "EPSG:32630"


def test_non_square_grid_has_distinct_pixel_width_and_height() -> None:
    t, _ = TOY_GRIDS["non_square"]
    assert t.is_rectilinear
    assert _pixel_sizes(t) == (10.0, 20.0)


def test_rotated_grid_is_rotated_with_square_pixels() -> None:
    t, _ = TOY_GRIDS["rotated"]
    assert not t.is_rectilinear
    assert t.b != 0.0 and t.d != 0.0
    # a / e are no longer the pixel size; the true size is still 10 m.
    assert abs(t.a) != pytest.approx(10.0)
    np.testing.assert_allclose(_pixel_sizes(t), (10.0, 10.0))
    assert abs(t.determinant) == pytest.approx(100.0)


def test_sheared_grid_shears_only_columns() -> None:
    t, _ = TOY_GRIDS["sheared"]
    assert t.b != 0.0 and t.d == 0.0
    # a / e look like a square 10 m grid; only b reveals the shear.
    assert (t.a, t.e) == (10.0, -10.0)
    assert abs(t.determinant) == pytest.approx(100.0)


def test_geographic_grid_is_geographic() -> None:
    t, crs = TOY_GRIDS["geographic"]
    assert CRS.from_user_input(crs).is_geographic
    assert t.is_rectilinear
    lon, lat = t * (0, 0)
    assert -180.0 <= lon <= 180.0
    assert -90.0 <= lat <= 90.0


def test_grid_composes_with_fill_pixels() -> None:
    gt = toy_geotensor(np.ones((2, 3, 3)), grid="non_square", with_fill_pixels=True)
    assert gt.transform == TOY_GRIDS["non_square"][0]
    assert np.asarray(gt)[0, 0, 0] == -9999
