"""`to_geoswath`: a model field on a lat-lon grid as a swath (#757).

The field and the grid are stubs: geoarray never imports gridax.
"""

from __future__ import annotations

import dataclasses
import sys
from typing import Any

import numpy as np
import pytest

import geoarray
from geoarray import GeoSwath
from geoarray.coords import CurvilinearGrid, VariableAxis


@dataclasses.dataclass
class Field:
    """What `gridax.CField` shows of itself."""

    data: Any
    dims: tuple[str, ...]
    stagger: tuple[tuple[str, str], ...] = ()


@dataclasses.dataclass
class Sphere:
    """What gridax's spherical grids show of themselves."""

    lat: np.ndarray
    lon: np.ndarray
    ghost: int = 1
    R: float = 6.371e6
    axes: tuple[str, ...] = ("y", "x")

    def coords(self, axis: str, *, at: str) -> np.ndarray:
        assert at == "T"
        return {"y": self.lat, "x": self.lon}[axis]


@pytest.fixture
def sphere() -> Sphere:
    lat, lon = np.linspace(-35.0, 35.0, 8), np.linspace(-5.0, 65.0, 9)
    return Sphere(np.deg2rad(lat), np.deg2rad(lon))


def _by_hand(data: np.ndarray, grid: Sphere, dims: str, **kwargs: Any) -> GeoSwath:
    """The interior of ``data``, built as before `to_geoswath`."""
    lat, lon = np.meshgrid(np.rad2deg(grid.lat), np.rad2deg(grid.lon), indexing="ij")
    cells = CurvilinearGrid(lat[1:-1, 1:-1], lon[1:-1, 1:-1], "EPSG:4326")
    return GeoSwath.from_grid(data[..., 1:-1, 1:-1], cells, dims=dims, **kwargs)


def _data(*shape: int) -> np.ndarray:
    return np.random.default_rng(0).random(shape)


def test_equals_the_hand_built_swath(sphere: Sphere) -> None:
    data = _data(8, 9)
    swath = geoarray.to_geoswath(Field(data, ("y", "x")), sphere)
    assert swath.identical(_by_hand(data, sphere, "row col"))
    assert swath.shape == (6, 7)


def test_other_dims_keep_their_order(sphere: Sphere) -> None:
    data = _data(3, 2, 8, 9)
    want = _by_hand(data, sphere, "time member row col")
    got = geoarray.to_geoswath(Field(data, ("time", "member", "y", "x")), sphere)
    assert got.identical(want)
    # any order of the stored dims
    moved = Field(np.moveaxis(data, -1, 0), ("x", "time", "member", "y"))
    assert geoarray.to_geoswath(moved, sphere).identical(want)


def test_coords_fill_and_attrs_are_the_carrier_s(sphere: Sphere) -> None:
    data = _data(2, 8, 9)
    extra: dict[str, Any] = {
        "coords": (VariableAxis(("h", "q"), ("m", "1")),),
        "fill_value": np.nan,
        "attrs": {"model": "toy"},
    }
    got = geoarray.to_geoswath(Field(data, ("variable", "y", "x")), sphere, **extra)
    assert got.identical(_by_hand(data, sphere, "variable row col", **extra))


def test_the_ring_of_a_vertical_axis_is_dropped_too(sphere: Sphere) -> None:
    grid = dataclasses.replace(sphere, axes=("z", "y", "x"))
    data = _data(2, 5, 8, 9)
    got = geoarray.to_geoswath(Field(data, ("time", "z", "y", "x")), grid)
    assert got.dims == ("time", "z", "row", "col")
    np.testing.assert_array_equal(got.data, data[:, 1:-1, 1:-1, 1:-1])


def test_a_grid_without_a_ghost_ring_keeps_every_cell(sphere: Sphere) -> None:
    grid = dataclasses.replace(sphere, ghost=0)
    got = geoarray.to_geoswath(Field(_data(8, 9), ("y", "x")), grid)
    np.testing.assert_allclose(got.lat[:, 0], np.linspace(-35.0, 35.0, 8))
    np.testing.assert_allclose(got.lon[0], np.linspace(-5.0, 65.0, 9))


def test_two_dimensional_latitudes_and_longitudes(sphere: Sphere) -> None:
    lat, lon = np.meshgrid(sphere.lat, sphere.lon, indexing="ij")

    class Tripolar:
        ghost = 1
        lat_T = lat + 0.1 * lon
        lon_T = lon

    data = _data(8, 9)
    got = geoarray.to_geoswath(Field(data, ("y", "x")), Tripolar())
    np.testing.assert_allclose(got.lat, np.rad2deg(Tripolar.lat_T)[1:-1, 1:-1])
    np.testing.assert_allclose(got.lon, np.rad2deg(lon)[1:-1, 1:-1])


def test_a_staggered_field_raises(sphere: Sphere) -> None:
    u = Field(_data(8, 9), ("y", "x"), (("x", "right"),))
    with pytest.raises(ValueError, match=r"gridax\.interpolate"):
        geoarray.to_geoswath(u, sphere)


def test_dims_and_shapes_are_checked(sphere: Sphere) -> None:
    with pytest.raises(ValueError, match="without"):
        geoarray.to_geoswath(Field(_data(8, 9), ("row", "x")), sphere)
    with pytest.raises(ValueError, match="along"):
        geoarray.to_geoswath(Field(_data(8, 10), ("y", "x")), sphere)


def test_a_cartesian_grid_raises(sphere: Sphere) -> None:
    class Plane:
        ghost = 1

        def coords(self, axis: str, *, at: str) -> np.ndarray:
            return np.arange(9.0)

    with pytest.raises(TypeError, match="Plane has no latitudes"):
        geoarray.to_geoswath(Field(_data(9, 9), ("y", "x")), Plane())


def test_gridax_is_not_imported() -> None:
    assert "gridax" not in sys.modules


def test_jax_data_stays_jax(sphere: Sphere, x64: None) -> None:
    jnp = pytest.importorskip("jax.numpy")
    data = _data(8, 9)
    got = geoarray.to_geoswath(Field(jnp.asarray(data), ("y", "x")), sphere)
    assert got.backend == "jax"
    want = _by_hand(data, sphere, "row col")
    np.testing.assert_array_equal(np.asarray(got.data), want.data)
    np.testing.assert_array_equal(np.asarray(got.lat), want.lat)
