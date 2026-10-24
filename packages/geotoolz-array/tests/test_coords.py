"""Tests for the standard coordinates in `geoarray.coords` (#518)."""

from __future__ import annotations

import numpy as np
import pytest

from geoarray.coords import (
    Coordinate,
    CurvilinearGrid,
    PixelGrid,
    PointAxis,
    SpectralAxis,
    SwathGrid,
    TimeAxis,
    VariableAxis,
    Vertical,
    VerticalAxis,
)


DEPTH = Vertical(kind="depth", units="m", positive="down")


def _axes() -> list[Coordinate]:
    return [
        SpectralAxis(("B2", "B4", "B8"), (490.0, 665.0, 842.0), (65.0, 30.0, 115.0)),
        VariableAxis(("t2m", "psfc", "wind"), ("K", "hPa", "m s-1")),
        VerticalAxis((5.0, 15.0, 25.0), DEPTH),
        TimeAxis(["2026-06-01", "2026-06-11", "2026-06-21"]),
    ]


@pytest.mark.parametrize("axis", _axes(), ids=lambda a: type(a).__name__)
def test_axes_hash_and_compare_by_value(axis: Coordinate) -> None:
    twin = _axes()[[type(a) for a in _axes()].index(type(axis))]
    assert axis == twin and axis is not twin
    assert hash(axis) == hash(twin)
    assert isinstance(axis, Coordinate)
    assert axis.size == 3 and len(axis.labels) == 3


@pytest.mark.parametrize("axis", _axes(), ids=lambda a: type(a).__name__)
def test_isel_keeps_one_entry_per_index(axis: Coordinate) -> None:
    """GA-G9 at the coordinate level: a slice or positions slice every field."""
    assert axis.isel(slice(1, None)).labels == axis.labels[1:]
    picked = axis.isel([2, 0])
    assert picked.labels == (axis.labels[2], axis.labels[0])
    assert axis.isel(np.array([1])).size == 1
    assert axis.isel(slice(0, 3, 2)) == axis.isel([0, 2])


def test_spectral_fields_travel_together() -> None:
    swir = SpectralAxis(("B11", "B12"), np.array([1610.0, 2190.0]), fwhm_nm=[90, 180])
    assert swir.wavelength_nm == (1610.0, 2190.0)
    assert swir.isel([1]) == SpectralAxis(("B12",), (2190.0,), (180.0,))
    assert swir.dim == "band"


def test_band_and_variable_are_different_dims() -> None:
    assert SpectralAxis(("a",)).dim == "band"
    assert VariableAxis(("a",)).dim == "variable"
    assert SpectralAxis(("a",)) != VariableAxis(("a",))


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: SpectralAxis(("a", "a")), "repeat"),
        (lambda: SpectralAxis(("a", "b"), (1.0,)), "entries"),
        (lambda: VariableAxis(("a",), ("K", "hPa")), "entries"),
        (lambda: Vertical("altitude", "m", "up"), "kind"),
        (lambda: Vertical("height", "m", "sideways"), "positive"),
        (lambda: TimeAxis(np.zeros((2, 2), "datetime64[ns]")), "1-D"),
        (lambda: PixelGrid((3,), None), "shape"),
        (lambda: PixelGrid((3, 3), None, "corner"), "convention"),
        (lambda: CurvilinearGrid(np.zeros((2, 2)), np.zeros((2, 3)), None), "shape"),
        (lambda: PointAxis(-1, None), "non-negative"),
    ],
)
def test_invalid_coordinates_raise(build: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        build()  # type: ignore[operator]


def test_time_stamps_are_ns_and_read_only() -> None:
    dates = TimeAxis(np.array(["2026-06-01T00", "2026-06-02T12"], "datetime64[h]"))
    assert dates.stamps.dtype == np.dtype("datetime64[ns]")
    with pytest.raises(ValueError):
        dates.stamps[0] = np.datetime64("2000-01-01")
    assert TimeAxis(["2026-06-01"]) != TimeAxis(["2026-06-02"])
    assert "2026-06-01" in repr(dates)


def test_vertical_axis_keeps_its_vertical() -> None:
    levels = VerticalAxis((1000.0, 850.0, 500.0), Vertical("pressure", "hPa", "down"))
    assert levels.isel([0]).vertical.units == "hPa"


def test_grids_hash_and_carry_their_crs() -> None:
    grid = PixelGrid((600, 600), "EPSG:32613")
    assert grid == PixelGrid([600, 600], 32613)
    assert hash(grid) == hash(PixelGrid((600, 600), "EPSG:32613"))
    assert grid.crs is not None and grid.crs.to_epsg() == 32613
    assert SwathGrid((4, 5), "EPSG:4326").dims == ("row", "col")
    assert PointAxis(40, "EPSG:4326").dims == ("obs",)


def test_curvilinear_grid_compares_by_value() -> None:
    lat, lon = np.meshgrid(np.arange(3.0), np.arange(4.0), indexing="ij")
    a = CurvilinearGrid(lat, lon, "EPSG:4326")
    b = CurvilinearGrid(lat.copy(), lon.copy(), 4326)
    assert a == b and hash(a) == hash(b)
    assert a != CurvilinearGrid(lat + 1, lon, "EPSG:4326")
    assert a.shape == (3, 4) and a.dims == ("row", "col")


@pytest.mark.parametrize("convention", ["area", "point"])
def test_pixel_centres(convention: str) -> None:
    """Centres follow rasterio.transform.xy(offset='center') for each convention.

    rasterio maps (row i, col j) to T·(j + ½, i + ½) for an area transform;
    a point transform already maps integer indices to centres.
    """
    t = (20.0, 0.5, 600000.0, -0.3, -20.0, 3550000.0)
    grid = PixelGrid((3, 4), "EPSG:32613", convention)
    x, y = grid.centres(t)
    off = 0.5 if convention == "area" else 0.0
    i, j = 2, 3
    assert x[i, j] == pytest.approx(t[0] * (j + off) + t[1] * (i + off) + t[2])
    assert y[i, j] == pytest.approx(t[3] * (j + off) + t[4] * (i + off) + t[5])
    # the area transform of a point grid puts the same centres at + ½
    area = PixelGrid((3, 4), "EPSG:32613")
    xa, ya = area.centres(grid.area_transform(np.asarray(t)))
    np.testing.assert_allclose(xa, x, rtol=0, atol=1e-9)
    np.testing.assert_allclose(ya, y, rtol=0, atol=1e-9)


# -- review fixes (Codex, #545) --------------------------------------------------------


def test_curvilinear_hash_follows_equality_for_signed_zero_and_nan() -> None:
    lat = np.array([[0.0, np.nan]])
    other = np.array([[-0.0, np.float64("nan") * -1]])
    a = CurvilinearGrid(lat, lat, "EPSG:4326")
    b = CurvilinearGrid(other, other, "EPSG:4326")
    assert a == b
    assert hash(a) == hash(b)


def test_time_axis_with_nat_equals_itself() -> None:
    t = TimeAxis(np.array(["2026-01-01", "NaT"], dtype="datetime64[ns]"))
    same = TimeAxis(np.array(["2026-01-01", "NaT"], dtype="datetime64[ns]"))
    assert t == same
    assert hash(t) == hash(same)


def test_point_grid_area_transform_takes_a_tuple() -> None:
    grid = PixelGrid(shape=(2, 2), crs="EPSG:32613", convention="point")
    out = grid.area_transform((20.0, 0.0, 100.0, 0.0, -20.0, 200.0))
    np.testing.assert_allclose(out, [20.0, 0.0, 90.0, 0.0, -20.0, 210.0])
