"""Readers name their dims (#527): band labels, wavelengths, a swath reader."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pytest
from geoarray import GeoArray, GeoSwath, PixelWindow
from geoarray.coords import SpectralAxis

from geoproducts import stack, toy_sensor


TRANSFORM = (0.1, 0.0, 10.0, 0.0, -0.1, 50.0)


@contextmanager
def no_rank_dims() -> Iterator[None]:
    """Fail on the 0.3 'dims read from the data's rank' deprecation."""
    with warnings.catch_warnings():
        warnings.filterwarnings("error", "GeoArray dims read from", DeprecationWarning)
        yield


def test_toy_reads_carry_band_names_and_wavelengths() -> None:
    reader = toy_sensor.Reader("scene", data=np.zeros((4, 6, 6), dtype=np.float32))

    with no_rank_dims():
        whole = reader.load()
        chip = reader.read_window(PixelWindow(col_off=2, row_off=2, width=3, height=3))

    want = SpectralAxis(("blue", "green", "red", "nir"), (469.0, 555.0, 645.0, 858.0))
    assert whole.dims == chip.dims == ("band", "y", "x")
    assert reader.coords == (want,)
    assert whole.coord("band") == chip.coord("band") == want


def test_a_reader_without_a_band_dim_has_no_coordinates() -> None:
    from geoproducts._src.base import ProductReader

    class Mask(ProductReader):
        _crs = "EPSG:4326"
        _transform = TRANSFORM
        _shape = (3, 4)
        _dtype = np.uint8
        _bands = ("mask",)
        _fill_value = 0
        _track = "A"

        def _read_window(self, window):  # type: ignore[no-untyped-def]
            return np.zeros((window.height, window.width), dtype=np.uint8)

    reader = Mask()
    with no_rank_dims():
        out = reader.load()
    assert out.dims == ("y", "x") and reader.coords == () and out.coords == ()


def test_stack_keeps_the_readers_dims_and_band_axis() -> None:
    parts = [
        GeoArray(
            np.ones((1, 3, 4), np.float32),
            TRANSFORM,
            4326,
            fill_value=np.nan,
            dims="band y x",
            coords=(SpectralAxis((name,), (wl,)),),
        )
        for name, wl in (("a", 500.0), ("b", 600.0))
    ]

    with no_rank_dims():
        out = stack(parts, resampling="nearest")

    assert out.dims == ("band", "y", "x")
    assert out.coord("band") == SpectralAxis(("a", "b"), (500.0, 600.0))


def test_stack_keeps_the_fwhm_of_every_reader() -> None:
    """Review on #573: geoarray.concat joins the whole axis, FWHM included."""
    parts = [
        GeoArray(
            np.ones((1, 3, 4), np.float32),
            TRANSFORM,
            4326,
            fill_value=np.nan,
            dims="band y x",
            coords=(SpectralAxis((name,), (wl,), (fw,)),),
        )
        for name, wl, fw in (("a", 500.0, 20.0), ("b", 600.0, 30.0))
    ]

    out = stack(parts, resampling="nearest")

    assert out.coord("band").fwhm_nm == (20.0, 30.0)


def test_stack_names_the_bands_of_readers_without_band_attrs() -> None:
    """Review of the stack: readers name bands in coords, not only in attrs."""
    readers = [
        toy_sensor.Reader(
            "a", data=np.zeros((4, 3, 4), np.float32), transform=TRANSFORM
        ),
        GeoArray(
            np.ones((1, 3, 4), np.float32),
            TRANSFORM,
            4326,
            fill_value=np.nan,
            dims="band y x",
            coords=(SpectralAxis(("swir",)),),
        ),
    ]

    out = stack(readers, resampling="nearest")

    assert out.coord("band").names == ("blue", "green", "red", "nir", "swir")
    assert out.coord("band").wavelength_nm is None  # the GeoArray declares none


def _lat_lon(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    lat, lon = np.meshgrid(
        np.linspace(40.0, 41.0, rows), np.linspace(-1.0, 0.0, cols), indexing="ij"
    )
    return lat, lon


def test_the_toy_swath_reader_returns_a_geoswath() -> None:
    lat, lon = _lat_lon(3, 5)
    data = np.arange(4 * 3 * 5, dtype=np.float32).reshape(4, 3, 5)
    reader = toy_sensor.SwathReader("granule", data=data, lat=lat, lon=lon)

    swath = reader.load()
    part = reader.read_window(PixelWindow(col_off=1, row_off=0, width=2, height=2))

    assert isinstance(swath, GeoSwath) and swath.dims == ("band", "row", "col")
    assert swath.coord("band").names == ("blue", "green", "red", "nir")
    np.testing.assert_array_equal(swath.lat, lat)
    assert part.data.shape == (4, 2, 2)
    np.testing.assert_array_equal(part.lon, lon[0:2, 1:3])


def test_the_toy_swath_round_trips_through_xarray() -> None:
    pytest.importorskip("xarray")

    lat, lon = _lat_lon(3, 5)
    swath = toy_sensor.SwathReader(
        "granule", data=np.ones((4, 3, 5), np.float32), lat=lat, lon=lon
    ).load()

    da = swath.to_xarray()
    back = GeoSwath.from_xarray(da)

    assert da["lat"].dims == ("row", "col") and da["lon"].dims == ("row", "col")
    assert isinstance(back, GeoSwath) and back.dims == swath.dims
    np.testing.assert_array_equal(back.lat, lat)
    assert back.coord("band") == swath.coord("band")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"data": np.ones((3, 3, 5))}, r"\(4, row, col\)"),
        ({"lat": np.ones((2, 5))}, "must be"),
    ],
)
def test_the_toy_swath_reader_checks_its_shapes(kwargs: dict, match: str) -> None:
    lat, lon = _lat_lon(3, 5)
    args = {"data": np.ones((4, 3, 5)), "lat": lat, "lon": lon, **kwargs}
    with pytest.raises(ValueError, match=match):
        toy_sensor.SwathReader("granule", **args)


def test_band_values_reads_one_number_per_band_in_band_order() -> None:
    """Review on #573: the shared band-table lookup both toy readers use."""
    from geoproducts._src.constants import band_values
    from geoproducts.toy_sensor import constants

    assert band_values(constants.BANDS, ("nir", "blue"), "wavelength_nm") == (
        858.0,
        469.0,
    )
    with pytest.raises(KeyError):
        band_values(constants.BANDS, ("swir",), "wavelength_nm")


def test_stack_drops_units_unless_every_reader_has_them() -> None:
    """Review on #573: one reader with per-band units, one without."""
    with_units = GeoArray(
        np.ones((1, 3, 4), np.float32),
        TRANSFORM,
        4326,
        fill_value=np.nan,
        attrs={"units": ("K",)},
        dims="band y x",
        coords=(SpectralAxis(("C13",)),),
    )
    toy = toy_sensor.Reader(
        "t", data=np.zeros((4, 3, 4), np.float32), transform=TRANSFORM
    )

    out = stack([with_units, toy], resampling="nearest")

    assert out.count == 5 and "units" not in out.attrs


def test_a_swath_window_past_the_edge_is_clipped_and_one_outside_raises() -> None:
    lat, lon = _lat_lon(3, 5)
    reader = toy_sensor.SwathReader("g", data=np.ones((4, 3, 5)), lat=lat, lon=lon)

    edge = reader.read_window(PixelWindow(col_off=3, row_off=-1, width=4, height=2))

    assert edge.data.shape == (4, 1, 2)
    np.testing.assert_array_equal(edge.lat, lat[0:1, 3:5])
    with pytest.raises(ValueError, match="misses"):
        reader.read_window(PixelWindow(col_off=9, row_off=0, width=2, height=2))
