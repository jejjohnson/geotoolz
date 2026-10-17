"""`RasterSource`: lazy readers, `GeoArray` as a source, georeader's readers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import CRS, TRANSFORM

from geoarray import (
    AsyncRasterSource,
    FrozenAttrs,
    GeoArray,
    Grid,
    PixelWindow,
    RasterSource,
)


class ToySource:
    """A minimal lazy reader: pixels made on demand from their indices."""

    def __init__(self) -> None:
        self.reads: list[PixelWindow] = []

    @property
    def grid(self) -> Grid:
        return Grid(TRANSFORM, CRS, (6, 8))

    @property
    def shape(self) -> tuple[int, ...]:
        return (2, 6, 8)

    @property
    def dtype(self) -> Any:
        return np.dtype(np.float32)

    @property
    def fill_value(self) -> Any:
        return -1.0

    @property
    def attrs(self) -> FrozenAttrs:
        return FrozenAttrs(band_names=["a", "b"])

    def load(self) -> GeoArray:
        data = np.arange(2 * 6 * 8, dtype=np.float32).reshape(self.shape)
        return GeoArray(data, TRANSFORM, CRS, self.fill_value, self.attrs)

    def read_window(self, window: PixelWindow, *, boundless: bool = True) -> GeoArray:
        self.reads.append(window)
        return self.load().read_window(window, boundless=boundless)


def _window(c: int, r: int, w: int, h: int) -> PixelWindow:
    return PixelWindow(col_off=c, row_off=r, width=w, height=h)


def test_geoarray_is_a_raster_source() -> None:
    g = ToySource().load()
    assert isinstance(g, RasterSource)
    assert g.load() is g
    assert g.read_window(_window(1, 2, 3, 3)).identical(g[..., 2:5, 1:4])
    padded = g.read_window(_window(-1, 0, 2, 2))
    assert padded.data[0, 0, 0] == -1.0
    assert g.read_window(_window(-1, 0, 2, 2), boundless=False).shape == (2, 2, 1)


def test_protocols_check_members() -> None:
    assert isinstance(ToySource(), RasterSource)
    assert not isinstance(np.zeros(3), RasterSource)

    class AsyncToy(ToySource):
        async def aread_window(
            self, window: PixelWindow, *, boundless: bool = True
        ) -> GeoArray:
            return self.read_window(window, boundless=boundless)

        async def aload(self) -> GeoArray:
            return self.load()

    assert isinstance(AsyncToy(), AsyncRasterSource)
    assert not isinstance(ToySource(), AsyncRasterSource)


@pytest.fixture
def tif(tmp_path: Path) -> Path:
    rasterio = pytest.importorskip("rasterio")
    from affine import Affine

    path = tmp_path / "scene.tif"
    data = np.arange(2 * 10 * 12, dtype=np.int16).reshape(2, 10, 12)
    profile = {
        "driver": "GTiff",
        "width": 12,
        "height": 10,
        "count": 2,
        "dtype": "int16",
        "crs": CRS,
        "transform": Affine(*TRANSFORM),
        "nodata": -9999,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


def test_source_from_a_rasterio_reader(tif: Path) -> None:
    pytest.importorskip("georeader.rasterio_reader")
    from georeader.rasterio_reader import RasterioReader

    from geoarray.interop.georeader import source_from_georeader

    source = source_from_georeader(RasterioReader(str(tif)))
    assert isinstance(source, RasterSource)
    assert source.grid == Grid(TRANSFORM, CRS, (10, 12))
    assert source.shape == (2, 10, 12)
    assert source.dtype == np.int16
    assert source.fill_value == -9999
    full = source.load()
    assert isinstance(full, GeoArray)
    assert full.shape == (2, 10, 12)
    window = _window(3, 4, 5, 2)
    part = source.read_window(window)
    assert part.identical(full[..., 4:6, 3:8])
    edge = source.read_window(_window(10, 8, 4, 4))
    assert edge.shape == (2, 4, 4)
    assert (edge.data[:, 2:, :] == -9999).all()
    assert edge.grid == source.grid.crop(_window(10, 8, 4, 4))
    assert "source_from_georeader" in repr(source)


def test_source_from_a_geotensor_converts_at_once() -> None:
    geotensor = pytest.importorskip("georeader.geotensor")
    from affine import Affine

    from geoarray.interop.georeader import source_from_georeader

    gt = geotensor.GeoTensor(np.zeros((3, 3)), Affine(*TRANSFORM), CRS)
    assert isinstance(source_from_georeader(gt), GeoArray)
    with pytest.raises(TypeError, match="lacks"):
        source_from_georeader(object())


def test_georeader_functions_read_a_raster_source() -> None:
    pytest.importorskip("georeader.read")
    from georeader import read

    from geoarray.interop.georeader import from_georeader, source_to_georeader

    toy = ToySource()
    view = source_to_georeader(toy)
    assert view.shape == (2, 6, 8)
    assert view.width == 8
    assert view.height == 6
    assert view.fill_value_default == -1.0
    assert view.dims == ["band", "y", "x"]
    assert view.attrs == {"band_names": ("a", "b")}
    assert view.crs == CRS
    np.testing.assert_array_equal(view.values, toy.load().data)
    bounds = toy.grid.bounds_of(_window(2, 1, 4, 3))
    tile = from_georeader(read.read_from_bounds(view, bounds))
    assert tile.identical(toy.load().read_window(_window(2, 1, 4, 3)))
    assert toy.reads  # read window by window, not by loading
    assert from_georeader(view.load()).identical(toy.load())
    assert "source_to_georeader" in repr(view)


def test_source_to_georeader_round_trips_wrappers(tif: Path) -> None:
    pytest.importorskip("georeader.rasterio_reader")
    from georeader.geotensor import GeoTensor
    from georeader.rasterio_reader import RasterioReader

    from geoarray.interop.georeader import source_from_georeader, source_to_georeader

    reader = RasterioReader(str(tif))
    assert source_to_georeader(source_from_georeader(reader)) is reader
    assert isinstance(source_to_georeader(ToySource().load()), GeoTensor)
    with pytest.raises(TypeError, match="takes a RasterSource"):
        source_to_georeader(np.zeros(3))  # type: ignore[arg-type]


def test_fractional_windows_are_refused() -> None:
    pytest.importorskip("georeader.read")
    from rasterio.windows import Window

    from geoarray.interop.georeader import source_to_georeader

    view = source_to_georeader(ToySource())
    with pytest.raises(ValueError, match="whole number"):
        view.read_from_window(Window(0.5, 0, 2, 2))
