"""Shared test helpers: field builders and minimal toy `Field` classes.

Kept deliberately small — only helpers that were previously duplicated
verbatim across test modules live here. Variants with genuinely
different semantics (e.g. a ``select`` that ignores slice indexers, or
a ``with_data`` that returns an inspection tuple) stay local to their
test module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import rasterio
from georeader.geotensor import GeoTensor

from geopatcher import RasterField


def make_raster_field(
    size: int = 64,
    *,
    dtype: type = np.float32,
    crs: str = "EPSG:32630",
) -> RasterField:
    """Standard test field: an NxN arange ramp over an identity transform.

    Reproduces bit-for-bit the fixture previously redeclared per module:
    ``np.arange(size * size, dtype=dtype).reshape(size, size)``.
    """
    arr = np.arange(size * size, dtype=dtype).reshape(size, size)
    return RasterField(
        GeoTensor(
            values=arr,
            transform=rasterio.Affine.identity(),
            crs=crs,
        )
    )


@dataclass
class StubDomain:
    """Minimal `Domain` — bounds and CRS are never read in most tests."""

    crs: str = "EPSG:4326"
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)


class StubField:
    """Minimal `Field` — `select` returns its name; `with_data` echoes."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._domain = StubDomain()

    @property
    def domain(self) -> Any:
        return self._domain

    def select(self, indexer: Any) -> Any:
        return f"{self._name}@{indexer}"

    def with_data(self, array: Any) -> Any:
        return array


class ArrField:
    """Minimal `Field` whose `select` returns a backing numpy array.

    Slice indexers slice the backing array; any other indexer returns
    the full array.
    """

    def __init__(self, values: np.ndarray) -> None:
        self._values = values
        self._domain = StubDomain()

    @property
    def domain(self) -> Any:
        return self._domain

    def select(self, indexer: Any) -> Any:
        if isinstance(indexer, slice):
            return self._values[indexer]
        return self._values

    def with_data(self, array: Any) -> Any:
        return array


class ArrayField:
    """Toy raster-like `Field` backed by a plain 2-D numpy array.

    `select` honours rasterio-style windows (row_off/col_off/height/width);
    `with_data` rewraps into a fresh `ArrayField`.
    """

    def __init__(self, array: np.ndarray) -> None:
        self.array = array
        self.shape = array.shape
        self.transform = rasterio.Affine.identity()
        self.crs = "EPSG:32630"

    @property
    def domain(self) -> ArrayField:
        return self

    def select(self, window: Any) -> np.ndarray:
        rows = slice(int(window.row_off), int(window.row_off + window.height))
        cols = slice(int(window.col_off), int(window.col_off + window.width))
        return self.array[rows, cols]

    def with_data(self, array: Any) -> ArrayField:
        return ArrayField(np.asarray(array))


def write_test_geotiff(
    path: Any,
    *,
    size: tuple[int, int] = (70, 70),
    bands: int = 2,
    nodata: float = -1.0,
    crs: str = "EPSG:32630",
) -> Any:
    """Write a small float32 GeoTIFF with a real UTM transform and nodata.

    Pixel ``(b, r, c)`` holds ``b * 10_000 + r * 100 + c`` so every read
    is easy to check against rasterio. The default 70x70 extent is
    deliberately misaligned with 16-px chips so edge handling engages.

    Returns:
        ``path``, for chaining.
    """
    height, width = size
    b, r, c = np.meshgrid(
        np.arange(bands), np.arange(height), np.arange(width), indexing="ij"
    )
    values = (b * 10_000 + r * 100 + c).astype(np.float32)
    transform = rasterio.Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_600_000.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(values)
    return path


def make_rasterio_reader_field(path: Any, **kwargs: Any) -> RasterField:
    """File-backed `RasterField` over a `RasterioReader` (lazy, real GeoTIFF).

    Writes the GeoTIFF via `write_test_geotiff` (``kwargs`` forwarded) and
    wraps a `georeader.rasterio_reader.RasterioReader` over it — the
    quickstart's file-backed case, as opposed to the in-memory
    `GeoTensor` fixtures above.
    """
    from georeader.rasterio_reader import RasterioReader

    write_test_geotiff(path, **kwargs)
    return RasterField(RasterioReader(str(path)))


# Spawn-worker entry points for `test_pickle.py`. They live here, not in a
# test module, so a spawned child imports them without re-collecting tests.
_WORKER_VIEW: Any = None


def init_view_worker(view: Any) -> None:
    """Pool initializer: the view arrives pickled, once per worker."""
    global _WORKER_VIEW
    _WORKER_VIEW = view


def read_view_item(i: int) -> tuple[np.ndarray, tuple[float, ...], str]:
    """Read ``view[i]`` in the worker; return pixels + georeferencing."""
    data = _WORKER_VIEW[i].data
    return np.asarray(data), tuple(data.transform)[:6], str(data.crs)
