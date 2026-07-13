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
