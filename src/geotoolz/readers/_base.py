"""Shared reader primitives for sensor-specific modules."""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from affine import Affine
from georeader.abstract_reader import GeoData
from georeader.geotensor import GeoTensor
from rasterio.windows import (
    Window,
    from_bounds as window_from_bounds,
    transform as window_transform,
)


Track = Literal["A", "B"]


class SensorReader(GeoData, ABC):
    """ABC for georeader-compatible sensor readers.

    Subclasses provide sensor-specific metadata and implement
    :meth:`_read_window`; the base class supplies the small GeoData surface
    needed by geotoolz operators.

    Examples:
        Implement a file-backed sensor reader::

            class Reader(SensorReader):
                def __init__(self, path): ...
                def _read_window(self, window): ...

        Load a full scene into a ``GeoTensor``::

            scene = Reader("scene.dat").load()

        Read a tile without changing the reader::

            tile = Reader("scene.dat").read_from_window(Window(0, 0, 256, 256))
    """

    @abstractmethod
    def _read_window(self, window: Window) -> np.ndarray:
        """Read a sensor-native pixel window into a numpy array."""

    @property
    @abstractmethod
    def _crs(self) -> Any:
        """Reader CRS."""

    @property
    @abstractmethod
    def _transform(self) -> Affine:
        """Reader affine transform."""

    @property
    @abstractmethod
    def _shape(self) -> tuple[int, ...]:
        """Reader array shape as ``(..., height, width)``."""

    @property
    @abstractmethod
    def _dtype(self) -> Any:
        """Reader array dtype."""

    @property
    @abstractmethod
    def _bands(self) -> Sequence[str]:
        """Band names in array order."""

    @property
    @abstractmethod
    def _fill_value(self) -> Any:
        """Default fill value for boundless reads."""

    @property
    @abstractmethod
    def _track(self) -> Track:
        """Track A for clean affine grids, Track B for irregular geolocation."""

    @property
    def crs(self) -> Any:
        """Reader CRS."""
        return self._crs

    @property
    def transform(self) -> Affine:
        """Affine transform for Track A readers."""
        return self._transform

    @property
    def shape(self) -> tuple[int, ...]:
        """Array shape as ``(..., height, width)``."""
        return self._shape

    @property
    def dtype(self) -> Any:
        """Numpy dtype read by this reader."""
        return self._dtype

    @property
    def dims(self) -> list[str]:
        """Dimension names compatible with georeader ``GeoData``."""
        return ["band", "y", "x"] if len(self.shape) == 3 else ["y", "x"]

    @property
    def bands(self) -> tuple[str, ...]:
        """Band names in array order."""
        return tuple(self._bands)

    @property
    def fill_value_default(self) -> Any:
        """Default fill value for out-of-bounds pixels."""
        return self._fill_value

    @property
    def track(self) -> Track:
        """Sensor reader track classification."""
        return self._track

    def load(self, boundless: bool = True) -> GeoTensor:
        """Load the reader's full extent into a ``GeoTensor``."""
        window = Window(col_off=0, row_off=0, width=self.width, height=self.height)
        return self.read_from_window(window, boundless=boundless)

    def read_from_window(self, window: Window, boundless: bool = True) -> GeoTensor:
        """Read a pixel window as a ``GeoTensor``."""
        if not boundless:
            window = self._clip_window(window)
        values = self._read_window(window)
        attrs = {"band_names": self.bands, "bands": self.bands}
        return GeoTensor(
            values,
            transform=window_transform(window, self.transform),
            crs=self.crs,
            fill_value_default=self.fill_value_default,
            attrs=attrs,
        )

    def read_from_bounds(
        self,
        bounds: tuple[float, float, float, float],
        boundless: bool = True,
    ) -> GeoTensor:
        """Read map-coordinate bounds as a ``GeoTensor``."""
        window = window_from_bounds(*bounds, transform=self.transform)
        return self.read_from_window(window.round_offsets().round_lengths(), boundless)

    def read_from_center_coords(
        self,
        x: float,
        y: float,
        width: int,
        height: int,
        boundless: bool = True,
    ) -> GeoTensor:
        """Read a window centered on map coordinates."""
        col_float, row_float = ~self.transform * (x, y)
        row = int(np.floor(row_float))
        col = int(np.floor(col_float))
        window = Window(
            col_off=col - width // 2,
            row_off=row - height // 2,
            width=width,
            height=height,
        )
        return self.read_from_window(window, boundless)

    def _clip_window(self, window: Window) -> Window:
        base = Window(col_off=0, row_off=0, width=self.width, height=self.height)
        return window.intersection(base)


def require_optional_dependency(package: str, *, extra: str) -> None:
    """Raise an actionable error when a sensor optional dependency is missing."""
    if importlib.util.find_spec(package) is not None:
        return
    raise ImportError(
        f"Missing optional dependency {package!r} required for "
        f"geotoolz.readers.{extra}. Install it with "
        f"`pip install 'geotoolz[{extra}]'`."
    )


def as_path(path: str | Path) -> Path:
    """Normalize a reader path argument."""
    return Path(path)
