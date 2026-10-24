"""Shared reader primitives for sensor-specific modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from affine import Affine
from geoarray import DIMS, FrozenAttrs, GeoArray, Grid, PixelWindow


if TYPE_CHECKING:
    import pyproj


Track = Literal["A", "B"]


class ProductReader(ABC):
    """ABC for product readers: each one a lazy `geoarray.RasterSource`.

    Subclasses provide sensor-specific metadata and implement
    :meth:`_read_window`; the base class supplies the `RasterSource`
    surface (``grid``, ``shape``, ``dtype``, ``fill_value``, ``attrs``,
    ``read_window``, ``load``) and georeader's map-coordinate reads
    (``read_from_bounds``, ``read_from_center_coords``), all returning a
    `geoarray.GeoArray`.

    Examples:
        Implement a file-backed product reader::

            class Reader(ProductReader):
                def __init__(self, path): ...
                def _read_window(self, window: PixelWindow) -> np.ndarray: ...

        Load a full scene into a ``GeoArray``::

            scene = Reader("scene.dat").load()  # (C, H, W)

        Read a tile without changing the reader::

            window = PixelWindow(col_off=0, row_off=0, width=256, height=256)
            tile = Reader("scene.dat").read_window(window)  # (C, 256, 256)
    """

    @abstractmethod
    def _read_window(self, window: PixelWindow) -> np.ndarray:
        """Read a sensor-native pixel window into a numpy array.

        Implementations receive windows clipped to the grid when
        ``boundless=False`` and may receive windows partly or wholly
        outside it when ``boundless=True`` (:meth:`_read_boundless`
        handles both).
        """

    @property
    @abstractmethod
    def _crs(self) -> Any:
        """Reader CRS (anything `geoarray.as_crs` accepts)."""

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
        """Fill value for nodata and out-of-grid pixels."""

    @property
    @abstractmethod
    def _track(self) -> Track:
        """Track A for clean affine grids, Track B for irregular geolocation."""

    # -- RasterSource metadata ---------------------------------------------

    @cached_property
    def grid(self) -> Grid:
        """The pixel grid: transform, CRS and ``(H, W)``."""
        return Grid.from_affine(self._transform, self._crs, self._shape[-2:])

    @property
    def crs(self) -> pyproj.CRS | None:
        """Reader CRS (``grid.crs``)."""
        return self.grid.crs

    @property
    def transform(self) -> Affine:
        """Affine transform of the grid (``grid.affine``), for Track A readers."""
        return self.grid.affine

    @property
    def res(self) -> tuple[float, float]:
        """Pixel size ``(x, y)`` in CRS units (``grid.res``)."""
        return self.grid.res

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(minx, miny, maxx, maxy)`` of the grid (``grid.bounds``)."""
        return self.grid.bounds

    @property
    def shape(self) -> tuple[int, ...]:
        """Array shape as ``(..., height, width)``."""
        return tuple(self._shape)

    @property
    def width(self) -> int:
        """Grid width in pixels."""
        return int(self.shape[-1])

    @property
    def height(self) -> int:
        """Grid height in pixels."""
        return int(self.shape[-2])

    @property
    def dtype(self) -> np.dtype:
        """Numpy dtype read by this reader."""
        return np.dtype(self._dtype)

    @property
    def dims(self) -> tuple[str, ...]:
        """Dimension names, as `geoarray.GeoArray.dims`.

        ``("time", "band", "y", "x")[-ndim:]`` — i.e. ``("y", "x")`` for 2-D,
        ``("band", "y", "x")`` for 3-D and ``("time", "band", "y", "x")``
        for 4-D readers.

        Raises:
            ValueError: If the reader shape is not 2-D, 3-D or 4-D.
        """
        ndim = len(self.shape)
        if ndim not in (2, 3, 4):
            raise ValueError(
                f"ProductReader expects a 2d-4d array shape; got {self.shape}."
            )
        return ("time", "band", "y", "x")[-ndim:]

    @property
    def bands(self) -> tuple[str, ...]:
        """Band names in array order."""
        return tuple(self._bands)

    @property
    def fill_value(self) -> Any:
        """Fill value of nodata and out-of-grid pixels."""
        return self._fill_value

    @property
    def track(self) -> Track:
        """Reader track classification."""
        return self._track

    @property
    def attrs(self) -> FrozenAttrs:
        """Metadata every read carries: ``band_names`` plus :meth:`_band_attrs`."""
        # `band_names` is the canonical key every band resolver reads first
        # (see geoops._src.bands.DEFAULT_BAND_KEYS).
        return FrozenAttrs({"band_names": self.bands, **self._band_attrs()})

    def _band_attrs(self) -> dict[str, Any]:
        """Extra ``attrs`` every read carries (per-band ``units``, ``wavelengths``, …).

        ``band_names`` is always set; readers override this to add more.
        """
        return {}

    # -- RasterSource reads ------------------------------------------------

    def read_window(self, window: PixelWindow, *, boundless: bool = True) -> GeoArray:
        """Read a pixel window as a ``GeoArray`` on the window's grid.

        Args:
            window: The pixels to read; may extend past the grid.
            boundless: ``True`` returns exactly ``window.shape``, filled with
                `fill_value` outside the grid; ``False`` returns only the
                part inside.

        Returns:
            ``(..., h, w)`` carrier whose ``attrs`` are :attr:`attrs`.

        Raises:
            TypeError: ``window`` is not a `geoarray.PixelWindow` (convert a
                rasterio ``Window`` with
                ``PixelWindow(col_off=..., row_off=..., width=..., height=...)``).
            ValueError: ``boundless=False`` and the window misses the grid.
        """
        if not isinstance(window, PixelWindow):
            raise TypeError(
                f"{type(self).__name__}.read_window takes a geoarray.PixelWindow, "
                f"got {type(window).__name__}; build one with PixelWindow("
                "col_off=..., row_off=..., width=..., height=...)"
            )
        if not boundless:
            whole = PixelWindow(
                col_off=0, row_off=0, width=self.width, height=self.height
            )
            inside = window.intersection(whole)
            if inside is None:
                raise ValueError(
                    f"window {window} does not overlap the grid "
                    f"{(self.height, self.width)}"
                )
            window = inside
        values = self._read_window(window)
        return GeoArray(
            values,
            self.grid.crop(window).transform,
            self.grid.crs,
            fill_value=self.fill_value,
            attrs=self.attrs,
            dims=DIMS[-np.ndim(values) :],
        )

    def load(self) -> GeoArray:
        """Load the reader's full extent into a ``GeoArray`` of `shape`."""
        return self.read_window(
            PixelWindow(col_off=0, row_off=0, width=self.width, height=self.height)
        )

    def read_from_bounds(
        self,
        bounds: tuple[float, float, float, float],
        *,
        boundless: bool = True,
        crs_bounds: Any = None,
    ) -> GeoArray:
        """Read map-coordinate bounds as a ``GeoArray``.

        georeader's :func:`georeader.read.read_from_bounds` computes the
        window (outward rounding, CRS transform of ``bounds``), so the
        window maths match georeader exactly; the pixels come from
        :meth:`read_window`.
        """
        from geoarray.interop.georeader import from_georeader, source_to_georeader
        from georeader import read

        return from_georeader(
            read.read_from_bounds(
                source_to_georeader(self),
                bounds,
                crs_bounds=crs_bounds,
                boundless=boundless,
            )
        )

    def read_from_center_coords(
        self,
        x: float,
        y: float,
        width: int,
        height: int,
        *,
        boundless: bool = True,
        crs_center_coords: Any = None,
    ) -> GeoArray:
        """Read a ``height x width`` window centered on map coordinates.

        georeader's :func:`georeader.read.read_from_center_coords` places
        the window, so the placement matches georeader exactly; the pixels
        come from :meth:`read_window`.
        """
        from geoarray.interop.georeader import from_georeader, source_to_georeader
        from georeader import read

        return from_georeader(
            read.read_from_center_coords(
                source_to_georeader(self),
                (x, y),
                (height, width),
                crs_center_coords=crs_center_coords,
                boundless=boundless,
            )
        )

    def _read_boundless(
        self, window: PixelWindow, read: Callable[[slice, slice], np.ndarray]
    ) -> np.ndarray:
        """Read ``window`` through ``read``, padding outside the grid with the fill.

        The shared body of a ``_read_window``: allocates the output filled
        with `fill_value`, asks ``read(rows, cols)`` only for the part of
        the window inside the grid (returning ``(..., h, w)``), and pastes
        it in place. Windows partly or wholly outside the grid (boundless
        reads) are handled here once for every reader.

        Args:
            window: Pixel window, possibly extending past the grid.
            read: Reads the in-grid ``rows`` / ``cols`` slices of every band.

        Returns:
            ``(..., window.height, window.width)`` array of ``self.dtype``.
        """
        row0, col0 = window.row_off, window.col_off
        n_rows, n_cols = window.height, window.width
        out = np.full(
            (*self.shape[:-2], n_rows, n_cols), self.fill_value, dtype=self.dtype
        )
        r_start, r_stop = max(row0, 0), min(row0 + n_rows, self.height)
        c_start, c_stop = max(col0, 0), min(col0 + n_cols, self.width)
        if r_start >= r_stop or c_start >= c_stop:
            return out
        out[..., r_start - row0 : r_stop - row0, c_start - col0 : c_stop - col0] = read(
            slice(r_start, r_stop), slice(c_start, c_stop)
        )
        return out


def resolve_fill_value(
    fill_value: float | None, dtype: np.dtype, *, owner: str = "reader"
) -> float | int | bool:
    """A fill value exactly representable in ``dtype``.

    ``None`` defaults to ``NaN`` for inexact dtypes and ``0`` otherwise.
    An explicit value that would change when cast to ``dtype`` (``NaN``,
    fractional or out-of-range values for integer data) raises instead of
    silently padding with a different value than ``fill_value`` reports.

    Args:
        fill_value: Requested fill, or ``None`` for the default.
        dtype: The reader's data dtype.
        owner: Name used in the error message.

    Raises:
        ValueError: ``fill_value`` is not representable in ``dtype``.

    Examples:
        >>> resolve_fill_value(None, np.dtype("uint8"))
        0
        >>> resolve_fill_value(255, np.dtype("uint8"))
        255
    """
    if fill_value is None:
        return np.nan if np.issubdtype(dtype, np.inexact) else dtype.type(0).item()
    if np.issubdtype(dtype, np.inexact):
        return fill_value
    try:
        with np.errstate(invalid="ignore", over="ignore"):
            cast = dtype.type(fill_value)
        ok = bool(np.isfinite(fill_value)) and cast == fill_value
    except (OverflowError, TypeError, ValueError):
        ok = False
    if not ok:
        raise ValueError(
            f"{owner}: fill_value={fill_value!r} is not "
            f"representable in data dtype {dtype}; pass a value that "
            "survives the cast (e.g. 0) or use floating-point data."
        )
    return cast.item()
