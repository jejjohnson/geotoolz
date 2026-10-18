"""Raster `Field` adapter — bridges a `geoarray.RasterSource` to the Patcher.

A `RasterSource` (a `geoarray.GeoArray`, a lazy file or product reader, a
COG) already knows its grid and reads windows as `GeoArray`. This adapter
renames ``read_window`` to ``select`` and rebuilds merged outputs on the
source grid (``with_data``), with no behaviour of its own.

Users wrap once at the boundary::

    field = RasterField(scene)                  # a GeoArray, or a lazy reader
    for patch in patcher.split(field):
        ...                                     # patch.data: GeoArray

A georeader reader crosses with `geoarray.interop.georeader`::

    field = RasterField(source_from_georeader(RasterioReader("scene.tif")))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from geoarray import (
    PER_BAND_KEYS,
    AsyncRasterSource,
    FrozenAttrs,
    GeoArray,
    Grid,
    RasterSource,
)

from geopatcher._src.carrier import (
    as_pixel_window,
    from_wire,
    output_fill,
    reject_georeader,
    to_wire,
)


def _require_source(reader: Any, where: str, protocol: type) -> None:
    """Refuse georeader objects and anything that is not a ``protocol``."""
    reject_georeader(reader, where)
    if not isinstance(reader, protocol):
        raise TypeError(
            f"{where} takes a geoarray.{protocol.__name__} (a GeoArray or a lazy "
            f"reader with grid / shape / read_window), got {type(reader).__name__}"
        )


@dataclass(eq=False)
class RasterField:
    """A `geoarray.RasterSource` (a `GeoArray`, a lazy reader, …) as a `Field`.

    The source doubles as the `RasterDomain`: its ``grid`` and ``shape`` are
    metadata, known before any pixel is read. Each ``select`` reads one
    window through ``source.read_window`` — a lazy source reads window by
    window, a `GeoArray` slices its memory.

    Args:
        reader: Any `geoarray.RasterSource`. A georeader ``GeoTensor`` or
            reader is refused; convert it with
            ``geoarray.interop.georeader.from_georeader`` /
            ``source_from_georeader``.

    Raises:
        TypeError: ``reader`` is a georeader object or not a `RasterSource`.

    Example:
        >>> import numpy as np
        >>> from geoarray import GeoArray
        >>> scene = GeoArray(
        ...     np.zeros((4, 512, 512), np.float32),        # (C, H, W) float32
        ...     (10.0, 0.0, 500000.0, 0.0, -10.0, 4600000.0),
        ...     "EPSG:32630",
        ...     fill_value=np.nan,
        ... )
        >>> field = RasterField(scene)
        >>> field.domain.grid.shape
        (512, 512)
    """

    reader: RasterSource

    def __post_init__(self) -> None:
        _require_source(self.reader, "RasterField", RasterSource)

    @property
    def domain(self) -> RasterSource:
        return self.reader

    def select(self, window: Any) -> GeoArray:
        """Read ``window`` (boundless) as a `GeoArray` on the window's grid.

        Args:
            window: A ``rasterio.windows.Window`` (what the geometries
                produce) or a `geoarray.PixelWindow`.

        Returns:
            ``(..., window.height, window.width)``; pixels past the grid
            hold the source's ``fill_value`` (a source without one raises
            when the window leaves the grid).
        """
        return self.reader.read_window(as_pixel_window(window), boundless=True)

    def with_data(self, array: Any) -> GeoArray:
        """``array`` (``(..., H, W)`` on the source grid) as a `GeoArray`."""
        return _rewrap(self.reader, array, self.reader.grid)

    def __getstate__(self) -> dict[str, Any]:
        return _pack_reader_state(self.__dict__)

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(_unpack_reader_state(state))


@dataclass(eq=False)
class AsyncRasterField:
    """Async mirror of `RasterField` over a `geoarray.AsyncRasterSource`.

    `select` is a coroutine (``reader.aread_window``); otherwise the
    surface is identical.

    Args:
        reader: Any `geoarray.AsyncRasterSource` (``geocloud.cog.CogSource``,
            ``geocloud.cog.AsyncCogReader``, …).

    Raises:
        TypeError: ``reader`` is a georeader object or not an
            `AsyncRasterSource`.
    """

    reader: AsyncRasterSource

    def __post_init__(self) -> None:
        _require_source(self.reader, "AsyncRasterField", AsyncRasterSource)

    @property
    def domain(self) -> AsyncRasterSource:
        return self.reader

    async def select(self, window: Any) -> GeoArray:
        return await self.reader.aread_window(as_pixel_window(window), boundless=True)

    async def aselect(self, window: Any) -> GeoArray:
        return await self.select(window)

    def with_data(self, array: Any) -> GeoArray:
        return _rewrap(self.reader, array, self.reader.grid)


def _band_count(shape: tuple[int, ...]) -> int:
    """Size of the band axis (``-3``) of a carrier of ``shape``; 1 when absent."""
    return int(shape[-3]) if len(shape) >= 3 else 1


def _rewrap(source: Any, array: Any, grid: Grid) -> GeoArray:
    """Wrap ``array`` as a `GeoArray` on ``grid``, described by ``source``.

    Used by every raster ``with_data`` and by `on_domain_grid` (the pipekit
    ``MergePatches`` operator), so a merged output keeps the source's
    georeferencing and follows the carrier contract:

    * ``attrs`` are the source's; when the band axis changed size (four
      bands merged into one index), the per-band keys
      (`geoarray.PER_BAND_KEYS`) are dropped instead of describing bands
      that are gone.
    * ``fill_value`` is the source's nodata while the output still carries
      the source's values. A float output on an integer source, or on a
      different band count, is a new quantity whose missing cells are NaN,
      so its fill is NaN.
    * The fill is always one the output's dtype holds (`output_fill`): a
      ``uint8`` output of a source whose fill is ``-9999.0`` declares no
      nodata (``None``) rather than a value it cannot store.

    Raises:
        TypeError: ``array`` is a georeader ``GeoTensor``.
    """
    reject_georeader(array, "with_data")
    values = np.asarray(array)
    source_shape = tuple(getattr(source, "shape", None) or values.shape)
    same_bands = _band_count(source_shape) == _band_count(values.shape)
    attrs = FrozenAttrs(getattr(source, "attrs", None) or {})
    if not same_bands:
        attrs = attrs.drop(*PER_BAND_KEYS)
    fill = getattr(source, "fill_value", None)
    source_dtype = getattr(source, "dtype", None)
    source_float = source_dtype is not None and np.issubdtype(
        np.dtype(source_dtype), np.floating
    )
    if np.issubdtype(values.dtype, np.floating) and not (source_float and same_bands):
        fill = np.nan
    fill = output_fill(fill, values.dtype)
    return GeoArray(values, grid.transform, grid.crs, fill_value=fill, attrs=attrs)


def _pack_reader_state(state: dict[str, Any]) -> dict[str, Any]:
    """Copy of a field's ``__dict__`` whose `GeoArray` reader pickles compactly.

    A `GeoArray` reader travels in the wire form of `geopatcher._src.carrier`
    (pixels plus the grid config, fill and attrs), rebuilt through the
    constructor in the worker, so the carrier compares equal to one built
    there. Lazy readers pickle as they define (by path or URL).
    """
    state = dict(state)
    if "reader" in state:
        state["reader"] = to_wire(state["reader"])
    return state


def _unpack_reader_state(state: dict[str, Any]) -> dict[str, Any]:
    """Inverse of `_pack_reader_state`."""
    if "reader" in state:
        state = {**state, "reader": from_wire(state["reader"])}
    return state
