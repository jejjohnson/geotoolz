"""`RasterSource` / `AsyncRasterSource`: rasters that read pixels on demand.

A lazy reader (a file, a COG, a product on disk) knows its grid before it
reads a pixel. Code that only needs a window of it, such as a patcher or a
crop operator, takes a `RasterSource` and asks for exactly those pixels; a
`GeoArray` is itself a `RasterSource` whose pixels are already in memory.

The protocols describe behaviour only: geoarray ships no reader. Readers
live in the packages that know the formats (geotoolz-products,
geotoolz-cloud), and `geoarray.interop.georeader` wraps georeader's.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


if TYPE_CHECKING:
    from geoarray._src.array import GeoArray
    from geoarray._src.grid import Grid, PixelWindow


__all__ = ["AsyncRasterSource", "RasterSource"]


@runtime_checkable
class RasterSource(Protocol):
    """A raster whose grid is known and whose pixels are read on demand.

    Implemented by every lazy reader of the stack, and by `GeoArray`
    itself (``load`` returns the carrier, ``read_window`` slices it).
    ``isinstance(x, RasterSource)`` checks that the members exist.

    Attributes:
        grid: The pixel grid: transform, CRS and ``(H, W)``.
        shape: ``(..., H, W)``, the shape `load` returns.
        dtype: The dtype of the pixels.
        fill_value: The nodata value of the pixels read (``None`` for none).
        attrs: Metadata every read carries (``band_names``, …).

    Example:
        A function that works on any raster, lazy or loaded::

            def corner(source: RasterSource) -> GeoArray:
                window = PixelWindow(col_off=0, row_off=0, width=256, height=256)
                return source.read_window(window)  # (..., 256, 256)
    """

    @property
    def grid(self) -> Grid: ...

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> Any: ...

    @property
    def fill_value(self) -> Any: ...

    @property
    def attrs(self) -> Mapping[str, Any]: ...

    def read_window(self, window: PixelWindow, *, boundless: bool = True) -> GeoArray:
        """The pixels of ``window`` as a `GeoArray` on the window's grid.

        ``boundless=True`` returns exactly ``window.shape``, filled with
        ``fill_value`` where the window leaves the grid; ``False`` returns
        only the part inside.
        """
        ...

    def load(self) -> GeoArray:
        """Every pixel, as a `GeoArray` of `shape`."""
        ...


@runtime_checkable
class AsyncRasterSource(Protocol):
    """`RasterSource` with coroutine reads, for sources behind a network.

    The metadata members are the same and synchronous; ``aread_window`` and
    ``aload`` are the awaitable reads. An object may implement both
    protocols.
    """

    @property
    def grid(self) -> Grid: ...

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> Any: ...

    @property
    def fill_value(self) -> Any: ...

    @property
    def attrs(self) -> Mapping[str, Any]: ...

    async def aread_window(
        self, window: PixelWindow, *, boundless: bool = True
    ) -> GeoArray:
        """Awaitable `RasterSource.read_window`."""
        ...

    async def aload(self) -> GeoArray:
        """Awaitable `RasterSource.load`."""
        ...
