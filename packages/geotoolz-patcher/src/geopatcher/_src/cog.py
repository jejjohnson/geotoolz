"""Async COG reading on the `ObstoreCogField` engine.

`AsyncCogReader` is the ``await``-native face of the same reader that
backs `ObstoreCogField`: one async-geotiff handle, one pooled obstore
client, the same tile de-duplication and concurrent fetch groups, the
same nodata / mask / snapping semantics. It exposes georeader's
`GeoData` metadata surface (``crs``, ``transform``, ``shape``,
``bounds``, ``res``, ``footprint`` …) so georeader's window helpers
work on it, lazy window views (``read_from_window`` does no I/O) and an
async ``load``.

The module-level coroutines mirror ``georeader.read``. Each one works
out which source pixels the request needs, fetches them asynchronously,
and hands the in-memory `GeoTensor` to the matching sync
``georeader.read`` function — so results match the sync path pixel for
pixel, while the event loop is free during I/O. Reprojections fetch the
same source chunk georeader reads from a lazy reader (the destination
footprint plus a 3-pixel margin, boundless); destinations that miss the
image fetch nothing. A destination far coarser than the source (a
low-zoom tile over a high-resolution scene) still spans a large source
window — read those from `AsyncCogReader.reader_overview`.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import numpy as np
import rasterio.windows
from georeader import read as _read, window_utils
from georeader.geotensor import GeoTensor
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_bounds as _transform_from_bounds
from rasterio.windows import Window

from geopatcher._src.fields.obstore_cog import (
    ObstoreCogField,
    _build_domain,
    _snap_window,
)


if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry


# Margin (source pixels) fetched around a reprojection footprint — the pad
# georeader's own ``read_reproject`` reads around it for its kernels.
_REPROJECT_PAD = 3
# Half the width of the EPSG:3857 square, in metres (the XYZ tile grid).
_WEB_MERCATOR_HALF = 20037508.342789244


def _window(col_off: float, row_off: float, width: float, height: float) -> Window:
    """``rasterio.windows.Window`` built by keyword (its attrs init)."""
    return Window(col_off=col_off, row_off=row_off, width=width, height=height)


@dataclasses.dataclass(eq=False)
class AsyncCogReader:
    """``await``-native COG reader over an `ObstoreCogField`.

    Open with :meth:`open`; metadata is then sync and free, reads are
    coroutines::

        reader = await AsyncCogReader.open("s3://bucket/scene.tif")
        view = reader.read_from_window(Window(0, 0, 512, 512))  # no I/O
        chip = await view.load()                                 # GeoTensor
        chips = await reader.load_many(windows)  # shared tiles fetched once

    A view shares the parsed header with its parent and only carries a
    different ``window_focus``. Out-of-image pixels, and pixels the
    COG's internal mask marks invalid, read as ``fill_value_default``.
    The reader pickles (by URL, through the field).

    Args:
        field: The `ObstoreCogField` doing the reading.
        window_focus: The pixel window this view covers, in the field's
            pixel grid (``None`` = the whole image). Snapped outward to
            whole pixels.
    """

    field: ObstoreCogField
    window_focus: Window | None = None

    def __post_init__(self) -> None:
        if self.window_focus is not None:
            self.window_focus = _window(*_snap_window(self.window_focus))

    @classmethod
    async def open(
        cls,
        url: str,
        *,
        storage_options: dict[str, Any] | None = None,
        ifd_index: int = 0,
        store: Any = None,
        path: str | None = None,
        timeout: float | None = 120.0,
    ) -> AsyncCogReader:
        """Fetch and parse the COG header; return a reader over the whole image.

        Arguments are those of `ObstoreCogField.from_url` (pooled store
        from ``url``, or an explicit ``store`` + ``path``; ``ifd_index``
        ``1+`` opens an overview).

        Raises:
            ImportError: ``[obstore-cog]`` extra missing.
            ValueError: See `ObstoreCogField.from_url`.
            TimeoutError: Opening took longer than ``timeout`` seconds.
        """
        field = await ObstoreCogField.afrom_url(
            url,
            storage_options=storage_options,
            ifd_index=ifd_index,
            store=store,
            path=path,
            timeout=timeout,
        )
        return cls(field)

    # ------------------------------------------------------------- metadata
    @property
    def _raster_window(self) -> Window:
        _, height, width = self.field.domain.shape
        return _window(0, 0, width, height)

    @property
    def _focus(self) -> Window:
        return (
            self.window_focus if self.window_focus is not None else self._raster_window
        )

    @property
    def crs(self) -> Any:
        """CRS of the image (``None`` when it could not be built)."""
        return self.field.domain.crs

    @property
    def transform(self) -> Affine:
        """Affine transform of this view's top-left pixel."""
        focus = self._focus
        return self.field.domain.transform * Affine.translation(
            focus.col_off, focus.row_off
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(bands, height, width)`` of this view."""
        focus = self._focus
        return (int(self.field.domain.shape[0]), int(focus.height), int(focus.width))

    @property
    def width(self) -> int:
        return self.shape[2]

    @property
    def height(self) -> int:
        return self.shape[1]

    @property
    def count(self) -> int:
        return self.shape[0]

    @property
    def dims(self) -> list[str]:
        return ["band", "y", "x"]

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.field.dtype)

    @property
    def fill_value_default(self) -> float | int:
        """Nodata, else ``0``: the value of out-of-image and masked pixels."""
        return self.field.fill_value_default

    @property
    def res(self) -> tuple[float, float]:
        return self.field.domain.res

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(minx, miny, maxx, maxy)`` of this view in its CRS."""
        return window_utils.window_bounds(self._focus, self.field.domain.transform)

    def footprint(self, crs: Any = None) -> BaseGeometry:
        """This view's outline as a polygon, in ``crs`` (default: its own)."""
        polygon = window_utils.window_polygon(self._focus, self.field.domain.transform)
        if crs is None or window_utils.compare_crs(crs, self.crs):
            return polygon
        return window_utils.polygon_to_crs(polygon, self.crs, crs)

    # ------------------------------------------------------------- reading
    def read_from_window(
        self, window: Window, boundless: bool = True
    ) -> AsyncCogReader:
        """A view of ``window`` (relative to this view). **Sync, no I/O.**

        ``boundless=False`` clips the window to the image and raises
        ``rasterio.windows.WindowError`` when they are disjoint;
        ``boundless=True`` lets it extend past the image (filled on load).
        """
        focus = self._focus
        absolute = _window(
            window.col_off + focus.col_off,
            window.row_off + focus.row_off,
            window.width,
            window.height,
        )
        if not boundless:
            absolute = rasterio.windows.intersection(
                _window(*_snap_window(absolute)), self._raster_window
            )
        return dataclasses.replace(self, window_focus=absolute)

    async def load(self, boundless: bool = True) -> GeoTensor:
        """Read this view as a `GeoTensor`.

        ``boundless=True`` returns the full view, out-of-image pixels
        filled; ``boundless=False`` returns its intersection with the
        image (``rasterio.windows.WindowError`` when disjoint).
        """
        focus = self._focus
        if not boundless:
            focus = rasterio.windows.intersection(focus, self._raster_window)
        return await self.field.aselect(focus)

    async def load_many(self, windows: list[Window]) -> list[GeoTensor]:
        """Read many windows (relative to this view) in one batch.

        Tiles shared between windows are fetched and decoded once (see
        `ObstoreCogField.aselect_many`); windows may extend past the image.
        """
        focus = self._focus
        absolute = [
            _window(
                w.col_off + focus.col_off, w.row_off + focus.row_off, w.width, w.height
            )
            for w in windows
        ]
        return await self.field.aselect_many(absolute)

    # ------------------------------------------------------------- overviews
    def overviews(self) -> list[int]:
        """Decimation factors of the COG's overviews (e.g. ``[2, 4, 8]``).

        Empty for a reader already opened on an overview.
        """
        level = self.field.level
        width = int(self.field.domain.shape[2])
        return [round(width / int(ov.width)) for ov in getattr(level, "overviews", [])]

    def reader_overview(self, overview: int) -> AsyncCogReader:
        """A whole-image reader on overview ``overview`` (0-based). **No I/O.**

        Shares the parsed header; pixel windows then refer to the
        overview's grid.

        Raises:
            ValueError: This reader is itself an overview, or ``overview``
                is out of range.
        """
        geotiff = self.field.level
        overviews = list(getattr(geotiff, "overviews", []))
        if not 0 <= overview < len(overviews):
            raise ValueError(
                f"AsyncCogReader: overview {overview} out of range for "
                f"{self.field.url!r} ({len(overviews)} overview(s) on this reader)."
            )
        level = overviews[overview]
        size = (int(level.width), int(level.height))
        ifd_index = next(
            i
            for i, ifd in enumerate(geotiff.tiff.ifds)
            if i > 0
            and (int(ifd.image_width), int(ifd.image_height)) == size
            and not int(getattr(ifd, "new_subfile_type", None) or 0) & 4
        )
        field = dataclasses.replace(
            self.field,
            level=level,
            ifd=level.ifd,
            domain=_build_domain(level, geotiff, url=self.field.url),
            ifd_index=ifd_index,
        )
        return AsyncCogReader(field)


# ----------------------------------------------------------------- read_*
async def read_from_window(
    reader: AsyncCogReader, window: Window, boundless: bool = True
) -> GeoTensor:
    """Read ``window`` of ``reader`` (see `AsyncCogReader.read_from_window`)."""
    return await reader.read_from_window(window, boundless=boundless).load(
        boundless=boundless
    )


async def read_from_bounds(
    reader: AsyncCogReader,
    bounds: tuple[float, float, float, float],
    crs_bounds: Any = None,
    pad_add: tuple[int, int] = (0, 0),
    boundless: bool = True,
) -> GeoTensor:
    """Async `georeader.read.read_from_bounds`: the pixels covering ``bounds``."""
    window = _read.window_from_bounds(reader, bounds, crs_bounds)
    if any(p > 0 for p in pad_add):
        window = window_utils.pad_window(window, pad_add)
    return await read_from_window(
        reader, window_utils.round_outer_window(window), boundless=boundless
    )


async def read_from_polygon(
    reader: AsyncCogReader,
    polygon: BaseGeometry,
    crs_polygon: Any = None,
    pad_add: tuple[int, int] = (0, 0),
    boundless: bool = True,
    window_surrounding: bool = False,
) -> GeoTensor:
    """Async `georeader.read.read_from_polygon`: the pixels covering ``polygon``."""
    window = _read.window_from_polygon(
        reader, polygon, crs_polygon, window_surrounding=window_surrounding
    )
    if any(p > 0 for p in pad_add):
        window = window_utils.pad_window(window, pad_add)
    return await read_from_window(
        reader, window_utils.round_outer_window(window), boundless=boundless
    )


async def read_from_center_coords(
    reader: AsyncCogReader,
    center_coords: tuple[float, float],
    shape: tuple[int, int],
    crs_center_coords: Any = None,
    boundless: bool = True,
) -> GeoTensor:
    """Async `georeader.read.read_from_center_coords`: a ``shape`` chip at a point."""
    window = _read.window_from_center_coords(
        reader, center_coords, shape, crs_center_coords
    )
    return await read_from_window(reader, window, boundless=boundless)


async def _load_footprint(
    reader: AsyncCogReader, polygon: BaseGeometry, crs_polygon: Any
) -> GeoTensor:
    """Fetch the source chunk georeader warps for ``polygon``.

    The window of ``polygon`` plus `_REPROJECT_PAD` pixels, rounded
    outward and read boundless — exactly what
    ``georeader.read.read_reproject`` reads from a lazy reader. GDAL
    derives its resampling scale from the source extent, so a different
    chunk (even one only trimmed of fill) changes interpolated values.
    Like georeader, a polygon that misses the image fetches nothing: a
    1x1 fill chunk just off the image carries the metadata for the
    all-nodata result.
    """
    if not reader.footprint(crs=crs_polygon).intersects(polygon):
        return await reader.read_from_window(_window(-1, -1, 1, 1)).load()
    window = _read.window_from_polygon(reader, polygon, crs_polygon)
    window = window_utils.round_outer_window(
        window_utils.pad_window(window, (_REPROJECT_PAD, _REPROJECT_PAD))
    )
    return await reader.read_from_window(window).load()


async def read_reproject(
    reader: AsyncCogReader,
    dst_crs: Any = None,
    bounds: tuple[float, float, float, float] | None = None,
    resolution_dst_crs: float | tuple[float, float] | None = None,
    dst_transform: Affine | None = None,
    window_out: Window | None = None,
    resampling: Resampling = Resampling.cubic_spline,
    dtype_dst: Any = None,
    dst_nodata: Any = None,
) -> GeoTensor:
    """Async `georeader.read.read_reproject` onto a destination grid.

    The grid is ``dst_transform`` + ``window_out``, or derived from
    ``bounds`` (+ ``resolution_dst_crs``) exactly as georeader does;
    only the source pixels under it are fetched.
    """
    dst_transform = window_utils.figure_out_transform(
        transform=dst_transform, bounds=bounds, resolution_dst=resolution_dst_crs
    )
    if window_out is None:
        if bounds is None:
            raise ValueError(
                "read_reproject: pass `window_out` or `bounds` to size the output."
            )
        window_out = rasterio.windows.from_bounds(
            *bounds, transform=dst_transform
        ).round_lengths(op="ceil", pixel_precision=window_utils.PIXEL_PRECISION)
    if dst_crs is None:
        dst_crs = reader.crs
    chunk = await _load_footprint(
        reader, window_utils.window_polygon(window_out, dst_transform), dst_crs
    )
    return _read.read_reproject(
        chunk,
        dst_crs=dst_crs,
        dst_transform=dst_transform,
        window_out=window_out,
        resampling=resampling,
        dtype_dst=dtype_dst,
        dst_nodata=dst_nodata,
    )


async def read_reproject_like(
    reader: AsyncCogReader,
    data_like: Any,
    resolution_dst: float | tuple[float, float] | None = None,
    resampling: Resampling = Resampling.cubic_spline,
    dtype_dst: Any = None,
    dst_nodata: Any = None,
) -> GeoTensor:
    """Async `georeader.read.read_reproject_like`: onto ``data_like``'s grid.

    With ``resolution_dst`` the grid keeps ``data_like``'s extent at the
    new pixel size.
    """
    height, width = data_like.shape[-2:]
    if resolution_dst is not None:
        res_y, res_x = (
            (resolution_dst, resolution_dst)
            if isinstance(resolution_dst, int | float)
            else resolution_dst
        )
        like_y, like_x = data_like.res
        height = round(height * like_y / res_y)
        width = round(width * like_x / res_x)
    return await read_reproject(
        reader,
        dst_crs=data_like.crs,
        dst_transform=data_like.transform,
        resolution_dst_crs=resolution_dst,
        window_out=_window(0, 0, width, height),
        resampling=resampling,
        dtype_dst=dtype_dst,
        dst_nodata=dst_nodata,
    )


async def read_to_crs(
    reader: AsyncCogReader,
    dst_crs: Any,
    resampling: Resampling = Resampling.cubic_spline,
    resolution_dst_crs: float | tuple[float, float] | None = None,
) -> GeoTensor:
    """Async `georeader.read.read_to_crs`: the whole view, reprojected.

    Already in ``dst_crs``: the view is loaded as is.
    """
    if window_utils.compare_crs(reader.crs, dst_crs):
        return await reader.load()
    window_out, dst_transform = _read.calculate_transform_window(
        reader, dst_crs, resolution_dst_crs
    )
    return await read_reproject(
        reader,
        dst_crs=dst_crs,
        dst_transform=dst_transform,
        window_out=window_out,
        resampling=resampling,
    )


async def read_from_tile(
    reader: AsyncCogReader,
    x: int,
    y: int,
    z: int,
    out_shape: tuple[int, int] = (256, 256),
    resampling: Resampling = Resampling.cubic_spline,
) -> GeoTensor:
    """The XYZ (slippy-map) tile ``(x, y, z)`` in EPSG:3857, ``out_shape`` pixels.

    Tiles off the image come back filled with ``fill_value_default``
    without any fetch. For tiles much coarser than the source pixels,
    pass a `AsyncCogReader.reader_overview` of matching resolution.
    """
    span = 2 * _WEB_MERCATOR_HALF / 2**z
    left = -_WEB_MERCATOR_HALF + x * span
    top = _WEB_MERCATOR_HALF - y * span
    height, width = out_shape
    return await read_reproject(
        reader,
        dst_crs="EPSG:3857",
        dst_transform=_transform_from_bounds(
            left, top - span, left + span, top, width, height
        ),
        window_out=_window(0, 0, width, height),
        resampling=resampling,
    )
