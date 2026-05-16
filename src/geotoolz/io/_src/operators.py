"""I/O operators wrapping :mod:`georeader` read and write primitives.

This module provides carrier-aware :class:`Operator` wrappers around the
file/cloud IO primitives in :mod:`georeader.read` and
:mod:`georeader.save`. Two families:

* **Source operators** (subclasses of :class:`SourceOperator`) take a
  raster source (path / :class:`RasterioReader` / STAC asset / EE image)
  and produce an in-memory :class:`georeader.geotensor.GeoTensor`. They
  override ``_apply`` with no positional input and are valid as the
  first step of a :class:`~geotoolz.core.Sequential`.
* **Sink operators** (subclasses of :class:`SinkOperator`) consume a
  ``GeoTensor`` and write it to disk / object storage. They are marked
  ``_terminal = True`` so :class:`~geotoolz.core.Sequential` only
  accepts them as the last step.

All public IO operators set ``forbid_in_yaml = True`` — their
``get_config()`` returns a debug-friendly dict but cannot round-trip
through YAML cleanly (file paths, STAC items, EE asset IDs, reference
grids, and ``shapely`` geometries are runtime references).

See `geotoolz` design report §4 (two-tier model) and §6.2 (round-trip
discipline).
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from georeader import read, save
from georeader.geotensor import GeoTensor
from georeader.rasterio_reader import RasterioReader
from rasterio.errors import RasterioIOError
from rasterio.io import DatasetReaderBase
from rasterio.windows import Window
from shapely.geometry import MultiPolygon, Polygon, box

from geotoolz.core import Operator


Source = str | PathLike[str] | Any
Bounds = tuple[float, float, float, float]
Resolution = float | tuple[float, float]


class GeoToolzIOError(RuntimeError):
    """Raised when a geotoolz I/O operator cannot read or write data."""


class SourceOperator(Operator):
    """Operator that produces a :class:`GeoTensor` without an input carrier.

    Source operators are valid as the *first* step of a
    :class:`~geotoolz.core.Sequential` because
    :class:`~geotoolz.core.Sequential` calls ``op()`` (no input) when the
    pipeline is invoked without a carrier.
    """

    forbid_in_yaml: ClassVar[bool] = True


class SinkOperator(Operator):
    """Terminal operator that consumes a :class:`GeoTensor` and writes it.

    Sink operators are marked ``_terminal = True`` so
    :class:`~geotoolz.core.Sequential` only accepts them as the *last*
    step (anywhere else they would break the GeoTensor → next op
    contract by returning ``None``).
    """

    _terminal: ClassVar[bool] = True
    forbid_in_yaml: ClassVar[bool] = True


def _window_config(window: Window) -> tuple[float, float, float, float]:
    return (window.col_off, window.row_off, window.width, window.height)


def _coerce_window(window: Window | tuple[float, float, float, float]) -> Window:
    if isinstance(window, Window):
        return window
    col_off, row_off, width, height = window
    return Window(col_off=col_off, row_off=row_off, width=width, height=height)


def _coerce_source(src: Source, indexes: list[int] | None = None) -> Any:
    """Normalise a user-supplied source into something georeader can read.

    Accepts a path-like, an existing :class:`RasterioReader`, an open
    :class:`rasterio.io.DatasetReaderBase`, or any object implementing the
    ``georeader`` ``GeoData`` protocol. When ``indexes`` is given, paths
    and readers are reconfigured to read only those 1-indexed bands;
    arbitrary objects without an explicit ``indexes`` are returned as-is.
    """
    if isinstance(src, str | PathLike):
        return RasterioReader(str(src), indexes=indexes)

    if isinstance(src, RasterioReader):
        reader = src.copy()
        if indexes is not None:
            reader.set_indexes(indexes, relative=False)
        return reader

    if indexes is not None and isinstance(src, DatasetReaderBase):
        return RasterioReader(src.name, indexes=indexes)

    if indexes is not None:
        raise GeoToolzIOError("indexes are only supported for path-like sources.")

    return src


def _src_config(src: Source) -> str | Source:
    if isinstance(src, str | PathLike):
        return str(src)
    return src


def _read_error(src: Source, exc: Exception) -> GeoToolzIOError:
    return GeoToolzIOError(f"Unable to read raster source {src!r}: {exc}")


class ReadWindow(SourceOperator):
    """Read a pixel-space rectangular window from a raster source.

    Wraps :func:`georeader.read.read_from_window`. The window is specified
    in pixel coordinates of the source raster as ``(col_off, row_off,
    width, height)``.

    Args:
        src: Path, :class:`RasterioReader`, or open rasterio dataset.
            Anything implementing georeader's ``GeoData`` protocol is also
            accepted, in which case ``indexes`` must be ``None``.
        window: :class:`rasterio.windows.Window` or 4-tuple
            ``(col_off, row_off, width, height)`` in source pixel space.
        indexes: 1-indexed list of bands to read. ``None`` reads all.
        boundless: If ``True``, requests outside the source bounds are
            zero-padded; if ``False`` they raise.

    Raises:
        GeoToolzIOError: If the source cannot be opened, the window does
            not intersect the source (``boundless=False``), or ``indexes``
            is given for a non-path-like source.

    Examples:
        Read a 256 by 256 chip from the top-left corner of a GeoTIFF::

            from geotoolz import io
            from rasterio.windows import Window

            chip = io.ReadWindow(
                src="/path/to/image.tif",
                window=Window(0, 0, 256, 256),
                indexes=[1, 2, 3],
            )()
    """

    def __init__(
        self,
        *,
        src: Source,
        window: Window | tuple[float, float, float, float],
        indexes: list[int] | None = None,
        boundless: bool = True,
    ) -> None:
        self.src = src
        self.window = _coerce_window(window)
        self.indexes = indexes
        self.boundless = boundless

    def _apply(self) -> GeoTensor:
        try:
            data = _coerce_source(self.src, self.indexes)
            out = read.read_from_window(data, self.window, boundless=self.boundless)
        except (FileNotFoundError, OSError, RasterioIOError) as exc:
            raise _read_error(self.src, exc) from exc
        if out is None:
            window = _window_config(self.window)
            raise GeoToolzIOError(f"Window {window!r} does not intersect {self.src!r}.")
        return out.load()

    def get_config(self) -> dict[str, Any]:
        return {
            "src": _src_config(self.src),
            "window": _window_config(self.window),
            "indexes": self.indexes,
            "boundless": self.boundless,
        }


class ReadBounds(SourceOperator):
    """Read all pixels intersecting a geographic bounding box.

    Wraps :func:`georeader.read.read_from_bounds`. The bounds are
    converted from ``crs`` to the source CRS if needed; the returned
    :class:`GeoTensor` keeps the *source* CRS and transform.

    Args:
        src: Path, :class:`RasterioReader`, open rasterio dataset, or any
            ``GeoData``-protocol object.
        bounds: ``(xmin, ymin, xmax, ymax)`` in ``crs`` coordinates.
        crs: CRS of ``bounds`` (e.g. ``"EPSG:4326"``). ``None`` uses the
            source CRS directly.
        indexes: 1-indexed list of bands to read. ``None`` reads all.
        boundless: If ``True``, pad the read with the source's nodata
            where the bounds extend outside the raster.

    Raises:
        GeoToolzIOError: If the source cannot be opened or ``indexes`` is
            given for a non-path-like source.

    Examples:
        Read the full extent of a GeoTIFF as a single chip::

            from geotoolz import io
            from rasterio.transform import array_bounds

            bounds = array_bounds(height, width, transform)
            full = io.ReadBounds(src="/path/to/image.tif", bounds=bounds)()
    """

    def __init__(
        self,
        *,
        src: Source,
        bounds: Bounds,
        crs: str | None = None,
        indexes: list[int] | None = None,
        boundless: bool = True,
    ) -> None:
        self.src = src
        self.bounds = bounds
        self.crs = crs
        self.indexes = indexes
        self.boundless = boundless

    def _apply(self) -> GeoTensor:
        try:
            data = _coerce_source(self.src, self.indexes)
            out = read.read_from_bounds(
                data,
                self.bounds,
                crs_bounds=self.crs,
                boundless=self.boundless,
            )
        except (FileNotFoundError, OSError, RasterioIOError) as exc:
            raise _read_error(self.src, exc) from exc
        return out.load()

    def get_config(self) -> dict[str, Any]:
        return {
            "src": _src_config(self.src),
            "bounds": self.bounds,
            "crs": self.crs,
            "indexes": self.indexes,
            "boundless": self.boundless,
        }


class ReadCenterCoords(SourceOperator):
    """Read a fixed-shape window centred on map coordinates.

    Wraps :func:`georeader.read.read_from_center_coords`. Useful for
    extracting fixed-size chips around POIs (training samples, validation
    sites, ground stations).

    Args:
        src: Path, :class:`RasterioReader`, open rasterio dataset, or any
            ``GeoData``-protocol object.
        center: ``(x, y)`` in ``crs`` coordinates.
        shape: ``(height, width)`` in pixels.
        crs: CRS of ``center`` (e.g. ``"EPSG:4326"``). ``None`` uses the
            source CRS directly.
        indexes: 1-indexed list of bands to read. ``None`` reads all.
        boundless: If ``True``, requests outside the source are
            zero-padded; if ``False`` they raise.

    Raises:
        GeoToolzIOError: If the source cannot be opened or ``indexes`` is
            given for a non-path-like source.

    Examples:
        Read a 64 by 64 chip around a longitude/latitude point::

            from geotoolz import io

            chip = io.ReadCenterCoords(
                src="/path/to/image.tif",
                center=(-122.3, 37.8),
                shape=(64, 64),
                crs="EPSG:4326",
            )()
    """

    def __init__(
        self,
        *,
        src: Source,
        center: tuple[float, float],
        shape: tuple[int, int],
        crs: str | None = None,
        indexes: list[int] | None = None,
        boundless: bool = True,
    ) -> None:
        self.src = src
        self.center = center
        self.shape = shape
        self.crs = crs
        self.indexes = indexes
        self.boundless = boundless

    def _apply(self) -> GeoTensor:
        try:
            data = _coerce_source(self.src, self.indexes)
            out = read.read_from_center_coords(
                data,
                self.center,
                self.shape,
                crs_center_coords=self.crs,
                boundless=self.boundless,
            )
        except (FileNotFoundError, OSError, RasterioIOError) as exc:
            raise _read_error(self.src, exc) from exc
        return out.load()

    def get_config(self) -> dict[str, Any]:
        return {
            "src": _src_config(self.src),
            "center": self.center,
            "shape": self.shape,
            "crs": self.crs,
            "indexes": self.indexes,
            "boundless": self.boundless,
        }


class ReadTile(SourceOperator):
    """Read a single web-map tile and reproject it to a target CRS.

    Wraps :func:`georeader.read.read_from_tile`. The tile is specified as
    ``(z, x, y)`` using the standard XYZ scheme used by web tilesets
    (Slippy Map, Mapbox, Google Maps).

    Args:
        src: Path, :class:`RasterioReader`, open rasterio dataset, or any
            ``GeoData``-protocol object.
        tile: ``(z, x, y)`` web-map tile coordinates.
        indexes: 1-indexed list of bands to read. ``None`` reads all.
        dst_crs: Target CRS for the returned tile. Default
            ``"EPSG:3857"`` (Web Mercator), which is what most XYZ tile
            schemes expect.
        out_shape: Output ``(height, width)`` in pixels. Default
            ``(256, 256)`` to match standard tile sizes.
        resolution: Target pixel size in ``dst_crs`` units. ``None`` lets
            georeader compute it from ``out_shape``.

    Raises:
        GeoToolzIOError: If the source cannot be opened or the tile does
            not intersect the source.

    Examples:
        Read tile ``(z=12, x=655, y=1583)`` of a GeoTIFF::

            from geotoolz import io

            tile = io.ReadTile(
                src="/path/to/image.tif",
                tile=(12, 655, 1583),
            )()
    """

    def __init__(
        self,
        *,
        src: Source,
        tile: tuple[int, int, int],
        indexes: list[int] | None = None,
        dst_crs: str | None = "EPSG:3857",
        out_shape: tuple[int, int] | None = (256, 256),
        resolution: Resolution | None = None,
    ) -> None:
        self.src = src
        self.tile = tile
        self.indexes = indexes
        self.dst_crs = dst_crs
        self.out_shape = out_shape
        self.resolution = resolution

    def _apply(self) -> GeoTensor:
        z, x, y = self.tile
        try:
            data = _coerce_source(self.src, self.indexes)
            out = read.read_from_tile(
                data,
                x=x,
                y=y,
                z=z,
                dst_crs=self.dst_crs,
                out_shape=self.out_shape,
                resolution_dst_crs=self.resolution,
            )
        except (FileNotFoundError, OSError, RasterioIOError) as exc:
            raise _read_error(self.src, exc) from exc
        if out is None:
            raise GeoToolzIOError(
                f"Tile {self.tile!r} does not intersect {self.src!r}."
            )
        return out.load()

    def get_config(self) -> dict[str, Any]:
        return {
            "src": _src_config(self.src),
            "tile": self.tile,
            "indexes": self.indexes,
            "dst_crs": self.dst_crs,
            "out_shape": self.out_shape,
            "resolution": self.resolution,
        }


class ReadPolygon(SourceOperator):
    """Read all pixels intersecting a polygon.

    Wraps :func:`georeader.read.read_from_polygon`. The polygon is
    converted from ``crs`` to the source CRS if needed; pixels outside
    the polygon's bounding box are not loaded.

    Args:
        src: Path, :class:`RasterioReader`, open rasterio dataset, or any
            ``GeoData``-protocol object.
        polygon: :class:`shapely.geometry.Polygon` or
            :class:`shapely.geometry.MultiPolygon`.
        crs: CRS of ``polygon``. ``None`` uses the source CRS directly.
        indexes: 1-indexed list of bands to read. ``None`` reads all.
        boundless: If ``True``, the read is padded with the source's
            nodata where the polygon's bounding box extends outside the
            raster.

    Raises:
        GeoToolzIOError: If the source cannot be opened or ``indexes`` is
            given for a non-path-like source.

    Examples:
        Read all pixels intersecting an AOI polygon::

            from geotoolz import io
            from shapely.geometry import box

            aoi = box(-122.5, 37.7, -122.3, 37.9)
            patch = io.ReadPolygon(
                src="/path/to/image.tif",
                polygon=aoi,
                crs="EPSG:4326",
            )()
    """

    def __init__(
        self,
        *,
        src: Source,
        polygon: Polygon | MultiPolygon,
        crs: str | None = None,
        indexes: list[int] | None = None,
        boundless: bool = True,
    ) -> None:
        self.src = src
        self.polygon = polygon
        self.crs = crs
        self.indexes = indexes
        self.boundless = boundless

    def _apply(self) -> GeoTensor:
        try:
            data = _coerce_source(self.src, self.indexes)
            out = read.read_from_polygon(
                data,
                self.polygon,
                crs_polygon=self.crs,
                boundless=self.boundless,
            )
        except (FileNotFoundError, OSError, RasterioIOError) as exc:
            raise _read_error(self.src, exc) from exc
        return out.load()

    def get_config(self) -> dict[str, Any]:
        return {
            "src": _src_config(self.src),
            "polygon": self.polygon.wkt,
            "crs": self.crs,
            "indexes": self.indexes,
            "boundless": self.boundless,
        }


class ReadReprojectLike(SourceOperator):
    """Read a source and reproject it onto another raster's grid.

    Wraps :func:`georeader.read.read_reproject_like`. The returned
    :class:`GeoTensor` has the *same* CRS, transform, and shape as
    ``like``.

    Args:
        src: Path, :class:`RasterioReader`, open rasterio dataset, or any
            ``GeoData``-protocol object to read from.
        like: Reference :class:`GeoTensor` / :class:`RasterioReader` /
            path whose grid (CRS + transform + shape) defines the output.
        indexes: 1-indexed list of bands to read from ``src``. ``None``
            reads all.
        resolution: Optional override for the output pixel size in
            ``like``'s CRS units; ``None`` keeps ``like``'s native
            resolution.

    Raises:
        GeoToolzIOError: If either source cannot be opened or ``indexes``
            is given for a non-path-like source.

    Examples:
        Align a single-band mask onto a reference reflectance grid::

            from geotoolz import io

            aligned = io.ReadReprojectLike(
                src="/path/to/mask.tif",
                like=reference_geotensor,
                indexes=[1],
            )()
    """

    def __init__(
        self,
        *,
        src: Source,
        like: Any,
        indexes: list[int] | None = None,
        resolution: Resolution | None = None,
    ) -> None:
        self.src = src
        self.like = like
        self.indexes = indexes
        self.resolution = resolution

    def _apply(self) -> GeoTensor:
        try:
            data = _coerce_source(self.src, self.indexes)
            out = read.read_reproject_like(
                data,
                self.like,
                resolution_dst=self.resolution,
            )
        except (FileNotFoundError, OSError, RasterioIOError) as exc:
            raise _read_error(self.src, exc) from exc
        return out.load()

    def get_config(self) -> dict[str, Any]:
        return {
            "src": _src_config(self.src),
            # ``like`` is often a runtime GeoTensor / reader / array — keep
            # the bare repr for debugging and rely on ``forbid_in_yaml``
            # to signal that this config cannot round-trip cleanly.
            "like": _src_config(self.like),
            "indexes": self.indexes,
            "resolution": self.resolution,
        }


class ReadToCRS(SourceOperator):
    """Read a source and reproject it to a target CRS.

    Dispatches to :func:`georeader.read.read_to_crs` when ``bounds`` is
    ``None`` (reproject the full extent) and
    :func:`georeader.read.read_reproject` otherwise (reproject and crop
    in one pass).

    Args:
        src: Path, :class:`RasterioReader`, open rasterio dataset, or any
            ``GeoData``-protocol object.
        dst_crs: Target CRS as an authority string (e.g. ``"EPSG:4326"``)
            or anything :mod:`pyproj` can parse.
        resolution: Output pixel size in ``dst_crs`` units. ``None`` lets
            georeader compute a sensible default from the source.
        bounds: Optional ``(xmin, ymin, xmax, ymax)`` crop in ``dst_crs``
            coordinates. When provided, switches to
            :func:`read.read_reproject`.
        indexes: 1-indexed list of bands to read. ``None`` reads all.

    Raises:
        GeoToolzIOError: If the source cannot be opened or ``indexes`` is
            given for a non-path-like source.

    Examples:
        Reproject a UTM tile to EPSG:4326::

            from geotoolz import io

            wgs84 = io.ReadToCRS(
                src="/path/to/utm_image.tif",
                dst_crs="EPSG:4326",
                resolution=0.0001,
            )()
    """

    def __init__(
        self,
        *,
        src: Source,
        dst_crs: str,
        resolution: Resolution | None = None,
        bounds: Bounds | None = None,
        indexes: list[int] | None = None,
    ) -> None:
        self.src = src
        self.dst_crs = dst_crs
        self.resolution = resolution
        self.bounds = bounds
        self.indexes = indexes

    def _apply(self) -> GeoTensor:
        try:
            data = _coerce_source(self.src, self.indexes)
            if self.bounds is None:
                out = read.read_to_crs(
                    data,
                    self.dst_crs,
                    resolution_dst_crs=self.resolution,
                )
            else:
                out = read.read_reproject(
                    data,
                    dst_crs=self.dst_crs,
                    bounds=self.bounds,
                    resolution_dst_crs=self.resolution,
                )
        except (FileNotFoundError, OSError, RasterioIOError) as exc:
            raise _read_error(self.src, exc) from exc
        return out.load()

    def get_config(self) -> dict[str, Any]:
        return {
            "src": _src_config(self.src),
            "dst_crs": self.dst_crs,
            "resolution": self.resolution,
            "bounds": self.bounds,
            "indexes": self.indexes,
        }


class WriteCOG(SinkOperator):
    """Write a :class:`GeoTensor` as a Cloud Optimized GeoTIFF.

    Delegates to :func:`georeader.save.save_cog`, which generates
    internal tiling and overviews automatically. Supports local paths and
    cloud storage URIs (``gs://``, ``s3://``, ``az://``, ``abfs://``,
    ``oss://``).

    Args:
        path: Output path or cloud URI.
        profile: Extra rasterio profile entries (e.g. ``{"compress":
            "zstd", "RESAMPLING": "NEAREST"}`` for categorical data).
            Merged on top of ``{"compress": compress}``.
        compress: Default compression algorithm. ``"deflate"`` is a
            good general default; use ``"zstd"`` for archival, ``"lzw"``
            for compatibility.
        descriptions: Optional band names (length must equal band count).
        tags: Optional rasterio tags stored as TIFF metadata.

    Raises:
        GeoToolzIOError: If the file cannot be written.

    Examples:
        Save an NDVI raster as a COG::

            from geotoolz import io

            io.WriteCOG(path="/out/ndvi.tif", compress="zstd")(ndvi_geotensor)
    """

    def __init__(
        self,
        *,
        path: str | PathLike[str],
        profile: dict[str, Any] | None = None,
        compress: str = "deflate",
        descriptions: list[str] | None = None,
        tags: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.profile = profile
        self.compress = compress
        self.descriptions = descriptions
        self.tags = tags

    def _apply(self, gt: GeoTensor) -> None:
        merged_profile: dict[str, Any] = {"compress": self.compress}
        if self.profile is not None:
            merged_profile.update(self.profile)
        try:
            save.save_cog(
                gt,
                str(self.path),
                profile=merged_profile,
                descriptions=self.descriptions,
                tags=self.tags,
            )
        except (FileNotFoundError, OSError, RasterioIOError) as exc:
            raise GeoToolzIOError(f"Unable to write COG {self.path!s}: {exc}") from exc
        return None

    def get_config(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "profile": self.profile,
            "compress": self.compress,
            "descriptions": self.descriptions,
            "tags": self.tags,
        }


class WriteGeoTIFF(SinkOperator):
    """Write a :class:`GeoTensor` as a tiled GeoTIFF (no overviews).

    Delegates to :func:`georeader.save.save_tiled_geotiff`. Use
    :class:`WriteCOG` instead when you need overviews / HTTP range access.

    Args:
        path: Output path or cloud URI (gs://, s3://, az://, abfs://,
            oss://).
        profile: Extra rasterio profile entries merged on top of the
            georeader defaults (e.g. ``{"compress": "lzw"}``).
        blocksize: Internal tile size in pixels (square). Must be a
            power of two; common values are 256 and 512.
        descriptions: Optional band names (length must equal band count).
        tags: Optional rasterio tags stored as TIFF metadata.

    Raises:
        GeoToolzIOError: If ``gt`` has neither 2D nor 3D values, or if
            the file cannot be written.

    Examples:
        Save the output of a pipeline as a tiled GeoTIFF::

            from geotoolz import io

            io.WriteGeoTIFF(path="/out/result.tif")(result_geotensor)
    """

    def __init__(
        self,
        *,
        path: str | PathLike[str],
        profile: dict[str, Any] | None = None,
        blocksize: int = 256,
        descriptions: list[str] | None = None,
        tags: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.profile = profile
        self.blocksize = blocksize
        self.descriptions = descriptions
        self.tags = tags

    def _apply(self, gt: GeoTensor) -> None:
        if np.ndim(gt.values) not in (2, 3):
            raise GeoToolzIOError(
                "GeoTIFF output expects 2D or 3D data, found shape "
                f"{np.shape(gt.values)!r}."
            )
        try:
            save.save_tiled_geotiff(
                gt,
                str(self.path),
                profile_arg=self.profile,
                descriptions=self.descriptions,
                tags=self.tags,
                blocksize=self.blocksize,
            )
        except (FileNotFoundError, OSError, RasterioIOError) as exc:
            raise GeoToolzIOError(
                f"Unable to write GeoTIFF {self.path!s}: {exc}"
            ) from exc
        except NotImplementedError as exc:
            raise GeoToolzIOError(
                f"Unable to write GeoTIFF {self.path!s}: {exc}"
            ) from exc
        return None

    def get_config(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "profile": self.profile,
            "blocksize": self.blocksize,
            "descriptions": self.descriptions,
            "tags": self.tags,
        }


class WriteZarr(SinkOperator):
    """Write a :class:`GeoTensor` to a Zarr store with spatial metadata.

    Stores the array under ``values`` and the spatial metadata (CRS as
    a string, ``transform`` as a 6- or 9-tuple, ``fill_value_default``)
    as group attributes. Requires the optional ``streaming`` extra
    (``pip install geotoolz[streaming]``).

    Args:
        store: Zarr store URI (``"/path/to/output.zarr"``,
            ``"s3://bucket/output.zarr"``, etc.).
        group: Optional sub-group inside the store. ``None`` writes at
            the root.
        chunks: Optional per-axis chunk size, keyed by axis name
            (``"band"``, ``"y"``, ``"x"``). Missing axes inherit the
            array shape.

    Raises:
        GeoToolzIOError: If ``zarr`` is not installed.

    Examples:
        Write a time-series result to a chunked Zarr store::

            from geotoolz import io

            io.WriteZarr(
                store="/out/ndvi.zarr",
                chunks={"y": 256, "x": 256},
            )(ndvi_geotensor)
    """

    def __init__(
        self,
        *,
        store: str,
        group: str | None = None,
        chunks: dict[str, int] | None = None,
    ) -> None:
        self.store = store
        self.group = group
        self.chunks = chunks

    def _apply(self, gt: GeoTensor) -> None:
        try:
            import zarr
        except ImportError as exc:
            raise GeoToolzIOError(
                "WriteZarr requires the optional zarr dependency."
            ) from exc

        root = zarr.open_group(self.store, mode="w")
        group = root if self.group is None else root.require_group(self.group)
        values = np.asarray(gt.values)
        chunk_shape: tuple[int, ...] | None = None
        if self.chunks is not None:
            axis_names = ("band", "y", "x")[-values.ndim :]
            chunk_shape = tuple(
                self.chunks.get(name, size)
                for name, size in zip(axis_names, values.shape, strict=True)
            )
        group.create_array("values", data=values, chunks=chunk_shape)
        group.attrs["crs"] = str(gt.crs)
        group.attrs["transform"] = tuple(gt.transform)
        group.attrs["fill_value_default"] = gt.fill_value_default
        return None

    def get_config(self) -> dict[str, Any]:
        return {"store": self.store, "group": self.group, "chunks": self.chunks}


class LoadFromSTAC(SourceOperator):
    """Load a raster asset from a STAC item by asset key.

    Resolves ``item.assets[asset_key].href`` and reads it. When
    ``bounds`` is ``None`` the full asset is materialised via
    :class:`RasterioReader`; otherwise the read is delegated to
    :class:`ReadToCRS` (assumed in ``EPSG:4326``) so the asset is
    reprojected and cropped in one pass.

    Args:
        item: A STAC item-like object exposing
            ``item.assets[key].href``. Any duck-typed equivalent works
            (this operator does not depend on ``pystac``).
        asset_key: Asset name to load (e.g. ``"visual"``, ``"B04"``).
        bounds: Optional ``(xmin, ymin, xmax, ymax)`` crop in
            ``EPSG:4326``. ``None`` loads the full asset.
        resolution: Output pixel size in ``EPSG:4326`` units. Ignored
            when ``bounds`` is ``None``.

    Raises:
        GeoToolzIOError: If the asset key is missing or the underlying
            read fails.

    Examples:
        Load a Sentinel-2 NIR asset over an AOI::

            from geotoolz import io

            nir = io.LoadFromSTAC(
                item=stac_item,
                asset_key="B08",
                bounds=(-122.5, 37.7, -122.3, 37.9),
            )()
    """

    def __init__(
        self,
        *,
        item: Any,
        asset_key: str,
        bounds: Bounds | None = None,
        resolution: float | None = None,
    ) -> None:
        self.item = item
        self.asset_key = asset_key
        self.bounds = bounds
        self.resolution = resolution

    def _apply(self) -> GeoTensor:
        try:
            href = self.item.assets[self.asset_key].href
        except KeyError as exc:
            raise GeoToolzIOError(
                f"STAC item has no asset {self.asset_key!r}."
            ) from exc
        if self.bounds is None:
            try:
                return RasterioReader(href).load()
            except (FileNotFoundError, OSError, RasterioIOError) as exc:
                raise _read_error(href, exc) from exc
        return ReadToCRS(
            src=href,
            dst_crs="EPSG:4326",
            bounds=self.bounds,
            resolution=self.resolution,
        )()

    def get_config(self) -> dict[str, Any]:
        # The STAC item itself is an arbitrary runtime object; surface its
        # repr in config so users can debug pipelines, and rely on
        # ``forbid_in_yaml`` to flag that this won't round-trip to YAML.
        return {
            "item": repr(self.item),
            "asset_key": self.asset_key,
            "bounds": self.bounds,
            "resolution": self.resolution,
        }


class LoadFromEE(SourceOperator):
    """Load an Earth Engine image into a :class:`GeoTensor`.

    Wraps :func:`georeader.readers.ee_image.export_image`. The output
    grid is derived from ``bounds`` (upper-left corner) and ``scale``
    (pixel size in ``crs`` units) — i.e. ``Affine(scale, 0, xmin, 0,
    -scale, ymax)``.

    Args:
        image_id: Earth Engine asset ID (e.g.
            ``"LANDSAT/LC08/C02/T1_L2/LC08_001001_20200101"``).
        bounds: ``(xmin, ymin, xmax, ymax)`` in ``crs`` coordinates.
        crs: Output CRS (e.g. ``"EPSG:4326"``).
        scale: Pixel size in ``crs`` units (e.g. ``30`` for Landsat).
        bands: Optional list of band names to export. ``None`` exports
            the default band set.

    Raises:
        GeoToolzIOError: If ``earthengine-api`` / georeader's EE extras
            are not installed, or if the export fails.

    Examples:
        Export a Landsat 8 SR scene::

            from geotoolz import io

            scene = io.LoadFromEE(
                image_id="LANDSAT/LC08/C02/T1_L2/LC08_001001_20200101",
                bounds=(-122.5, 37.7, -122.3, 37.9),
                crs="EPSG:4326",
                scale=30.0,
                bands=["SR_B4", "SR_B5"],
            )()
    """

    def __init__(
        self,
        *,
        image_id: str,
        bounds: Bounds,
        crs: str,
        scale: float,
        bands: list[str] | None = None,
    ) -> None:
        self.image_id = image_id
        self.bounds = bounds
        self.crs = crs
        self.scale = scale
        self.bands = bands

    def _apply(self) -> GeoTensor:
        try:
            import ee
            from affine import Affine
            from georeader.readers.ee_image import export_image
        except ImportError as exc:
            raise GeoToolzIOError(
                "LoadFromEE requires georeader's Earth Engine dependencies."
            ) from exc

        xmin, ymax = self.bounds[0], self.bounds[3]
        transform: Affine = Affine(self.scale, 0.0, xmin, 0.0, -self.scale, ymax)
        try:
            return export_image(
                self.image_id,
                geometry=box(*self.bounds),
                transform=transform,
                crs=self.crs,
                bands_gee=[] if self.bands is None else self.bands,
                resolution_dst=self.scale,
            )
        except (ee.EEException, RuntimeError, ValueError, OSError) as exc:
            raise GeoToolzIOError(
                f"Unable to load Earth Engine image {self.image_id!r}: {exc}"
            ) from exc

    def get_config(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "bounds": self.bounds,
            "crs": self.crs,
            "scale": self.scale,
            "bands": self.bands,
        }


__all__ = [
    "GeoToolzIOError",
    "LoadFromEE",
    "LoadFromSTAC",
    "ReadBounds",
    "ReadCenterCoords",
    "ReadPolygon",
    "ReadReprojectLike",
    "ReadTile",
    "ReadToCRS",
    "ReadWindow",
    "SinkOperator",
    "SourceOperator",
    "WriteCOG",
    "WriteGeoTIFF",
    "WriteZarr",
]
