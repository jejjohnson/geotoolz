"""I/O operators: read rasters into a `geoarray.GeoArray`, write one out.

This module provides :class:`Operator` wrappers around file / cloud IO.
The reads and writes are computed by :mod:`georeader.read` and
:mod:`georeader.save` (through the engine boundary,
:mod:`geoops._src.engine`), but no georeader type crosses the public API.
Two families:

* **Source operators** (subclasses of :class:`SourceOperator`) take a
  raster source and produce an in-memory `geoarray.GeoArray`. A source is
  a path, an open rasterio dataset, a `geoarray.RasterSource` (a lazy
  reader: a geoproducts reader, or a georeader reader wrapped with
  ``geoarray.interop.georeader.source_from_georeader``) or a GeoArray.
  A georeader ``GeoTensor`` / ``GeoData`` is refused with a ``TypeError``
  naming ``from_georeader`` / ``source_from_georeader``. Sources override
  ``_apply`` with no positional input and are valid as the first step of
  a :class:`~pipekit.Sequential`.
* **Sink operators** (subclasses of :class:`SinkOperator`) consume a
  ``GeoArray`` and write it to disk / object storage. They are marked
  ``_terminal = True`` so :class:`~pipekit.Sequential` only
  accepts them as the last step.

Configs built from paths, bounds, windows and asset IDs are plain JSON
and round-trip through ``Operator.from_state`` / YAML. Mapping options
(sink ``profile`` / ``tags`` / ``chunks``) are emitted as
``[[key, value], ...]`` pairs and accepted in either form. A source given as
a runtime object (an open reader, a ``GeoArray``) stays in the config
as-is, so ``from_state`` refuses it as non-primitive. ``LoadFromSTAC``
(a runtime STAC item) and ``ReadReprojectLike`` (a runtime reference
grid) always hold runtime objects and set ``forbid_in_yaml = True``.

See the docs' Concepts page ("Round-trip discipline") for the
``forbid_in_yaml`` contract.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from geoarray import GeoArray, RasterSource
from geoarray.interop.georeader import source_from_georeader
from pipekit import Operator

from geoops._src.config import (
    as_tuple,
    jsonable,
    mapping_from_pairs,
    mapping_to_pairs,
)
from geoops._src.engine import from_engine, to_engine
from geoops._src.geo import reject_georeader, require_geoarray
from geoops._src.optional import optional_name, optional_root
from geoops.io._src.errors import GeoToolzIOError


if TYPE_CHECKING:
    # annotations only: rasterio is the [raster] extra, shapely [vector]
    from rasterio.io import DatasetReaderBase
    from rasterio.windows import Window
    from shapely.geometry import MultiPolygon, Polygon

    Source = str | PathLike[str] | DatasetReaderBase | RasterSource | GeoArray

Bounds = tuple[float, float, float, float]
Resolution = float | tuple[float, float]

#: The backend module a geotoolz-cloud reader needs → the geotoolz extra.
_BACKEND_EXTRAS = {
    "h5py": "hdf5",
    "pyhdf": "hdf4",
    "pyhdf.SD": "hdf4",
    "netCDF4": "netcdf",
}


def _cloud_call(extra: str, feature: str, call: Callable[[], Any]) -> Any:
    """Run ``call`` (which uses geotoolz-cloud), naming our extras on failure.

    ``extra`` is the geotoolz extra that installs geotoolz-cloud with the
    backend ``feature`` needs; a backend geotoolz-cloud reports missing is
    mapped to its own geotoolz extra (``h5py`` → ``hdf5``, …).
    """
    try:
        return call()
    except ImportError as exc:
        missing = _BACKEND_EXTRAS.get(getattr(exc, "name", None) or "", extra)
        raise ImportError(
            f"{feature} requires an optional dependency that is not installed. "
            f"Install it with `pip install 'geotoolz-ops[{missing}]'`."
        ) from exc


class SourceOperator(Operator):
    """Operator that produces a `GeoArray` without an input carrier.

    Source operators are valid as the *first* step of a
    :class:`~pipekit.Sequential` because
    :class:`~pipekit.Sequential` calls ``op()`` (no input) when the
    pipeline is invoked without a carrier.

    Graph mode: a source is not a :class:`~pipekit.Graph` node. Its
    ``_apply`` takes no carrier, while a graph node is always built from
    (and later called with) at least one parent ``Node`` — and calling
    ``op()`` with no argument while building a graph simply reads
    eagerly. Run the source first (or as the head of a ``Sequential``
    that wraps the graph) and feed its ``GeoArray`` to the graph's
    ``Input``.
    """


class SinkOperator(Operator):
    """Terminal operator that consumes a `GeoArray` and writes it.

    Sink operators are marked ``_terminal = True`` so
    :class:`~pipekit.Sequential` only accepts them as the *last*
    step (anywhere else they would break the GeoArray → next op
    contract by returning ``None``).
    """

    _terminal: ClassVar[bool] = True


def _window_config(window: Window) -> tuple[float, float, float, float]:
    return (window.col_off, window.row_off, window.width, window.height)


def _coerce_window(window: Window | tuple[float, float, float, float]) -> Window:
    Window = optional_name(
        "rasterio.windows",
        "Window",
        "raster",
        feature="geoops.io",
    )
    if isinstance(window, Window):
        return window
    col_off, row_off, width, height = window
    return Window(col_off=col_off, row_off=row_off, width=width, height=height)


def _coerce_source(
    src: Source, indexes: list[int] | None = None, op_name: str = "geoops.io"
) -> RasterSource:
    """Normalise a user-supplied source into a `geoarray.RasterSource`.

    Accepts a path-like, an open :class:`rasterio.io.DatasetReaderBase`
    (reopened by name), a `RasterSource` or a `GeoArray` (itself a
    `RasterSource`). Paths and datasets are opened with georeader's
    ``RasterioReader`` wrapped by ``source_from_georeader``. When
    ``indexes`` is given, the source is reconfigured to read only those
    1-indexed bands; that needs a rasterio-backed source (a path, a
    dataset, or a ``source_from_georeader`` reader).

    Raises:
        TypeError: ``src`` is a georeader ``GeoTensor`` / ``GeoData``
            (convert it with ``from_georeader`` / ``source_from_georeader``)
            or not a raster source at all.
        GeoToolzIOError: ``indexes`` is given for a source that cannot
            select bands.
    """
    DatasetReaderBase = optional_name(
        "rasterio.io",
        "DatasetReaderBase",
        "raster",
        feature="geoops.io",
    )
    RasterioReader = optional_name(
        "georeader.rasterio_reader",
        "RasterioReader",
        "sensors",
        feature="geoops.io",
    )
    reject_georeader(src, op_name)
    if isinstance(src, str | PathLike):
        return source_from_georeader(RasterioReader(str(src), indexes=indexes))

    if isinstance(src, DatasetReaderBase):
        # An open dataset is not a raster source; reopen it by name.
        return source_from_georeader(RasterioReader(src.name, indexes=indexes))

    if not isinstance(src, GeoArray | RasterSource):
        raise TypeError(
            f"{op_name} reads a path, an open rasterio dataset, a "
            f"geoarray.RasterSource or a GeoArray; got {type(src).__name__}"
        )

    if indexes is None:
        return src

    engine = None if isinstance(src, GeoArray) else to_engine(src)
    if isinstance(engine, RasterioReader):
        reader = engine.copy()
        reader.set_indexes(indexes, relative=False)
        return source_from_georeader(reader)
    raise GeoToolzIOError(
        "indexes are only supported for path-like and rasterio-backed sources."
    )


def _src_config(src: Source) -> str | Source:
    if isinstance(src, str | PathLike):
        return str(src)
    return src


def _read_error(src: Source, exc: Exception) -> GeoToolzIOError:
    return GeoToolzIOError(f"Unable to read raster source {src!r}: {exc}")


def _load_or_raise(out: Any, src: Source, what: str) -> GeoArray:
    """Materialise a georeader read as a `GeoArray`, or raise if it missed.

    georeader's ``read_from_*`` helpers return ``None`` when a
    ``boundless=False`` request (or a web-map tile) lies outside the raster.

    Args:
        out: Lazy georeader result, or ``None``.
        src: User-supplied source, used in the error message.
        what: Human-readable description of the request.

    Raises:
        GeoToolzIOError: If ``out`` is ``None``.
    """
    if out is None:
        raise GeoToolzIOError(f"{what} does not intersect {src!r}.")
    return from_engine(out.load())


class ReadWindow(SourceOperator):
    """Read a pixel-space rectangular window from a raster source.

    Wraps :func:`georeader.read.read_from_window`. The window is specified
    in pixel coordinates of the source raster as ``(col_off, row_off,
    width, height)``.

    Args:
        src: Path, open rasterio dataset, `geoarray.RasterSource` (e.g. a
            georeader reader wrapped with ``source_from_georeader``) or
            `GeoArray`. ``indexes`` needs a rasterio-backed source.
        window: :class:`rasterio.windows.Window` or 4-tuple
            ``(col_off, row_off, width, height)`` in source pixel space.
        indexes: 1-indexed list of bands to read. ``None`` reads all.
        boundless: If ``True``, requests outside the source bounds are
            zero-padded; if ``False`` they raise.

    Raises:
        GeoToolzIOError: If the source cannot be opened, the window does
            not intersect the source (``boundless=False``), or ``indexes``
            is given for a source that cannot select bands.
        TypeError: If ``src`` is a georeader ``GeoTensor`` / ``GeoData``.

    Examples:
        Read a 256 by 256 chip from the top-left corner of a GeoTIFF::

            from geoops import io
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

    def _apply(self) -> GeoArray:
        RasterioIOError = optional_name(
            "rasterio.errors",
            "RasterioIOError",
            "raster",
            feature="geoops.io.ReadWindow",
        )
        read = optional_name(
            "georeader", "read", "sensors", feature="geoops.io.ReadWindow"
        )
        try:
            data = to_engine(_coerce_source(self.src, self.indexes, "ReadWindow"))
            out = read.read_from_window(data, self.window, boundless=self.boundless)
        except (FileNotFoundError, OSError, RasterioIOError) as exc:
            raise _read_error(self.src, exc) from exc
        return _load_or_raise(out, self.src, f"Window {_window_config(self.window)!r}")

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
    `GeoArray` keeps the *source* CRS and transform.

    Args:
        src: Path, open rasterio dataset, `geoarray.RasterSource` or
            `GeoArray`.
        bounds: ``(xmin, ymin, xmax, ymax)`` in ``crs`` coordinates.
        crs: CRS of ``bounds`` (e.g. ``"EPSG:4326"``). ``None`` uses the
            source CRS directly.
        indexes: 1-indexed list of bands to read. ``None`` reads all.
        boundless: If ``True``, pad the read with the source's nodata
            where the bounds extend outside the raster.

    Raises:
        GeoToolzIOError: If the source cannot be opened, the request does
            not intersect the source (``boundless=False``), or ``indexes``
            is given for a source that cannot select bands.

    Examples:
        Read the full extent of a GeoTIFF as a single chip::

            from geoops import io
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
        self.bounds = as_tuple(bounds)
        self.crs = crs
        self.indexes = indexes
        self.boundless = boundless

    def _apply(self) -> GeoArray:
        RasterioIOError = optional_name(
            "rasterio.errors",
            "RasterioIOError",
            "raster",
            feature="geoops.io.ReadBounds",
        )
        read = optional_name(
            "georeader", "read", "sensors", feature="geoops.io.ReadBounds"
        )
        try:
            data = to_engine(_coerce_source(self.src, self.indexes, "ReadBounds"))
            out = read.read_from_bounds(
                data,
                self.bounds,
                crs_bounds=self.crs,
                boundless=self.boundless,
            )
        except (FileNotFoundError, OSError, RasterioIOError) as exc:
            raise _read_error(self.src, exc) from exc
        return _load_or_raise(out, self.src, f"Bounds {self.bounds!r}")

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
        src: Path, open rasterio dataset, `geoarray.RasterSource` or
            `GeoArray`.
        center: ``(x, y)`` in ``crs`` coordinates.
        shape: ``(height, width)`` in pixels.
        crs: CRS of ``center`` (e.g. ``"EPSG:4326"``). ``None`` uses the
            source CRS directly.
        indexes: 1-indexed list of bands to read. ``None`` reads all.
        boundless: If ``True``, requests outside the source are
            zero-padded; if ``False`` they raise.

    Raises:
        GeoToolzIOError: If the source cannot be opened, the request does
            not intersect the source (``boundless=False``), or ``indexes``
            is given for a source that cannot select bands.

    Examples:
        Read a 64 by 64 chip around a longitude/latitude point::

            from geoops import io

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
        self.center = as_tuple(center)
        self.shape = as_tuple(shape)
        self.crs = crs
        self.indexes = indexes
        self.boundless = boundless

    def _apply(self) -> GeoArray:
        RasterioIOError = optional_name(
            "rasterio.errors",
            "RasterioIOError",
            "raster",
            feature="geoops.io.ReadCenterCoords",
        )
        read = optional_name(
            "georeader", "read", "sensors", feature="geoops.io.ReadCenterCoords"
        )
        try:
            source = _coerce_source(self.src, self.indexes, "ReadCenterCoords")
            out = read.read_from_center_coords(
                to_engine(source),
                self.center,
                self.shape,
                crs_center_coords=self.crs,
                boundless=self.boundless,
            )
        except (FileNotFoundError, OSError, RasterioIOError) as exc:
            raise _read_error(self.src, exc) from exc
        return _load_or_raise(
            out, self.src, f"Window of shape {self.shape!r} centred on {self.center!r}"
        )

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
        src: Path, open rasterio dataset, `geoarray.RasterSource` or
            `GeoArray`.
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

            from geoops import io

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
        self.tile = as_tuple(tile)
        self.indexes = indexes
        self.dst_crs = dst_crs
        self.out_shape = as_tuple(out_shape)
        self.resolution = as_tuple(resolution)

    def _apply(self) -> GeoArray:
        RasterioIOError = optional_name(
            "rasterio.errors",
            "RasterioIOError",
            "raster",
            feature="geoops.io.ReadTile",
        )
        read = optional_name(
            "georeader", "read", "sensors", feature="geoops.io.ReadTile"
        )
        z, x, y = self.tile
        try:
            data = to_engine(_coerce_source(self.src, self.indexes, "ReadTile"))
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
        return _load_or_raise(out, self.src, f"Tile {self.tile!r}")

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
        src: Path, open rasterio dataset, `geoarray.RasterSource` or
            `GeoArray`.
        polygon: :class:`shapely.geometry.Polygon` or
            :class:`shapely.geometry.MultiPolygon`, or its WKT string (the
            form ``get_config`` emits).
        crs: CRS of ``polygon``. ``None`` uses the source CRS directly.
        indexes: 1-indexed list of bands to read. ``None`` reads all.
        boundless: If ``True``, the read is padded with the source's
            nodata where the polygon's bounding box extends outside the
            raster.

    Raises:
        GeoToolzIOError: If the source cannot be opened, the request does
            not intersect the source (``boundless=False``), or ``indexes``
            is given for a source that cannot select bands.

    Examples:
        Read all pixels intersecting an AOI polygon::

            from geoops import io
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
        polygon: Polygon | MultiPolygon | str,
        crs: str | None = None,
        indexes: list[int] | None = None,
        boundless: bool = True,
    ) -> None:
        shapely = optional_root(
            "shapely.wkt", "vector", feature="geoops.io.ReadPolygon"
        )
        self.src = src
        self.polygon = (
            shapely.wkt.loads(polygon) if isinstance(polygon, str) else polygon
        )
        self.crs = crs
        self.indexes = indexes
        self.boundless = boundless

    def _apply(self) -> GeoArray:
        RasterioIOError = optional_name(
            "rasterio.errors",
            "RasterioIOError",
            "raster",
            feature="geoops.io.ReadPolygon",
        )
        read = optional_name(
            "georeader", "read", "sensors", feature="geoops.io.ReadPolygon"
        )
        try:
            data = to_engine(_coerce_source(self.src, self.indexes, "ReadPolygon"))
            out = read.read_from_polygon(
                data,
                self.polygon,
                crs_polygon=self.crs,
                boundless=self.boundless,
            )
        except (FileNotFoundError, OSError, RasterioIOError) as exc:
            raise _read_error(self.src, exc) from exc
        return _load_or_raise(
            out, self.src, f"Polygon with bounds {self.polygon.bounds!r}"
        )

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
    `GeoArray` has the *same* CRS, transform, and shape as
    ``like``.

    Args:
        src: Path, open rasterio dataset, `geoarray.RasterSource` or
            `GeoArray` to read from.
        like: Reference `GeoArray`, `geoarray.RasterSource` or path whose
            grid (CRS + transform + shape) defines the output.
        indexes: 1-indexed list of bands to read from ``src``. ``None``
            reads all.
        resolution: Optional override for the output pixel size in
            ``like``'s CRS units; ``None`` keeps ``like``'s native
            resolution.

    Raises:
        GeoToolzIOError: If either source cannot be opened or ``indexes``
            is given for a source that cannot select bands.

    Examples:
        Align a single-band mask onto a reference reflectance grid::

            from geoops import io

            aligned = io.ReadReprojectLike(
                src="/path/to/mask.tif",
                like=reference,  # a GeoArray (C, H, W)
                indexes=[1],
            )()
    """

    forbid_in_yaml: ClassVar[bool] = True

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
        self.resolution = as_tuple(resolution)

    def _apply(self) -> GeoArray:
        RasterioIOError = optional_name(
            "rasterio.errors",
            "RasterioIOError",
            "raster",
            feature="geoops.io.ReadReprojectLike",
        )
        read = optional_name(
            "georeader", "read", "sensors", feature="geoops.io.ReadReprojectLike"
        )
        try:
            like = to_engine(_coerce_source(self.like, None, "ReadReprojectLike"))
        except (FileNotFoundError, OSError, RasterioIOError) as exc:
            raise _read_error(self.like, exc) from exc
        try:
            source = _coerce_source(self.src, self.indexes, "ReadReprojectLike")
            out = read.read_reproject_like(
                to_engine(source),
                like,
                resolution_dst=self.resolution,
            )
        except (FileNotFoundError, OSError, RasterioIOError) as exc:
            raise _read_error(self.src, exc) from exc
        return from_engine(out.load())

    def get_config(self) -> dict[str, Any]:
        return {
            "src": _src_config(self.src),
            # ``like`` is often a runtime GeoArray / reader / array — keep
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
        src: Path, open rasterio dataset, `geoarray.RasterSource` or
            `GeoArray`.
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
            given for a source that cannot select bands.

    Examples:
        Reproject a UTM tile to EPSG:4326::

            from geoops import io

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
        self.resolution = as_tuple(resolution)
        self.bounds = as_tuple(bounds)
        self.indexes = indexes

    def _apply(self) -> GeoArray:
        RasterioIOError = optional_name(
            "rasterio.errors",
            "RasterioIOError",
            "raster",
            feature="geoops.io.ReadToCRS",
        )
        read = optional_name(
            "georeader", "read", "sensors", feature="geoops.io.ReadToCRS"
        )
        try:
            data = to_engine(_coerce_source(self.src, self.indexes, "ReadToCRS"))
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
        return from_engine(out.load())

    def get_config(self) -> dict[str, Any]:
        return {
            "src": _src_config(self.src),
            "dst_crs": self.dst_crs,
            "resolution": self.resolution,
            "bounds": self.bounds,
            "indexes": self.indexes,
        }


class ReadHDF(SourceOperator):
    """Read a dataset from an HDF4/HDF-EOS or HDF5 file, local or remote.

    Delegates to `geocloud.hdf.read_hdf`: HDF5 is read with h5py over
    ranged requests (only the selected bands), HDF4 / HDF-EOS with pyhdf
    on a local copy; the format is told by the file signature. The result
    has an identity transform unless sensor-specific georeferencing is
    applied later. A swath's latitude / longitude datasets lie on the same
    pixel grid: read each with its own ``ReadHDF``.

    Args:
        path: HDF file path, or any URI geotoolz-cloud reads (``s3://``,
            ``gs://``, ``az://``, signed ``https://``, ``hf://``).
        dataset: Dataset path inside the file.
        indexes: Optional 1-based band indexes along the leading dataset axis.
        metadata_groups: Optional HDF5 group paths whose attributes are copied
            into ``attrs["metadata"]``.

    Returns (on call):
        A `GeoArray` ``(..., H, W)``.
    """

    def __init__(
        self,
        *,
        path: str | PathLike[str],
        dataset: str,
        indexes: list[int] | None = None,
        metadata_groups: list[str] | None = None,
    ) -> None:
        self.path = os.fspath(path)
        self.dataset = dataset
        self.indexes = indexes
        self.metadata_groups = metadata_groups

    def _apply(self) -> GeoArray:
        def call() -> Any:
            from geocloud.hdf import read_hdf

            return read_hdf(
                self.path,
                self.dataset,
                indexes=self.indexes,
                metadata_groups=self.metadata_groups,
            )

        try:
            out = _cloud_call("hdf5", "ReadHDF", call)
        except (KeyError, OSError, ValueError) as exc:
            raise _read_error(self.path, exc) from exc
        return out  # geocloud returns a GeoArray


class ReadNetCDF(SourceOperator):
    """Read a variable from a NetCDF-3 / NetCDF-4 file, local or remote.

    Delegates to `geocloud.hdf.read_netcdf` (netCDF4 on a local copy of a
    remote file): CF masking and scale / offset, the CRS from the
    variable's CF ``grid_mapping`` and the transform from GDAL's
    ``GeoTransform`` attribute.

    Args:
        path: NetCDF file path, or any URI geotoolz-cloud reads.
        variable: Variable name within ``group``.
        group: Optional slash-separated NetCDF group path.
        indexes: Optional 1-based band indexes along the leading variable axis.
        decode_cf: If ``True``, apply the backend's mask/scale handling.
        use_cf_grid_mapping: If ``True``, recover CRS from the variable's
            ``grid_mapping`` attribute when present.
    """

    def __init__(
        self,
        *,
        path: str | PathLike[str],
        variable: str,
        group: str | None = None,
        indexes: list[int] | None = None,
        decode_cf: bool = True,
        use_cf_grid_mapping: bool = True,
    ) -> None:
        self.path = os.fspath(path)
        self.variable = variable
        self.group = group
        self.indexes = indexes
        self.decode_cf = decode_cf
        self.use_cf_grid_mapping = use_cf_grid_mapping

    def _apply(self) -> GeoArray:
        def call() -> Any:
            from geocloud.hdf import read_netcdf

            return read_netcdf(
                self.path,
                self.variable,
                group=self.group,
                indexes=self.indexes,
                decode_cf=self.decode_cf,
                use_cf_grid_mapping=self.use_cf_grid_mapping,
            )

        try:
            out = _cloud_call("netcdf", "ReadNetCDF", call)
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            raise _read_error(self.path, exc) from exc
        return out  # geocloud returns a GeoArray


class WriteCOG(SinkOperator):
    """Write a `GeoArray` as a validated Cloud-Optimized GeoTIFF.

    Delegates to `geocloud.cog.write_cog` (the ``[cloud]`` extra): staged
    in a temporary directory, translated by GDAL's ``COG`` driver, checked,
    then renamed into place or uploaded — so a local path or a bucket URI
    (``s3://``, ``gs://``, ``az://``) only ever holds a complete COG.

    Args:
        path: Output path or cloud URI.
        compress: ``"deflate"`` (default), ``"zstd"``, ``"lzw"``,
            ``"lerc"`` / ``"lerc_deflate"`` / ``"lerc_zstd"``, ``"webp"``,
            ``"jpeg"`` or ``"none"``.
        level: Compression level; ``None`` keeps GDAL's default.
        predictor: Apply the dtype's predictor (DEFLATE / LZW / ZSTD).
        blocksize: Tile size in pixels (a power of two).
        overviews: Build overviews down to a single tile.
        resampling: Overview resampling; ``None`` picks ``"nearest"`` for
            integers and ``"average"`` for floats.
        nodata: The nodata value; ``"auto"`` takes the carrier's
            ``fill_value`` when the dtype can hold it, ``None`` writes none.
        descriptions: Band names (length must equal band count); ``None``
            takes the carrier's ``band_names``.
        tags: Rasterio tags stored as TIFF metadata.
        creation_options: Extra GDAL ``COG`` driver options, passed as is.

    Raises:
        TypeError: If ``gt`` is not a `GeoArray`.
        GeoToolzIOError: If ``gt`` is not 2D or 3D (a 4-D ``(T, C, H, W)``
            stack has no GeoTIFF layout), or the file cannot be written.

    Examples:
        Save an NDVI raster as a COG::

            from geoops import io

            # ndvi: GeoArray (1, H, W) float32
            io.WriteCOG(path="s3://bucket/ndvi.tif", compress="zstd")(ndvi)
    """

    def __init__(
        self,
        *,
        path: str | PathLike[str],
        compress: str = "deflate",
        level: int | None = None,
        predictor: bool = True,
        blocksize: int = 512,
        overviews: bool = True,
        resampling: str | None = None,
        nodata: float | int | str | None = "auto",
        descriptions: list[str] | None = None,
        tags: dict[str, Any] | list[list[Any]] | None = None,
        creation_options: dict[str, Any] | list[list[Any]] | None = None,
    ) -> None:
        self.path = os.fspath(path)
        self.compress = compress
        self.level = level
        self.predictor = predictor
        self.blocksize = blocksize
        self.overviews = overviews
        self.resampling = resampling
        self.nodata = nodata
        self.descriptions = descriptions
        self.tags = mapping_from_pairs(tags)
        self.creation_options = mapping_from_pairs(creation_options)

    def _apply(self, gt: GeoArray) -> None:
        RasterioIOError = optional_name(
            "rasterio.errors",
            "RasterioIOError",
            "raster",
            feature="geoops.io.WriteCOG",
        )
        require_geoarray(gt, "WriteCOG")
        if len(gt.shape) not in (2, 3):
            raise GeoToolzIOError(
                "WriteCOG expects 2D or 3D data ((H, W) or (C, H, W)), found shape "
                f"{gt.shape!r}; write each frame of a (T, C, H, W) "
                "stack separately or use WriteZarr."
            )

        def call() -> None:
            from geocloud.cog import write_cog

            write_cog(
                gt,
                self.path,
                compress=self.compress,
                level=self.level,
                predictor=self.predictor,
                blocksize=self.blocksize,
                overviews=self.overviews,
                resampling=self.resampling,
                nodata=self.nodata,  # type: ignore[arg-type]
                descriptions=self.descriptions,
                tags=self.tags,
                creation_options=self.creation_options,
            )

        try:
            _cloud_call("cloud", "WriteCOG", call)
        except (OSError, RasterioIOError, RuntimeError, ValueError) as exc:
            raise GeoToolzIOError(f"Unable to write COG {self.path!s}: {exc}") from exc
        return None

    def get_config(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "compress": self.compress,
            "level": self.level,
            "predictor": self.predictor,
            "blocksize": self.blocksize,
            "overviews": self.overviews,
            "resampling": self.resampling,
            "nodata": self.nodata,
            "descriptions": self.descriptions,
            "tags": mapping_to_pairs(self.tags),
            "creation_options": mapping_to_pairs(self.creation_options),
        }


class WriteGeoTIFF(SinkOperator):
    """Write a `GeoArray` as a tiled GeoTIFF (no overviews).

    Delegates to :func:`georeader.save.save_tiled_geotiff`. Use
    :class:`WriteCOG` instead when you need overviews / HTTP range access.
    A cloud URI is written to a temporary local file first, then uploaded
    with `geocloud.files.upload` (the ``[cloud]`` extra), so the object
    only appears once complete.

    Args:
        path: Output path, or a cloud URI geotoolz-cloud writes to
            (``s3://``, ``gs://``, ``az://`` / ``abfs[s]://``).
        profile: Extra rasterio profile entries merged on top of the
            georeader defaults (e.g. ``{"compress": "lzw"}``).
        blocksize: Internal tile size in pixels (square). Must be a
            power of two; common values are 256 and 512.
        descriptions: Optional band names (length must equal band count).
        tags: Optional rasterio tags stored as TIFF metadata.

    Raises:
        TypeError: If ``gt`` is not a `GeoArray`.
        GeoToolzIOError: If ``gt`` has neither 2D nor 3D values, or if
            the file cannot be written.

    Examples:
        Save the output of a pipeline as a tiled GeoTIFF::

            from geoops import io

            # result: GeoArray (C, H, W)
            io.WriteGeoTIFF(path="/out/result.tif")(result)
    """

    def __init__(
        self,
        *,
        path: str | PathLike[str],
        profile: dict[str, Any] | list[list[Any]] | None = None,
        blocksize: int = 256,
        descriptions: list[str] | None = None,
        tags: dict[str, Any] | list[list[Any]] | None = None,
    ) -> None:
        self.path = os.fspath(path)
        self.profile = mapping_from_pairs(profile)
        self.blocksize = blocksize
        self.descriptions = descriptions
        self.tags = mapping_from_pairs(tags)

    def _apply(self, gt: GeoArray) -> None:
        RasterioIOError = optional_name(
            "rasterio.errors",
            "RasterioIOError",
            "raster",
            feature="geoops.io.WriteGeoTIFF",
        )
        require_geoarray(gt, "WriteGeoTIFF")
        if len(gt.shape) not in (2, 3):
            raise GeoToolzIOError(
                "WriteGeoTIFF expects 2D or 3D data ((H, W) or (C, H, W)), found shape "
                f"{gt.shape!r}; write each frame of a (T, C, H, W) "
                "stack separately or use WriteZarr."
            )
        remote = "://" in self.path and not self.path.startswith("file://")
        try:
            if not remote:
                self._save(gt, self.path.removeprefix("file://"))
                return None
            with tempfile.TemporaryDirectory() as tmp:
                staged = str(Path(tmp) / "out.tif")
                self._save(gt, staged)

                def upload() -> None:
                    from geocloud import files

                    files.upload(staged, self.path)

                _cloud_call("cloud", "WriteGeoTIFF to a cloud URI", upload)
        except (FileNotFoundError, OSError, RasterioIOError) as exc:
            raise GeoToolzIOError(
                f"Unable to write GeoTIFF {self.path!s}: {exc}"
            ) from exc
        except NotImplementedError as exc:
            raise GeoToolzIOError(
                f"Unable to write GeoTIFF {self.path!s}: {exc}"
            ) from exc
        return None

    def _save(self, gt: GeoArray, path: str) -> None:
        save = optional_name(
            "georeader", "save", "sensors", feature="geoops.io.WriteGeoTIFF"
        )
        save.save_tiled_geotiff(
            to_engine(gt),
            path,
            profile_arg=self.profile,
            descriptions=self.descriptions,
            tags=self.tags,
            blocksize=self.blocksize,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "profile": mapping_to_pairs(self.profile),
            "blocksize": self.blocksize,
            "descriptions": self.descriptions,
            "tags": mapping_to_pairs(self.tags),
        }


class WriteZarr(SinkOperator):
    """Write a `GeoArray` to a Zarr store with spatial metadata.

    Stores the array under ``values`` and the spatial metadata as group
    attributes: ``crs`` (a string), ``transform`` (the six affine
    coefficients ``(a, b, c, d, e, f)``) and ``fill_value`` (stores written
    before the GeoArray switch call it ``fill_value_default`` and hold the
    nine coefficients of an ``Affine``; a reader should accept both). The
    store is opened in append mode: other groups already in it are kept,
    while an existing ``values`` array in the target group is overwritten.
    Requires the optional ``zarr`` extra (``pip install
    geotoolz-ops[zarr]``, zarr-python >= 3).

    Args:
        store: Zarr store URI (``"/path/to/output.zarr"``,
            ``"s3://bucket/output.zarr"``, etc.).
        group: Optional sub-group inside the store. ``None`` writes at
            the root.
        chunks: Optional per-axis chunk size, keyed by axis name. Axes
            are named ``("time", "band", "y", "x")`` right-aligned to the
            array, so a 3-D array has ``"band"``, ``"y"``, ``"x"``.
            Missing axes inherit the array shape.

    Raises:
        TypeError: If ``gt`` is not a `GeoArray`.
        GeoToolzIOError: If ``zarr`` is not installed or ``gt`` is not
            2-D to 4-D (a batched carrier).

    Examples:
        Write a time-series result to a chunked Zarr store::

            from geoops import io

            # ndvi: GeoArray (T, 1, H, W) float32
            io.WriteZarr(
                store="/out/ndvi.zarr",
                chunks={"y": 256, "x": 256},
            )(ndvi)
    """

    def __init__(
        self,
        *,
        store: str,
        group: str | None = None,
        chunks: dict[str, int] | list[list[Any]] | None = None,
    ) -> None:
        self.store = store
        self.group = group
        self.chunks = mapping_from_pairs(chunks)

    def get_config(self) -> dict[str, Any]:
        # ``from_state`` rejects dict leaves; emit ``chunks`` as pairs.
        return {**super().get_config(), "chunks": mapping_to_pairs(self.chunks)}

    def _apply(self, gt: GeoArray) -> None:
        require_geoarray(gt, "WriteZarr")
        try:
            import zarr
        except ImportError as exc:
            # GeoToolzIOError (not ImportError) is WriteZarr's documented
            # contract; the message still names the extra to install.
            raise GeoToolzIOError(
                "WriteZarr requires the optional zarr dependency; install it "
                "with `pip install 'geotoolz-ops[zarr]'`."
            ) from exc

        values = np.asarray(gt)
        all_axes = ("time", "band", "y", "x")
        if not 2 <= values.ndim <= len(all_axes):
            raise GeoToolzIOError(
                f"Zarr output expects 2D to 4D data, found shape {values.shape!r}."
            )
        # Append mode: never truncate other groups already in the store.
        root = zarr.open_group(self.store, mode="a")
        group = root if self.group is None else root.require_group(self.group)
        if self.chunks is not None:
            axis_names = all_axes[-values.ndim :]
            chunk_shape: tuple[int, ...] = tuple(
                self.chunks.get(name, size)
                for name, size in zip(axis_names, values.shape, strict=True)
            )
            group.create_array(
                "values", data=values, chunks=chunk_shape, overwrite=True
            )
        else:
            group.create_array("values", data=values, overwrite=True)
        group.attrs["crs"] = str(gt.crs)
        group.attrs["transform"] = np.asarray(gt.transform).tolist()
        group.attrs["fill_value"] = jsonable(gt.fill_value)
        return None


class LoadFromSTAC(SourceOperator):
    """Load a raster asset from a STAC item by asset key.

    Resolves ``item.assets[asset_key].href`` and reads it. When
    ``bounds`` is ``None`` the full asset is materialised (georeader's
    ``RasterioReader``, as a `geoarray.RasterSource`); otherwise the read
    is delegated to :class:`ReadToCRS` (assumed in ``EPSG:4326``) so the asset is
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

            from geoops import io

            nir = io.LoadFromSTAC(
                item=stac_item,
                asset_key="B08",
                bounds=(-122.5, 37.7, -122.3, 37.9),
            )()
    """

    forbid_in_yaml: ClassVar[bool] = True

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
        self.bounds = as_tuple(bounds)
        self.resolution = as_tuple(resolution)

    def _apply(self) -> GeoArray:
        RasterioIOError = optional_name(
            "rasterio.errors",
            "RasterioIOError",
            "raster",
            feature="geoops.io.LoadFromSTAC",
        )
        RasterioReader = optional_name(
            "georeader.rasterio_reader",
            "RasterioReader",
            "sensors",
            feature="geoops.io.LoadFromSTAC",
        )
        try:
            href = self.item.assets[self.asset_key].href
        except KeyError as exc:
            raise GeoToolzIOError(
                f"STAC item has no asset {self.asset_key!r}."
            ) from exc
        if self.bounds is None:
            try:
                return source_from_georeader(RasterioReader(href)).load()
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
    """Load an Earth Engine image into a `GeoArray`.

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

            from geoops import io

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
        self.bounds = as_tuple(bounds)
        self.crs = crs
        self.scale = scale
        self.bands = bands

    def _apply(self) -> GeoArray:
        box = optional_name(
            "shapely.geometry",
            "box",
            "vector",
            feature="geoops.io.LoadFromEE",
        )
        try:
            import ee
        except ImportError as exc:
            raise GeoToolzIOError(
                "LoadFromEE requires the Earth Engine API: pip install earthengine-api."
            ) from exc
        export_image = optional_name(
            "georeader.readers.ee_image",
            "export_image",
            "sensors",
            feature="geoops.io.LoadFromEE",
        )
        from affine import Affine

        xmin, ymax = self.bounds[0], self.bounds[3]
        transform: Affine = Affine(self.scale, 0.0, xmin, 0.0, -self.scale, ymax)
        try:
            # ``bounds`` are in ``crs`` (georeader assumes EPSG:4326 unless
            # told otherwise). The pixel size is already encoded in
            # ``transform``, so no ``resolution_dst`` is passed.
            out = export_image(
                self.image_id,
                geometry=box(*self.bounds),
                transform=transform,
                crs=self.crs,
                crs_polygon=self.crs,
                bands_gee=[] if self.bands is None else self.bands,
            )
        except (ee.EEException, RuntimeError, ValueError, OSError) as exc:
            raise GeoToolzIOError(
                f"Unable to load Earth Engine image {self.image_id!r}: {exc}"
            ) from exc
        return from_engine(out)


__all__ = [
    "GeoToolzIOError",
    "LoadFromEE",
    "LoadFromSTAC",
    "ReadBounds",
    "ReadCenterCoords",
    "ReadHDF",
    "ReadNetCDF",
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
