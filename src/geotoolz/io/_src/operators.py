"""I/O operators wrapping georeader read and write primitives."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import rasterio
from georeader import read
from georeader.geotensor import GeoTensor
from georeader.rasterio_reader import RasterioReader
from rasterio.errors import RasterioIOError
from rasterio.io import DatasetReaderBase
from rasterio.windows import Window
from shapely.geometry import MultiPolygon, Polygon, box

from geotoolz.core import Operator


if TYPE_CHECKING:
    from affine import Affine


Source = str | PathLike[str] | Any
Bounds = tuple[float, float, float, float]
Resolution = float | tuple[float, float]


class GeoToolzIOError(RuntimeError):
    """Raised when a geotoolz I/O operator cannot read or write data."""


class SourceOperator(Operator):
    """Operator that produces a ``GeoTensor`` without an input carrier."""


class SinkOperator(Operator):
    """Terminal operator that consumes a ``GeoTensor`` and writes side effects."""

    _terminal = True


def _window_config(window: Window) -> tuple[float, float, float, float]:
    return (window.col_off, window.row_off, window.width, window.height)


def _coerce_window(window: Window | tuple[float, float, float, float]) -> Window:
    if isinstance(window, Window):
        return window
    col_off, row_off, width, height = window
    return Window(col_off=col_off, row_off=row_off, width=width, height=height)


def _coerce_source(src: Source, indexes: list[int] | None = None) -> Any:
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
    """Read a pixel window from a raster source."""

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
    """Read data intersecting geographic bounds from a raster source."""

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
    """Read a window centered on map coordinates from a raster source."""

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
    """Read a web-map tile from a raster source."""

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
    """Read data intersecting a polygon from a raster source."""

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
    """Read a source and reproject it onto another GeoData grid."""

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
            "like": self.like,
            "indexes": self.indexes,
            "resolution": self.resolution,
        }


class ReadToCRS(SourceOperator):
    """Read a source and reproject it to a target CRS."""

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
    """Write a ``GeoTensor`` as a Cloud Optimized GeoTIFF."""

    def __init__(
        self,
        *,
        path: str | PathLike[str],
        profile: dict[str, Any] | None = None,
        overviews: list[int] | None = None,
        compress: str = "deflate",
    ) -> None:
        self.path = Path(path)
        self.profile = profile
        self.overviews = [2, 4, 8, 16] if overviews is None else overviews
        self.compress = compress

    def _apply(self, gt: GeoTensor) -> None:
        from georeader import save

        profile = {"compress": self.compress}
        if self.profile is not None:
            profile.update(self.profile)
        try:
            save.save_cog(gt, str(self.path), profile=profile)
        except (FileNotFoundError, OSError, RasterioIOError) as exc:
            raise GeoToolzIOError(f"Unable to write COG {self.path!s}: {exc}") from exc
        return None

    def get_config(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "profile": self.profile,
            "overviews": self.overviews,
            "compress": self.compress,
        }


class WriteGeoTIFF(SinkOperator):
    """Write a ``GeoTensor`` as a standard GeoTIFF."""

    def __init__(
        self,
        *,
        path: str | PathLike[str],
        profile: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.profile = profile

    def _apply(self, gt: GeoTensor) -> None:
        arr = np.asarray(gt.values)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        if arr.ndim != 3:
            raise GeoToolzIOError(
                f"GeoTIFF output expects 2D or 3D data, found shape {arr.shape!r}."
            )

        write_profile: dict[str, Any] = {
            "driver": "GTiff",
            "height": arr.shape[1],
            "width": arr.shape[2],
            "count": arr.shape[0],
            "dtype": str(arr.dtype),
            "crs": gt.crs,
            "transform": gt.transform,
        }
        if gt.fill_value_default is not None:
            write_profile["nodata"] = gt.fill_value_default
        if self.profile is not None:
            write_profile.update(self.profile)

        try:
            with rasterio.open(self.path, "w", **write_profile) as dst:
                dst.write(arr)
        except (FileNotFoundError, OSError, RasterioIOError) as exc:
            raise GeoToolzIOError(
                f"Unable to write GeoTIFF {self.path!s}: {exc}"
            ) from exc
        return None

    def get_config(self) -> dict[str, Any]:
        return {"path": str(self.path), "profile": self.profile}


class WriteZarr(SinkOperator):
    """Write a ``GeoTensor`` array and spatial metadata to a Zarr store."""

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
        chunk_shape = None
        if self.chunks is not None:
            axis_names = ("band", "y", "x")[-np.asarray(gt.values).ndim :]
            chunk_shape = tuple(
                self.chunks.get(name, size)
                for name, size in zip(axis_names, gt.shape, strict=True)
            )
        group.create_array("values", data=np.asarray(gt.values), chunks=chunk_shape)
        group.attrs["crs"] = str(gt.crs)
        group.attrs["transform"] = tuple(gt.transform)
        group.attrs["fill_value_default"] = gt.fill_value_default
        return None

    def get_config(self) -> dict[str, Any]:
        return {"store": self.store, "group": self.group, "chunks": self.chunks}


class LoadFromSTAC(SourceOperator):
    """Load a raster asset from a STAC item."""

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
        return {
            "item": self.item,
            "asset_key": self.asset_key,
            "bounds": self.bounds,
            "resolution": self.resolution,
        }


class LoadFromEE(SourceOperator):
    """Load an Earth Engine image into a ``GeoTensor``."""

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
            from affine import Affine
            from georeader.readers.ee_image import export_image
        except ImportError as exc:
            raise GeoToolzIOError(
                "LoadFromEE requires georeader's Earth Engine dependencies."
            ) from exc

        xmin, ymax = self.bounds[0], self.bounds[3]
        transform: Affine = Affine(self.scale, 0.0, xmin, 0.0, -self.scale, ymax)
        ee_module = export_image.__globals__.get("ee")
        ee_exception = getattr(ee_module, "EEException", RuntimeError)
        try:
            return export_image(
                self.image_id,
                geometry=box(*self.bounds),
                transform=transform,
                crs=self.crs,
                bands_gee=[] if self.bands is None else self.bands,
                resolution_dst=self.scale,
            )
        except (ee_exception, RuntimeError, ValueError, OSError) as exc:
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
