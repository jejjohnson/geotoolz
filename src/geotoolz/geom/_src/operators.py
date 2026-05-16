"""Geometry Operators wrapping georeader's GeoTensor utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import rasterio.windows
from affine import Affine
from georeader import griddata, mosaic, rasterize, read, slices, vectorize
from georeader.geotensor import GeoTensor
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from shapely.ops import unary_union

from geotoolz.core import Operator


if TYPE_CHECKING:
    import geopandas as gpd
    from shapely.geometry.base import BaseGeometry


def _resampling(name: str | Resampling) -> Resampling:
    if isinstance(name, Resampling):
        return name
    aliases = {"bicubic": "cubic", "linear": "bilinear"}
    return Resampling[aliases.get(name, name)]


def _interpolation(name: str) -> str:
    aliases = {
        "cubic": "bicubic",
        "cubic_spline": "bicubic",
        "average": "bilinear",
    }
    return aliases.get(name, name)


def _center_offsets(
    current: tuple[int, int], target: tuple[int, int]
) -> tuple[int, int]:
    return ((current[0] - target[0]) // 2, (current[1] - target[1]) // 2)


def _assert_north_up(transform: Affine) -> None:
    if transform.b != 0 or transform.d != 0:
        raise ValueError("Stitch only supports north-up, non-rotated GeoTensors.")


def _feather_weights(shape: tuple[int, int], width: int) -> np.ndarray:
    height, width_px = shape
    if width <= 0:
        return np.ones(shape, dtype=np.float32)
    y = np.minimum(np.arange(height) + 1, np.arange(height, 0, -1))
    x = np.minimum(np.arange(width_px) + 1, np.arange(width_px, 0, -1))
    y = np.clip(y / width, 0.0, 1.0)
    x = np.clip(x / width, 0.0, 1.0)
    return np.outer(y, x).astype(np.float32)


class Reproject(Operator):
    """Reproject a GeoTensor to a destination CRS and optional resolution."""

    def __init__(
        self,
        *,
        dst_crs: str,
        resolution: tuple[float, float] | None = None,
        resampling: str = "bilinear",
    ) -> None:
        self.dst_crs = dst_crs
        self.resolution = resolution
        self.resampling = resampling

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        return read.read_to_crs(
            gt,
            self.dst_crs,
            resampling=_resampling(self.resampling),
            resolution_dst_crs=self.resolution,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "dst_crs": self.dst_crs,
            "resolution": self.resolution,
            "resampling": self.resampling,
        }


class ReprojectLike(Operator):
    """Reproject a GeoTensor onto another GeoTensor's grid."""

    def __init__(self, *, like: GeoTensor, resampling: str = "bilinear") -> None:
        self.like = like
        self.resampling = resampling

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        return read.read_reproject_like(
            gt,
            self.like,
            resampling=_resampling(self.resampling),
        )

    def get_config(self) -> dict[str, Any]:
        return {"like": self.like, "resampling": self.resampling}


class ResampleLike(ReprojectLike):
    """Resample a GeoTensor onto another GeoTensor's spatial grid."""


class Resize(Operator):
    """Resize a GeoTensor to a target spatial shape."""

    def __init__(
        self,
        *,
        shape: tuple[int, int],
        anti_aliasing: bool = True,
        resampling: str = "bilinear",
    ) -> None:
        self.shape = shape
        self.anti_aliasing = anti_aliasing
        self.resampling = resampling

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        return gt.resize(
            output_shape=self.shape,
            anti_aliasing=self.anti_aliasing,
            interpolation=_interpolation(self.resampling),
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "anti_aliasing": self.anti_aliasing,
            "resampling": self.resampling,
        }


class Resample(Operator):
    """Resample a GeoTensor to a target spatial resolution."""

    def __init__(
        self,
        *,
        resolution: tuple[float, float],
        resampling: str = "bilinear",
        anti_aliasing: bool = True,
    ) -> None:
        self.resolution = resolution
        self.resampling = resampling
        self.anti_aliasing = anti_aliasing

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        return gt.resize(
            resolution_dst=self.resolution,
            anti_aliasing=self.anti_aliasing,
            interpolation=_interpolation(self.resampling),
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "resolution": self.resolution,
            "resampling": self.resampling,
            "anti_aliasing": self.anti_aliasing,
        }


class PadTo(Operator):
    """Pad a GeoTensor up to a target spatial shape."""

    def __init__(
        self,
        *,
        shape: tuple[int, int],
        mode: str = "constant",
        fill: float | None = None,
    ) -> None:
        self.shape = shape
        self.mode = mode
        self.fill = fill

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        height, width = gt.shape[-2:]
        pad_y = max(self.shape[0] - height, 0)
        pad_x = max(self.shape[1] - width, 0)
        if pad_y == 0 and pad_x == 0:
            return gt
        pad_width = {
            "y": (pad_y // 2, pad_y - pad_y // 2),
            "x": (pad_x // 2, pad_x - pad_x // 2),
        }
        kwargs = {}
        if self.mode == "constant":
            kwargs["constant_values"] = self.fill
        return gt.pad(pad_width, mode=self.mode, **kwargs)

    def get_config(self) -> dict[str, Any]:
        return {"shape": self.shape, "mode": self.mode, "fill": self.fill}


class CropTo(Operator):
    """Crop a GeoTensor to a target spatial shape."""

    def __init__(self, *, shape: tuple[int, int], anchor: str = "center") -> None:
        self.shape = shape
        self.anchor = anchor

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        height, width = gt.shape[-2:]
        if self.shape[0] > height or self.shape[1] > width:
            raise ValueError(f"Cannot crop shape {(height, width)} to {self.shape}.")
        if self.anchor == "center":
            row_off, col_off = _center_offsets((height, width), self.shape)
        elif self.anchor == "upper_left":
            row_off, col_off = 0, 0
        else:
            raise ValueError("anchor must be 'center' or 'upper_left'.")
        window = rasterio.windows.Window(
            col_off=col_off,
            row_off=row_off,
            width=self.shape[1],
            height=self.shape[0],
        )
        return gt.read_from_window(window, boundless=False)

    def get_config(self) -> dict[str, Any]:
        return {"shape": self.shape, "anchor": self.anchor}


class CropToBounds(Operator):
    """Crop a GeoTensor to map bounds."""

    def __init__(
        self,
        *,
        bounds: tuple[float, float, float, float],
        crs: str | None = None,
    ) -> None:
        self.bounds = bounds
        self.crs = crs

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        bounds = self.bounds
        if self.crs is not None and CRS.from_user_input(
            self.crs
        ) != CRS.from_user_input(gt.crs):
            bounds = transform_bounds(self.crs, gt.crs, *bounds)
        window = rasterio.windows.from_bounds(*bounds, transform=gt.transform)
        window = window.round_offsets().round_lengths()
        return gt.read_from_window(window, boundless=False)

    def get_config(self) -> dict[str, Any]:
        return {"bounds": self.bounds, "crs": self.crs}


class Tile(Operator):
    """Split a GeoTensor into spatial tiles."""

    def __init__(
        self,
        *,
        size: tuple[int, int],
        stride: tuple[int, int] | None = None,
        include_incomplete: bool = True,
    ) -> None:
        self.size = size
        self.stride = stride
        self.include_incomplete = include_incomplete

    def _apply(self, gt: GeoTensor) -> list[GeoTensor]:
        overlap = None
        if self.stride is not None:
            overlap = (self.size[0] - self.stride[0], self.size[1] - self.stride[1])
            if overlap[0] < 0 or overlap[1] < 0:
                raise ValueError("stride must be less than or equal to size.")
        windows = slices.create_windows(
            gt.shape[-2:],
            self.size,
            overlap=overlap,
            include_incomplete=self.include_incomplete,
        )
        return [gt.read_from_window(window, boundless=True) for window in windows]

    def get_config(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "stride": self.stride,
            "include_incomplete": self.include_incomplete,
        }


class SlidingWindow(Tile):
    """Split a GeoTensor into overlapping sliding-window tiles."""

    def __init__(self, *, size: tuple[int, int], overlap: int = 0) -> None:
        self.overlap = overlap
        super().__init__(
            size=size,
            stride=(size[0] - overlap, size[1] - overlap),
            include_incomplete=True,
        )

    def get_config(self) -> dict[str, Any]:
        return {"size": self.size, "overlap": self.overlap}


class Stitch(Operator):
    """Stitch georeferenced tiles into one GeoTensor."""

    def __init__(
        self,
        *,
        blend: str = "average",
        feather_width: int = 16,
        target_shape: tuple[int, int] | None = None,
        target_transform: Affine | None = None,
        target_crs: str | None = None,
        fill: float | int | None = None,
    ) -> None:
        self.blend = blend
        self.feather_width = feather_width
        self.target_shape = target_shape
        self.target_transform = target_transform
        self.target_crs = target_crs
        self.fill = fill

    def _apply(self, tiles: list[GeoTensor]) -> GeoTensor:
        if not tiles:
            raise ValueError("Stitch requires at least one tile.")
        first = tiles[0]
        _assert_north_up(first.transform)
        transform, shape = self._target_grid(tiles)
        fill = first.fill_value_default if self.fill is None else self.fill
        dtype = first.dtype if self.blend in {"first", "max"} else np.float32
        out_shape = first.shape[:-2] + shape
        if self.blend == "max":
            values = np.full(out_shape, fill, dtype=dtype)
            filled = np.zeros(shape, dtype=bool)
            self._stitch_max(tiles, transform, values, filled)
        elif self.blend == "first":
            values = np.full(out_shape, fill, dtype=dtype)
            filled = np.zeros(shape, dtype=bool)
            self._stitch_first(tiles, transform, values, filled)
        elif self.blend in {"average", "feather"}:
            values = np.zeros(out_shape, dtype=dtype)
            weights = np.zeros(shape, dtype=np.float32)
            self._stitch_average(tiles, transform, values, weights)
            valid = weights > 0
            values[..., valid] /= weights[valid]
            values[..., ~valid] = fill
        else:
            raise ValueError("blend must be 'average', 'feather', 'max', or 'first'.")
        return GeoTensor(
            values,
            transform,
            self.target_crs or first.crs,
            fill_value_default=fill,
            attrs=first.attrs,
        )

    def _target_grid(
        self,
        tiles: list[GeoTensor],
    ) -> tuple[Affine, tuple[int, int]]:
        first = tiles[0]
        if self.target_transform is not None:
            transform = self.target_transform
        else:
            bounds = [tile.bounds for tile in tiles]
            minx = min(bound[0] for bound in bounds)
            maxy = max(bound[3] for bound in bounds)
            transform = Affine(
                first.transform.a, 0.0, minx, 0.0, first.transform.e, maxy
            )
        if self.target_shape is not None:
            return transform, self.target_shape
        bounds = [tile.bounds for tile in tiles]
        maxx = max(bound[2] for bound in bounds)
        miny = min(bound[1] for bound in bounds)
        height = round((transform.f - miny) / abs(transform.e))
        width = round((maxx - transform.c) / abs(transform.a))
        return transform, (height, width)

    def _target_slices(
        self,
        tile: GeoTensor,
        transform: Affine,
        shape: tuple[int, int],
    ) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
        row = round((tile.transform.f - transform.f) / transform.e)
        col = round((tile.transform.c - transform.c) / transform.a)
        height, width = tile.shape[-2:]
        row0 = max(row, 0)
        col0 = max(col, 0)
        row1 = min(row + height, shape[0])
        col1 = min(col + width, shape[1])
        tile_row0 = row0 - row
        tile_col0 = col0 - col
        tile_row1 = tile_row0 + (row1 - row0)
        tile_col1 = tile_col0 + (col1 - col0)
        return (slice(row0, row1), slice(col0, col1)), (
            slice(tile_row0, tile_row1),
            slice(tile_col0, tile_col1),
        )

    def _stitch_average(
        self,
        tiles: list[GeoTensor],
        transform: Affine,
        values: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        for tile in tiles:
            out_slices, tile_slices = self._target_slices(
                tile, transform, weights.shape
            )
            weight = (
                _feather_weights(tile.shape[-2:], self.feather_width)
                if self.blend == "feather"
                else np.ones(tile.shape[-2:], dtype=np.float32)
            )[tile_slices]
            values[..., out_slices[0], out_slices[1]] += (
                np.asarray(tile)[..., tile_slices[0], tile_slices[1]] * weight
            )
            weights[out_slices] += weight

    def _stitch_first(
        self,
        tiles: list[GeoTensor],
        transform: Affine,
        values: np.ndarray,
        filled: np.ndarray,
    ) -> None:
        for tile in tiles:
            out_slices, tile_slices = self._target_slices(tile, transform, filled.shape)
            mask = ~filled[out_slices]
            values_window = values[..., out_slices[0], out_slices[1]]
            tile_window = np.asarray(tile)[..., tile_slices[0], tile_slices[1]]
            values_window[..., mask] = tile_window[..., mask]
            filled[out_slices] |= mask

    def _stitch_max(
        self,
        tiles: list[GeoTensor],
        transform: Affine,
        values: np.ndarray,
        filled: np.ndarray,
    ) -> None:
        for tile in tiles:
            out_slices, tile_slices = self._target_slices(tile, transform, filled.shape)
            values_window = values[..., out_slices[0], out_slices[1]]
            tile_window = np.asarray(tile)[..., tile_slices[0], tile_slices[1]]
            mask = filled[out_slices]
            values_window[..., mask] = np.maximum(
                values_window[..., mask],
                tile_window[..., mask],
            )
            values_window[..., ~mask] = tile_window[..., ~mask]
            filled[out_slices] = True

    def get_config(self) -> dict[str, Any]:
        return {
            "blend": self.blend,
            "feather_width": self.feather_width,
            "target_shape": self.target_shape,
            "target_transform": self.target_transform,
            "target_crs": self.target_crs,
            "fill": self.fill,
        }


class Mosaic(Operator):
    """Mosaic multiple GeoTensors."""

    def __init__(self, *, method: str = "first", resampling: str = "bilinear") -> None:
        self.method = method
        self.resampling = resampling

    def _apply(self, gts: list[GeoTensor]) -> GeoTensor:
        if self.method == "first":
            return mosaic.spatial_mosaic(gts, resampling=_resampling(self.resampling))
        base = mosaic.spatial_mosaic(gts, resampling=_resampling(self.resampling))
        fill = base.fill_value_default
        arrays = []
        for gt in gts:
            aligned = read.read_reproject_like(
                gt,
                base,
                resampling=_resampling(self.resampling),
            )
            arr = np.asarray(aligned, dtype=np.float32)
            if fill is not None:
                arr = np.where(arr == fill, np.nan, arr)
            arrays.append(arr)
        stack = np.stack(arrays, axis=0)
        reducers = {
            "mean": np.nanmean,
            "average": np.nanmean,
            "median": np.nanmedian,
            "max": np.nanmax,
            "min": np.nanmin,
        }
        if self.method not in reducers:
            raise ValueError(
                "method must be 'first', 'mean', 'median', 'max', or 'min'."
            )
        values = reducers[self.method](stack, axis=0)
        if fill is not None:
            values = np.where(np.isnan(values), fill, values)
        return base.array_as_geotensor(values.astype(base.dtype, copy=False))

    def get_config(self) -> dict[str, Any]:
        return {"method": self.method, "resampling": self.resampling}


class Georeference(Operator):
    """Georeference swath data using a georeader GLT GeoTensor."""

    def __init__(self, *, glt: GeoTensor, valid_glt: np.ndarray | None = None) -> None:
        self.glt = glt
        self.valid_glt = valid_glt

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        return griddata.georreference(
            self.glt,
            np.asarray(gt),
            valid_glt=self.valid_glt,
            fill_value_default=gt.fill_value_default,
        )

    def get_config(self) -> dict[str, Any]:
        return {"glt": self.glt, "valid_glt": self.valid_glt}


class Rasterize(Operator):
    """Rasterize geometries onto the input GeoTensor grid."""

    def __init__(
        self,
        *,
        geometries: list[BaseGeometry] | gpd.GeoDataFrame,
        column: str | None = None,
        all_touched: bool = False,
        fill: float = 0.0,
    ) -> None:
        self.geometries = geometries
        self.column = column
        self.all_touched = all_touched
        self.fill = fill

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        return _rasterize_like(
            self.geometries,
            gt,
            column=self.column,
            fill=self.fill,
            all_touched=self.all_touched,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "geometries": self.geometries,
            "column": self.column,
            "all_touched": self.all_touched,
            "fill": self.fill,
        }


class RasterizeLike(Operator):
    """Rasterize geometries onto a stored reference GeoTensor grid."""

    def __init__(
        self,
        *,
        like: GeoTensor,
        geometries: gpd.GeoDataFrame,
        column: str | None = None,
        all_touched: bool = False,
        fill: float = 0.0,
    ) -> None:
        self.like = like
        self.geometries = geometries
        self.column = column
        self.all_touched = all_touched
        self.fill = fill

    def _apply(self, gt: GeoTensor | None = None) -> GeoTensor:
        return _rasterize_like(
            self.geometries,
            self.like,
            column=self.column,
            fill=self.fill,
            all_touched=self.all_touched,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "like": self.like,
            "geometries": self.geometries,
            "column": self.column,
            "all_touched": self.all_touched,
            "fill": self.fill,
        }


class Vectorize(Operator):
    """Vectorize non-zero regions of a mask GeoTensor."""

    def __init__(
        self,
        *,
        min_area: float = 25.5,
        simplify_tolerance: float | None = None,
    ) -> None:
        self.min_area = min_area
        self.simplify_tolerance = simplify_tolerance

    def _apply(self, gt: GeoTensor) -> list[BaseGeometry]:
        polygons = vectorize.get_polygons(gt, min_area=self.min_area, tolerance=0.0)
        if self.simplify_tolerance is None:
            return polygons
        return [polygon.simplify(self.simplify_tolerance) for polygon in polygons]

    def get_config(self) -> dict[str, Any]:
        return {
            "min_area": self.min_area,
            "simplify_tolerance": self.simplify_tolerance,
        }


def _rasterize_like(
    geometries: list[BaseGeometry] | gpd.GeoDataFrame,
    like: GeoTensor,
    *,
    column: str | None,
    fill: float,
    all_touched: bool,
) -> GeoTensor:
    if hasattr(geometries, "geometry"):
        dataframe = geometries
        column_name = column or "__geotoolz_value__"
        if column is None:
            dataframe = geometries.assign(**{column_name: 1.0})
        return rasterize.rasterize_geopandas_like(
            dataframe,
            like,
            column=column_name,
            fill=fill,
            all_touched=all_touched,
        )
    geometry = unary_union(geometries)
    return rasterize.rasterize_geometry_like(
        geometry,
        like,
        value=1,
        fill=fill,
        all_touched=all_touched,
    )
