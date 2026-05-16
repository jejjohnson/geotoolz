"""Tier-B Operators — carrier-aware geometry transforms.

Each Operator wraps either a primitive in
:mod:`geotoolz.geom._src.array` or a `georeader` utility
(``read.read_to_crs``, ``mosaic.spatial_mosaic``, ``rasterize.*``,
``vectorize.get_polygons``, ``griddata.georreference``). The two-tier
discipline (``_src/array.py`` for numpy primitives, ``_src/operators.py``
for ``GeoTensor`` wrappers) mirrors the rest of geotoolz.

Operators that carry concrete ``GeoTensor`` / ``GeoDataFrame`` /
``Affine`` references in their constructor — `ReprojectLike`,
`ResampleLike`, `RasterizeLike`, `Rasterize`, `Georeference`, and
`Stitch` with a pinned target grid — set
``forbid_in_yaml = True`` because those objects are not JSON-safe and
hydra-zen ``builds()`` cannot recreate them from ``get_config()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import rasterio.windows
from affine import Affine
from georeader import griddata, mosaic, rasterize, read, slices, vectorize
from georeader.geotensor import GeoTensor
from pipekit import Operator
from pyproj import CRS
from rasterio.warp import transform_bounds
from scipy.ndimage import shift as ndi_shift
from shapely.ops import unary_union
from skimage.registration import (
    optical_flow_ilk,
    optical_flow_tvl1,
    phase_cross_correlation,
)

from geotoolz.geom._src.array import (
    center_offsets,
    feather_weights,
    is_north_up,
    resolve_interpolation,
    resolve_resampling,
    target_slices,
    valid_pixel_mask,
)


if TYPE_CHECKING:
    import geopandas as gpd
    from shapely.geometry.base import BaseGeometry


def _registration_band(values: np.ndarray, band: int | str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 2:
        return arr
    if isinstance(band, str):
        raise ValueError(
            "string band selection is not available for raw GeoTensor arrays"
        )
    return np.take(arr, int(band), axis=0)


class Reproject(Operator):
    """Reproject a `GeoTensor` to a destination CRS (and optional resolution).

    Thin Operator wrapping :func:`georeader.read.read_to_crs`. The
    destination grid is anchored to the input's bounding box reprojected
    into ``dst_crs``; the resolution defaults to the input's pixel size
    expressed in the destination CRS.

    Args:
        dst_crs: Target CRS as a string (EPSG code, WKT, PROJ string, or
            anything :class:`pyproj.CRS.from_user_input` accepts).
        resolution: Optional ``(pixel_size_x, pixel_size_y)`` in
            destination-CRS units. ``None`` lets georeader derive it
            from the source pixel size.
        resampling: One of ``"nearest"``, ``"bilinear"`` /
            ``"linear"``, ``"cubic"`` / ``"bicubic"``,
            ``"cubic_spline"``, ``"lanczos"``, ``"average"``, ``"mode"``.

    Examples:
        >>> import geotoolz as gz
        >>> # Reproject a UTM scene into geographic coords for plotting.
        >>> reproj = gz.geom.Reproject(
        ...     dst_crs="EPSG:4326", resampling="bilinear"
        ... )
        >>> geographic = reproj(utm_scene)
    """

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
            resampling=resolve_resampling(self.resampling),
            resolution_dst_crs=self.resolution,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "dst_crs": self.dst_crs,
            "resolution": (None if self.resolution is None else list(self.resolution)),
            "resampling": self.resampling,
        }


class ReprojectLike(Operator):
    """Reproject a `GeoTensor` onto another `GeoTensor`'s CRS + grid.

    Wraps :func:`georeader.read.read_reproject_like`: the output has the
    same CRS, transform, and spatial shape as ``like``. Useful for
    aligning auxiliary data (DEMs, masks, predictions from another
    sensor) onto a reference scene grid before downstream pipelines.

    Note:
        ``like`` is a concrete ``GeoTensor``, so this Operator is
        flagged ``forbid_in_yaml = True``.

    Args:
        like: The reference `GeoTensor` whose grid (CRS + transform +
            spatial shape) is matched.
        resampling: See :class:`Reproject`.

    Examples:
        >>> import geotoolz as gz
        >>> # Align a coarse cloud mask onto the scene grid.
        >>> aligned_mask = gz.geom.ReprojectLike(
        ...     like=scene, resampling="nearest"
        ... )(coarse_mask)
    """

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(self, *, like: GeoTensor, resampling: str = "bilinear") -> None:
        self.like = like
        self.resampling = resampling

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        return read.read_reproject_like(
            gt,
            self.like,
            resampling=resolve_resampling(self.resampling),
        )

    def get_config(self) -> dict[str, Any]:
        return {"like": repr(self.like), "resampling": self.resampling}


class ResampleLike(ReprojectLike):
    """Resample a `GeoTensor` onto another `GeoTensor`'s spatial grid.

    Alias of :class:`ReprojectLike`. The intent of the rename is purely
    pipeline-readability: in workflows where both inputs already share a
    CRS, ``ResampleLike`` reads as "match this grid" rather than
    "warp into a new projection".

    Examples:
        >>> import geotoolz as gz
        >>> # Downsample a 10 m S2 band onto a 20 m reference grid.
        >>> rs = gz.geom.ResampleLike(like=s2_20m, resampling="average")
        >>> b04_20m = rs(b04_10m)
    """


class PhaseAlign(Operator):
    """Sub-pixel image registration via phase cross-correlation."""

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self,
        *,
        reference: GeoTensor,
        upsample_factor: int = 10,
        band: int | str = 0,
        apply: bool = True,
    ) -> None:
        self.reference = reference
        self.upsample_factor = upsample_factor
        self.band = band
        self.apply = apply

    def _apply(self, gt: GeoTensor) -> GeoTensor | tuple[float, float, float]:
        arr = np.asarray(gt)
        shift, error, _phase = phase_cross_correlation(
            _registration_band(np.asarray(self.reference), self.band),
            _registration_band(arr, self.band),
            upsample_factor=self.upsample_factor,
        )
        shift_y = float(shift[0])
        shift_x = float(shift[1])
        if not self.apply:
            return shift_y, shift_x, float(error)
        shifted = ndi_shift(
            arr,
            shift=(0.0, shift_y, shift_x) if arr.ndim == 3 else (shift_y, shift_x),
            order=1,
            mode="nearest",
        )
        return GeoTensor(
            shifted,
            transform=gt.transform * Affine.translation(-shift_x, -shift_y),
            crs=gt.crs,
            fill_value_default=gt.fill_value_default,
            attrs=gt.attrs,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "reference": {
                "shape": list(np.asarray(self.reference).shape),
                "dtype": str(np.asarray(self.reference).dtype),
            },
            "upsample_factor": self.upsample_factor,
            "band": self.band,
            "apply": self.apply,
        }


class OpticalFlowTVL1(Operator):
    """Dense per-pixel displacement via TV-L1 optical flow."""

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(self, *, reference: GeoTensor, band: int | str = 0) -> None:
        self.reference = reference
        self.band = band

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        flow = np.asarray(
            optical_flow_tvl1(
                _registration_band(np.asarray(self.reference), self.band),
                _registration_band(np.asarray(gt), self.band),
            )
        )
        return gt.array_as_geotensor(flow)

    def get_config(self) -> dict[str, Any]:
        return {
            "reference": {
                "shape": list(np.asarray(self.reference).shape),
                "dtype": str(np.asarray(self.reference).dtype),
            },
            "band": self.band,
        }


class OpticalFlowILK(OpticalFlowTVL1):
    """Dense per-pixel displacement via iterative Lucas-Kanade optical flow."""

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        flow = np.asarray(
            optical_flow_ilk(
                _registration_band(np.asarray(self.reference), self.band),
                _registration_band(np.asarray(gt), self.band),
            )
        )
        return gt.array_as_geotensor(flow)


class Resize(Operator):
    """Resize a `GeoTensor` to a target spatial shape.

    Delegates to :meth:`georeader.geotensor.GeoTensor.resize`, which
    re-bases the affine transform so the geographic extent is preserved
    while the pixel grid is densified or coarsened. ``anti_aliasing``
    applies a Gaussian pre-filter when downscaling (recommended).

    Note:
        The interpolation name is mapped onto
        :func:`skimage.transform.resize`'s conventions
        (``"bilinear"``, ``"bicubic"``, ``"nearest"``).

    Args:
        shape: Target spatial shape ``(H, W)`` in pixels.
        anti_aliasing: Whether to anti-alias before downscaling.
            Default ``True``.
        resampling: Interpolation mode. Default ``"bilinear"``.

    Examples:
        >>> import geotoolz as gz
        >>> tile = gz.geom.Resize(shape=(256, 256))(scene)
    """

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
            interpolation=resolve_interpolation(self.resampling),
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "anti_aliasing": self.anti_aliasing,
            "resampling": self.resampling,
        }


class Resample(Operator):
    """Resample a `GeoTensor` to a target spatial resolution.

    Delegates to :meth:`georeader.geotensor.GeoTensor.resize` via the
    ``resolution_dst`` argument. The output shape is implied by the
    ratio of the input pixel size to the requested resolution.

    Args:
        resolution: Target ``(pixel_size_x, pixel_size_y)`` in
            input-CRS units.
        resampling: Interpolation mode. Default ``"bilinear"``.
        anti_aliasing: Whether to anti-alias before downscaling.
            Default ``True``.

    Examples:
        >>> import geotoolz as gz
        >>> # Downsample a 10 m raster to 30 m.
        >>> coarse = gz.geom.Resample(resolution=(30.0, 30.0))(fine_10m)
    """

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
            interpolation=resolve_interpolation(self.resampling),
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "resolution": list(self.resolution),
            "resampling": self.resampling,
            "anti_aliasing": self.anti_aliasing,
        }


class PadTo(Operator):
    """Pad a `GeoTensor` up to a target spatial shape.

    Pads the spatial axes (``"y"``, ``"x"``) symmetrically so the output
    has *at least* ``shape`` pixels along each spatial axis. If the
    input already exceeds the target on an axis, that axis is left
    alone (no truncation — pair with :class:`CropTo` for that).

    The affine transform is updated by `GeoTensor.pad` so the
    *geographic* origin of the existing data does not move; new rows
    appear to the top/bottom and new columns to the left/right.

    Args:
        shape: Target minimum spatial shape ``(H, W)``.
        mode: Numpy pad mode. ``"constant"`` (default), ``"edge"``,
            ``"reflect"``, ``"symmetric"``, ...
        fill: Constant value when ``mode == "constant"``. ``None``
            falls back to the carrier's ``fill_value_default``.

    Examples:
        >>> import geotoolz as gz
        >>> # Pad to a multiple of 256 before tiling.
        >>> snug = gz.geom.PadTo(shape=(2048, 2048), fill=0)(scene)
    """

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
        kwargs: dict[str, Any] = {}
        if self.mode == "constant":
            kwargs["constant_values"] = self.fill
        return gt.pad(pad_width, mode=self.mode, **kwargs)

    def get_config(self) -> dict[str, Any]:
        return {"shape": list(self.shape), "mode": self.mode, "fill": self.fill}


class CropTo(Operator):
    """Crop a `GeoTensor` to a target spatial shape.

    Anchors either at the centre (default) or upper-left corner of the
    input. The affine transform is updated through
    :meth:`georeader.geotensor.GeoTensor.read_from_window` so the
    geographic extent of the surviving pixels is preserved.

    Args:
        shape: Target spatial shape ``(H, W)``. Must be ``<=`` the
            input shape; oversize crops raise ``ValueError``.
        anchor: ``"center"`` (default) or ``"upper_left"``.

    Examples:
        >>> import geotoolz as gz
        >>> # Centre-crop a 1024×1024 scene to a 512×512 chip.
        >>> chip = gz.geom.CropTo(shape=(512, 512))(scene)
    """

    def __init__(self, *, shape: tuple[int, int], anchor: str = "center") -> None:
        self.shape = shape
        self.anchor = anchor

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        height, width = gt.shape[-2:]
        if self.shape[0] > height or self.shape[1] > width:
            raise ValueError(f"Cannot crop shape {(height, width)} to {self.shape}.")
        if self.anchor == "center":
            row_off, col_off = center_offsets((height, width), self.shape)
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
        return {"shape": list(self.shape), "anchor": self.anchor}


class CropToBounds(Operator):
    """Crop a `GeoTensor` to a geographic bounding box.

    Computes the pixel window covering ``bounds`` via
    :func:`rasterio.windows.from_bounds` (after reprojecting ``bounds``
    into the carrier's CRS if ``crs`` differs from it) and reads the
    intersection with the carrier. Non-intersecting bounds raise
    :class:`rasterio.windows.WindowError` via
    :meth:`georeader.geotensor.GeoTensor.read_from_window`.

    Args:
        bounds: ``(minx, miny, maxx, maxy)`` in CRS units of ``crs``.
        crs: CRS of ``bounds``. ``None`` (default) means "same as the
            carrier".

    Examples:
        >>> import geotoolz as gz
        >>> # Crop a global mosaic to a small AOI in geographic coords.
        >>> aoi = gz.geom.CropToBounds(
        ...     bounds=(-10.0, 35.0, -8.0, 36.5), crs="EPSG:4326"
        ... )
        >>> patch = aoi(global_scene)
    """

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
        return {"bounds": list(self.bounds), "crs": self.crs}


class Tile(Operator):
    """Split a `GeoTensor` into spatial tiles.

    Wraps :func:`georeader.slices.create_windows` and then reads each
    window with ``boundless=True`` so the trailing-edge tiles match
    ``size`` exactly. Out-of-bounds areas are padded with the carrier's
    ``fill_value_default``; use :func:`Stitch` (which masks the
    sentinel) to recover the original extent.

    Args:
        size: Tile spatial shape ``(H, W)``.
        stride: Step between consecutive tile origins ``(sy, sx)``. The
            implied overlap is ``size - stride``. ``None`` (default)
            means non-overlapping tiles (``stride == size``).
        include_incomplete: Whether to keep edge tiles that don't fully
            fit (still emitted at ``size`` shape because of
            ``boundless=True``).

    Examples:
        >>> import geotoolz as gz
        >>> # 512x512 tiles with 64-pixel overlap.
        >>> tiles = gz.geom.Tile(size=(512, 512), stride=(448, 448))(scene)
    """

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
        overlap: tuple[int, int] | None = None
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
            "size": list(self.size),
            "stride": None if self.stride is None else list(self.stride),
            "include_incomplete": self.include_incomplete,
        }


class SlidingWindow(Tile):
    """Split a `GeoTensor` into overlapping sliding-window tiles.

    Convenience subclass of :class:`Tile` that takes a scalar
    ``overlap`` (in pixels) and derives the stride. Equivalent to
    ``Tile(size=size, stride=(size[0] - overlap, size[1] - overlap))``.

    Args:
        size: Tile spatial shape ``(H, W)``.
        overlap: Overlap in pixels along both axes. Default ``0``.

    Examples:
        >>> import geotoolz as gz
        >>> # 256x256 tiles, 32-pixel overlap on every side.
        >>> tiles = gz.geom.SlidingWindow(size=(256, 256), overlap=32)(scene)
    """

    def __init__(self, *, size: tuple[int, int], overlap: int = 0) -> None:
        self.overlap = overlap
        super().__init__(
            size=size,
            stride=(size[0] - overlap, size[1] - overlap),
            include_incomplete=True,
        )

    def get_config(self) -> dict[str, Any]:
        return {"size": list(self.size), "overlap": self.overlap}


class Stitch(Operator):
    r"""Stitch georeferenced tiles back into one `GeoTensor`.

    Blends a ``list[GeoTensor]`` (typically the output of :class:`Tile`)
    onto a unioning north-up grid, masking out the
    ``fill_value_default`` sentinels that :class:`Tile` introduces
    along the bottom / right edges with ``boundless=True``.

    Supported blend modes:

    - ``"average"`` — unweighted mean of overlapping pixels.
    - ``"feather"`` — overlap-add with cosine-edge weights, kernel from
      :func:`geotoolz.geom._src.array.feather_weights`.
    - ``"first"`` — first tile wins on overlap.
    - ``"max"`` — element-wise maximum across overlapping tiles.

    The mathematical model for ``"average"`` / ``"feather"`` is the
    standard overlap-add reconstruction:

    .. math::

        \hat{x}_{ij} \;=\; \frac{\sum_k w_k(i, j)\, m_k(i, j)\, x_k(i, j)}
                                {\sum_k w_k(i, j)\, m_k(i, j)}

    where ``m_k`` is the validity mask of tile ``k`` and ``w_k`` is the
    weight kernel (1 everywhere for ``"average"``, the feather kernel
    otherwise). Pixels with no valid contributor are filled with
    ``fill`` (default: the first tile's ``fill_value_default``).

    Note:
        Set ``forbid_in_yaml = True`` when ``target_transform`` is
        supplied — :class:`affine.Affine` is not JSON-safe. The auto-
        derived (bbox-union) path round-trips cleanly through hydra-zen.

    Args:
        blend: Blend mode (see above). Default ``"average"``.
        feather_width: Pixel ramp width for ``"feather"`` blending.
        target_shape: Optional ``(H, W)`` override for the output grid.
        target_transform: Optional :class:`affine.Affine` override for
            the output grid origin.
        target_crs: Optional CRS override for the output. Defaults to
            the first tile's CRS.
        fill: Output fill value. ``None`` (default) inherits from the
            first tile.

    Examples:
        >>> import geotoolz as gz
        >>> # Round-trip a tiled inference pipeline.
        >>> tiles = gz.geom.Tile(size=(256, 256))(scene)
        >>> predictions = [model(tile) for tile in tiles]
        >>> stitched = gz.geom.Stitch(blend="feather", feather_width=16)(
        ...     predictions
        ... )
    """

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
        if self.blend not in {"average", "feather", "max", "first"}:
            raise ValueError("blend must be 'average', 'feather', 'max', or 'first'.")
        first = tiles[0]
        for index, tile in enumerate(tiles):
            if not is_north_up(tile.transform):
                raise ValueError(
                    f"Stitch only supports north-up, non-rotated GeoTensors; "
                    f"tile {index} has a rotated/sheared transform."
                )
        transform, shape = self._target_grid(tiles)
        fill = first.fill_value_default if self.fill is None else self.fill
        dtype = first.dtype if self.blend in {"first", "max"} else np.float32
        out_shape = first.shape[:-2] + shape
        if self.blend == "max":
            values = np.full(out_shape, fill, dtype=dtype)
            filled = np.zeros(shape, dtype=bool)
            self._stitch_max(tiles, transform, values, filled, fill)
        elif self.blend == "first":
            values = np.full(out_shape, fill, dtype=dtype)
            filled = np.zeros(shape, dtype=bool)
            self._stitch_first(tiles, transform, values, filled, fill)
        else:  # average / feather
            values = np.zeros(out_shape, dtype=dtype)
            weights = np.zeros(shape, dtype=np.float32)
            self._stitch_average(tiles, transform, values, weights, fill)
            valid = weights > 0
            values[..., valid] /= weights[valid]
            values[..., ~valid] = fill
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

    def _stitch_average(
        self,
        tiles: list[GeoTensor],
        transform: Affine,
        values: np.ndarray,
        weights: np.ndarray,
        fill: float | int | None,
    ) -> None:
        for tile in tiles:
            arr = np.asarray(tile)
            out_slices, tile_slices = target_slices(
                tile.transform, tile.shape[-2:], transform, weights.shape
            )
            mask_tile = valid_pixel_mask(arr, fill)[tile_slices]
            kernel = (
                feather_weights(tile.shape[-2:], self.feather_width)
                if self.blend == "feather"
                else np.ones(tile.shape[-2:], dtype=np.float32)
            )[tile_slices]
            weight = kernel * mask_tile.astype(np.float32)
            values[..., out_slices[0], out_slices[1]] += (
                arr[..., tile_slices[0], tile_slices[1]] * weight
            )
            weights[out_slices] += weight

    def _stitch_first(
        self,
        tiles: list[GeoTensor],
        transform: Affine,
        values: np.ndarray,
        filled: np.ndarray,
        fill: float | int | None,
    ) -> None:
        for tile in tiles:
            arr = np.asarray(tile)
            out_slices, tile_slices = target_slices(
                tile.transform, tile.shape[-2:], transform, filled.shape
            )
            mask_tile = valid_pixel_mask(arr, fill)[tile_slices]
            mask = (~filled[out_slices]) & mask_tile
            values_window = values[..., out_slices[0], out_slices[1]]
            tile_window = arr[..., tile_slices[0], tile_slices[1]]
            values_window[..., mask] = tile_window[..., mask]
            filled[out_slices] |= mask

    def _stitch_max(
        self,
        tiles: list[GeoTensor],
        transform: Affine,
        values: np.ndarray,
        filled: np.ndarray,
        fill: float | int | None,
    ) -> None:
        for tile in tiles:
            arr = np.asarray(tile)
            out_slices, tile_slices = target_slices(
                tile.transform, tile.shape[-2:], transform, filled.shape
            )
            mask_tile = valid_pixel_mask(arr, fill)[tile_slices]
            values_window = values[..., out_slices[0], out_slices[1]]
            tile_window = arr[..., tile_slices[0], tile_slices[1]]
            both = filled[out_slices] & mask_tile
            only_tile = (~filled[out_slices]) & mask_tile
            values_window[..., both] = np.maximum(
                values_window[..., both],
                tile_window[..., both],
            )
            values_window[..., only_tile] = tile_window[..., only_tile]
            filled[out_slices] |= mask_tile

    def get_config(self) -> dict[str, Any]:
        return {
            "blend": self.blend,
            "feather_width": self.feather_width,
            "target_shape": (
                None if self.target_shape is None else list(self.target_shape)
            ),
            "target_transform": (
                None
                if self.target_transform is None
                else list(self.target_transform)[:6]
            ),
            "target_crs": self.target_crs,
            "fill": self.fill,
        }


class Mosaic(Operator):
    r"""Mosaic multiple `GeoTensor`s onto a single grid.

    For ``method="first"`` this is a thin wrapper around
    :func:`georeader.mosaic.spatial_mosaic`: rasters are processed in
    order and the first valid pixel wins. For all other methods the
    function is still used to compute the *frame* (union of extents
    + first raster's CRS / dtype), then every input is reprojected onto
    that frame and the per-pixel reduction is applied:

    .. math::

        \hat{x}_{ij} \;=\; \mathrm{agg}\!\left(
            \{ x^{(k)}_{ij} \;|\; x^{(k)}_{ij} \neq \text{fill} \}
        \right)

    with ``agg`` chosen from ``mean``, ``median``, ``max``, ``min``.

    Args:
        method: ``"first"`` (default), ``"mean"`` / ``"average"``,
            ``"median"``, ``"max"``, ``"min"``.
        resampling: Resampling mode used for reprojection onto the
            mosaic frame.

    Examples:
        >>> import geotoolz as gz
        >>> # Median-composite a stack of overlapping S2 scenes.
        >>> composite = gz.geom.Mosaic(method="median")(s2_scene_list)
    """

    _REDUCERS: ClassVar[dict[str, Any]] = {
        "mean": np.nanmean,
        "average": np.nanmean,
        "median": np.nanmedian,
        "max": np.nanmax,
        "min": np.nanmin,
    }

    def __init__(self, *, method: str = "first", resampling: str = "bilinear") -> None:
        self.method = method
        self.resampling = resampling

    def _apply(self, gts: list[GeoTensor]) -> GeoTensor:
        if self.method == "first":
            return mosaic.spatial_mosaic(
                gts, resampling=resolve_resampling(self.resampling)
            )
        if self.method not in self._REDUCERS:
            raise ValueError(
                "method must be 'first', 'mean', 'median', 'max', or 'min'."
            )
        base = mosaic.spatial_mosaic(
            gts, resampling=resolve_resampling(self.resampling)
        )
        fill = base.fill_value_default
        arrays = []
        for gt in gts:
            aligned = read.read_reproject_like(
                gt,
                base,
                resampling=resolve_resampling(self.resampling),
            )
            arr = np.asarray(aligned, dtype=np.float32)
            if fill is not None:
                arr = np.where(arr == fill, np.nan, arr)
            arrays.append(arr)
        stack = np.stack(arrays, axis=0)
        with np.errstate(invalid="ignore"):
            values = self._REDUCERS[self.method](stack, axis=0)
        if fill is not None:
            values = np.where(np.isnan(values), fill, values)
        return base.array_as_geotensor(values.astype(base.dtype, copy=False))

    def get_config(self) -> dict[str, Any]:
        return {"method": self.method, "resampling": self.resampling}


class Georeference(Operator):
    """Georeference swath data using a `georeader` GLT `GeoTensor`.

    Thin wrapper around :func:`georeader.griddata.georreference` (note
    the upstream spelling). The GLT (Geolocation Lookup Table) maps
    output-grid pixels to source-sensor pixels for fast, exact
    orthorectification with no resampling artifacts.

    Args:
        glt: GLT `GeoTensor` of shape ``(2, H_out, W_out)``. ``glt[0]``
            holds source columns; ``glt[1]`` holds source rows.
        valid_glt: Optional boolean mask of valid GLT pixels.

    Note:
        Carries a concrete `GeoTensor` and (optionally) a numpy array
        reference, so flagged ``forbid_in_yaml = True``.

    Examples:
        >>> import geotoolz as gz
        >>> # Orthorectify a hyperspectral scene predicted in sensor space.
        >>> ortho = gz.geom.Georeference(glt=glt)(prediction_swath)
    """

    forbid_in_yaml: ClassVar[bool] = True

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
        return {"glt": repr(self.glt), "valid_glt": None}


class Rasterize(Operator):
    """Rasterize geometries onto the input `GeoTensor`'s grid.

    Burns ``geometries`` into a fresh array aligned with the input's
    CRS and transform. Accepts either:

    - a ``list`` of shapely geometries (each gets value ``1``); or
    - a :class:`geopandas.GeoDataFrame` (the ``column`` argument
      selects which attribute supplies the burn-in value, default
      ``1.0``).

    Delegates to :func:`georeader.rasterize.rasterize_geopandas_like`
    or :func:`georeader.rasterize.rasterize_geometry_like`.

    Note:
        Carries Python-level geometry objects (shapely / geopandas),
        so flagged ``forbid_in_yaml = True``.

    Args:
        geometries: Shapely geometries or a `GeoDataFrame`.
        column: For `GeoDataFrame` inputs, the column with burn-in
            values. ``None`` means "burn ``1``".
        all_touched: Whether to mark every pixel the geometry touches
            (vs. only those whose centre is inside).
        fill: Background value. Default ``0``.

    Examples:
        >>> import geotoolz as gz
        >>> from shapely.geometry import box
        >>> aoi_mask = gz.geom.Rasterize(geometries=[box(*aoi_bounds)])(scene)
    """

    forbid_in_yaml: ClassVar[bool] = True

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
            "geometries": f"<{type(self.geometries).__name__} n={len(self.geometries)}>",
            "column": self.column,
            "all_touched": self.all_touched,
            "fill": self.fill,
        }


class RasterizeLike(Operator):
    """Rasterize geometries onto a stored reference grid.

    Same as :class:`Rasterize` but the reference grid is pinned at
    construction (``like``) rather than supplied per call. The
    operator takes *no* positional input (it is a producer), or accepts
    a `GeoTensor` that is ignored — useful as the head of a `Sequential`
    pipeline that consumes the burned mask.

    Note:
        Flagged ``forbid_in_yaml = True``.

    Args:
        like: Reference `GeoTensor` whose CRS + transform + shape define
            the output grid.
        geometries: Shapely geometries or a `GeoDataFrame`.
        column: Attribute column for the burn-in value, or ``None``.
        all_touched: As in :class:`Rasterize`.
        fill: Background value.

    Examples:
        >>> import geotoolz as gz
        >>> burner = gz.geom.RasterizeLike(
        ...     like=ref_scene, geometries=polygons_gdf, column="class_id"
        ... )
        >>> labels = burner()
    """

    forbid_in_yaml: ClassVar[bool] = True

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
            "like": repr(self.like),
            "geometries": f"<{type(self.geometries).__name__} n={len(self.geometries)}>",
            "column": self.column,
            "all_touched": self.all_touched,
            "fill": self.fill,
        }


class Vectorize(Operator):
    """Vectorize non-zero regions of a mask `GeoTensor` into polygons.

    Wraps :func:`georeader.vectorize.get_polygons`. The carrier's
    transform is used to express polygons in geographic coordinates.
    Polygons with area below ``min_area`` (in square pixels of the
    mask) are dropped.

    Args:
        min_area: Minimum polygon area in square *pixels*. Default
            ``25.5`` (~5×5 px), matching the georeader default.
        simplify_tolerance: Optional post-hoc :meth:`shapely.simplify`
            tolerance (in CRS units). ``None`` (default) leaves the
            staircase pixel boundaries intact.

    Examples:
        >>> import geotoolz as gz
        >>> polys = gz.geom.Vectorize(min_area=100.0)(water_mask)
    """

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
    """Dispatch onto the right `georeader.rasterize` API for the inputs.

    Picks :func:`rasterize_geopandas_like` for `GeoDataFrame` inputs
    (synthesising a unit-value column when none is provided) and falls
    back to :func:`rasterize_geometry_like` over the union of shapely
    geometries otherwise.
    """
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
