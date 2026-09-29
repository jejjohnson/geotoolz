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

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
import rasterio.windows
from affine import Affine
from georeader import (
    griddata,
    mosaic,
    rasterize,
    read,
    slices,
    vectorize,
    window_utils,
)
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

from geotoolz._src.bands import band_count
from geotoolz._src.blending import normalize_overlap_add, overlap_add
from geotoolz._src.config import as_tuple
from geotoolz._src.shape import BAND_AXIS, require_ndim, single_band
from geotoolz._src.valid import valid_pixels, wrap_filled
from geotoolz._src.wrap import adopt_attrs, rewrap_attrs, wrap_like
from geotoolz.geom._src.array import (
    center_offsets,
    feather_weights,
    is_north_up,
    resolve_interpolation,
    resolve_resampling,
    target_slices,
    valid_pixel_mask,
)


_MIN_BOWTIE_COS = 0.1
_RAY_DISCRIMINANT_TOLERANCE = 1.0e-6


if TYPE_CHECKING:
    import geopandas as gpd
    from shapely.geometry.base import BaseGeometry


def _registration_band(values: np.ndarray, band: int, name: str) -> np.ndarray:
    """The ``(H, W)`` registration band of a 2-D map or ``(C, H, W)`` cube.

    Registration pairs one scene with one reference, so a ``(T, C, H, W)``
    stack is rejected (``ValueError`` naming the operator) rather than
    having a time slice taken as a band.
    """
    arr = np.asarray(values)
    if require_ndim(arr, (2, 3), name) == 2:
        return arr
    return np.take(arr, int(band), axis=BAND_AXIS)


def _require_geotensor(value: Any, op_name: str) -> None:
    """Raise a clear ``TypeError`` when a geo-dependent op gets a plain array.

    Geo-dependent operators (reprojection, geometry rasterisation,
    transform-updating resizes, ...) are only meaningful with an affine
    transform + CRS attached. Duck-typed on ``transform`` so any
    GeoTensor-compatible carrier passes.
    """
    if not hasattr(value, "transform"):
        raise TypeError(
            f"{op_name} requires a georeferenced GeoTensor input; "
            f"got a plain array ({type(value).__name__})."
        )


def _read_window_boundless(
    arr: np.ndarray, window: rasterio.windows.Window
) -> np.ndarray:
    """Plain-array analogue of ``GeoTensor.read_from_window(boundless=True)``.

    Slices the window from ``arr`` and zero-pads any part that extends
    past the bottom/right edges (a plain array has no
    ``fill_value_default`` to use as the pad sentinel).
    """
    height, width = arr.shape[-2:]
    row0, col0 = int(window.row_off), int(window.col_off)
    row1 = row0 + int(window.height)
    col1 = col0 + int(window.width)
    view = arr[..., row0 : min(row1, height), col0 : min(col1, width)]
    pad_y = row1 - min(row1, height)
    pad_x = col1 - min(col1, width)
    if pad_y == 0 and pad_x == 0:
        return view
    pad_width = [(0, 0)] * (arr.ndim - 2) + [(0, pad_y), (0, pad_x)]
    return np.pad(view, pad_width, mode="constant", constant_values=0)


def _resize_to_shape(
    gt: GeoTensor,
    shape: tuple[int, int],
    *,
    anti_aliasing: bool,
    interpolation: str,
) -> GeoTensor:
    """``GeoTensor.resize(output_shape=...)`` with a per-axis transform.

    ``GeoTensor.resize`` scales the transform's *x* axis by the *height*
    ratio, which is only right when both axes are scaled by the same
    factor. Rebuild the transform from the per-axis pixel-space scale so
    the output keeps the input's geographic extent.
    """
    height, width = gt.shape[-2:]
    out_h, out_w = int(shape[0]), int(shape[1])
    resized = gt.resize(
        output_shape=(out_h, out_w),
        anti_aliasing=anti_aliasing,
        interpolation=interpolation,
    )
    transform = gt.transform * Affine.scale(width / out_w, height / out_h)
    return wrap_like(
        gt,
        resized.values,
        transform=transform,
        fill_value_default=resized.fill_value_default,
    )


def _copied_attrs(ref: GeoTensor, values: np.ndarray) -> dict[str, Any]:
    """Fresh attrs for a new-grid output built from ``ref`` (see ``rewrap_attrs``)."""
    return rewrap_attrs(
        ref.attrs, in_bands=band_count(ref.shape), out_bands=band_count(values.shape)
    )


def _mosaic_frame(gts: Sequence[GeoTensor]) -> tuple[Affine, rasterio.windows.Window]:
    """Union-of-bounds frame on the first input's pixel grid.

    Mirrors the frame :func:`georeader.mosaic.spatial_mosaic` builds
    (first raster's CRS + pixel grid, outer-rounded to cover every
    input) without warping any data. Returns the frame transform and the
    ``(0, 0, W, H)`` output window.
    """
    first = gts[0]
    first_crs = CRS.from_user_input(first.crs)
    all_bounds = []
    for gt in gts:
        bounds = gt.bounds
        if CRS.from_user_input(gt.crs) != first_crs:
            bounds = transform_bounds(gt.crs, first.crs, *bounds)
        all_bounds.append(bounds)
    union = (
        min(bound[0] for bound in all_bounds),
        min(bound[1] for bound in all_bounds),
        max(bound[2] for bound in all_bounds),
        max(bound[3] for bound in all_bounds),
    )
    window = window_utils.round_outer_window(
        rasterio.windows.from_bounds(*union, transform=first.transform)
    )
    transform = rasterio.windows.transform(window, first.transform)
    return transform, rasterio.windows.Window(
        col_off=0, row_off=0, width=window.width, height=window.height
    )


def _nan_masked(gt: GeoTensor) -> GeoTensor:
    """``float32`` copy of ``gt`` with its fill sentinel replaced by NaN.

    NaN then serves as both source and destination nodata when warping
    onto the mosaic frame, which also covers inputs whose
    ``fill_value_default`` is ``None``.
    """
    arr = np.asarray(gt.values, dtype=np.float32)
    fill = gt.fill_value_default
    if fill is not None and not np.isnan(fill):
        arr = np.where(arr == fill, np.float32(np.nan), arr)
    return GeoTensor(arr, transform=gt.transform, crs=gt.crs, fill_value_default=np.nan)


def _nan_first(stack: np.ndarray, axis: int = 0) -> np.ndarray:
    """First non-NaN value along ``axis`` (NaN where every entry is NaN)."""
    index = np.expand_dims(np.argmax(~np.isnan(stack), axis=axis), axis)
    return np.take_along_axis(stack, index, axis=axis).squeeze(axis)


class Reproject(Operator):
    """Reproject a `GeoTensor` to a destination CRS (and optional resolution).

    Thin Operator wrapping :func:`georeader.read.read_to_crs`. The
    destination grid is anchored to the input's bounding box reprojected
    into ``dst_crs``; the resolution defaults to the input's pixel size
    expressed in the destination CRS.

    Geo-dependent: requires a georeferenced ``GeoTensor`` input
    (transform + CRS); plain arrays raise ``TypeError``.

    Args:
        dst_crs: Target CRS as a string (EPSG code, WKT, PROJ string, or
            anything :class:`pyproj.CRS.from_user_input` accepts).
        resolution: Optional ``(pixel_size_x, pixel_size_y)`` in
            destination-CRS units. ``None`` lets georeader derive it
            from the source pixel size. Honoured even when ``dst_crs``
            equals the input CRS (the call then acts as a resample).
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
        _require_geotensor(gt, "Reproject")
        resampling = resolve_resampling(self.resampling)
        if self.resolution is None:
            return adopt_attrs(
                gt, read.read_to_crs(gt, self.dst_crs, resampling=resampling)
            )
        # ``read_to_crs`` returns its input untouched when the CRS already
        # matches, silently dropping ``resolution``; build the destination
        # grid explicitly so a same-CRS call still resamples.
        window, dst_transform = read.calculate_transform_window(
            gt, self.dst_crs, self.resolution
        )
        return adopt_attrs(
            gt,
            read.read_reproject(
                gt,
                dst_crs=self.dst_crs,
                dst_transform=dst_transform,
                window_out=window,
                resampling=resampling,
            ),
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

    Geo-dependent: requires a georeferenced ``GeoTensor`` input
    (transform + CRS); plain arrays raise ``TypeError``.

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
        _require_geotensor(gt, type(self).__name__)
        return adopt_attrs(
            gt,
            read.read_reproject_like(
                gt,
                self.like,
                resampling=resolve_resampling(self.resampling),
            ),
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
    """Sub-pixel image registration via phase cross-correlation.

    Arrays are interpreted as ``(H, W)`` or channel-first ``(C, H, W)``;
    the selected registration band is taken from the band axis (``-3``).
    A 4-D ``(T, C, H, W)`` stack raises ``ValueError``: register each
    frame separately. The estimated
    ``(dy, dx)`` shift is applied with linear interpolation
    (``scipy.ndimage.shift``) when ``apply=True``.

    The registration math is pixel-space, so plain ``np.ndarray`` inputs
    are supported: they come back as plain shifted arrays. For
    ``GeoTensor`` input the shifted array keeps the input's affine
    transform (which must share the reference's grid), so a feature at
    pixel ``(r, c)`` of the output sits at the same world coordinate as
    pixel ``(r, c)`` of the reference. Edge pixels uncovered by the
    shift are filled from the nearest valid pixel.

    Args:
        reference: The fixed scene the input is registered against.
        upsample_factor: Sub-pixel upsampling factor passed to
            :func:`skimage.registration.phase_cross_correlation`.
        band: Band index (along the band axis) used for registration on
            3-D inputs.
        apply: If ``True`` (default), return the shifted image; if
            ``False``, return the raw ``(shift_y, shift_x, error)``
            tuple instead; the operator is then terminal (last step
            only) in a ``Sequential``.
    """

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self,
        *,
        reference: GeoTensor,
        upsample_factor: int = 10,
        band: int = 0,
        apply: bool = True,
    ) -> None:
        self.reference = reference
        self.upsample_factor = upsample_factor
        self.band = band
        self.apply = apply

    @property
    def _terminal(self) -> bool:  # ty: ignore[invalid-attribute-override]
        """The ``apply=False`` tuple breaks carrier-in / carrier-out."""
        return not self.apply

    def _apply(
        self, gt: GeoTensor | np.ndarray
    ) -> GeoTensor | np.ndarray | tuple[float, float, float]:
        arr = np.asarray(gt)
        mov_band = _registration_band(arr, self.band, "PhaseAlign")
        ref_band = _registration_band(
            np.asarray(self.reference), self.band, "PhaseAlign reference"
        )
        if ref_band.shape != mov_band.shape:
            raise ValueError(
                "PhaseAlign requires reference and moving bands of identical "
                f"spatial shape; got reference={ref_band.shape}, "
                f"moving={mov_band.shape}."
            )
        shift, error, _phase = phase_cross_correlation(
            ref_band,
            mov_band,
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
        # The array shift already moves the content onto the reference
        # grid; keep the original transform (translating it as well would
        # cancel the alignment in world coordinates, #118).
        return wrap_like(gt, np.asarray(shifted))

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


def _wrap_flow(gt: Any, reference: Any, flow: np.ndarray) -> Any:
    """Rewrap a ``(2, H, W)`` displacement field with ``NaN`` as its nodata.

    A displacement is a new quantity: pixels that are nodata in either
    scene (non-finite or equal to that scene's ``fill_value_default``)
    hold ``NaN`` and the output declares ``fill_value_default=NaN``.
    """
    valid = valid_pixels(gt) & valid_pixels(reference)
    return wrap_filled(gt, flow, fill_value_default=np.nan, valid=valid)


class OpticalFlowTVL1(Operator):
    """Dense per-pixel displacement via TV-L1 optical flow.

    Wraps :func:`skimage.registration.optical_flow_tvl1` on the selected
    registration band. The output is a ``(2, H, W)`` displacement field
    (``[dy, dx]`` per pixel) on the input's carrier: pixel-space math,
    so both ``GeoTensor`` and plain ``np.ndarray`` inputs are supported
    and returned in kind. Pixels that are nodata in either scene are
    ``NaN`` (``fill_value_default=NaN``). Inputs are ``(H, W)`` or
    ``(C, H, W)``; a 4-D ``(T, C, H, W)`` stack raises ``ValueError``
    (compute the flow of each frame separately).

    Args:
        reference: The fixed scene the input is registered against.
        band: Band index (along the band axis) used for registration on
            3-D inputs.
    """

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(self, *, reference: GeoTensor, band: int = 0) -> None:
        self.reference = reference
        self.band = band

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        name = type(self).__name__
        flow = np.asarray(
            optical_flow_tvl1(
                _registration_band(
                    np.asarray(self.reference), self.band, f"{name} reference"
                ),
                _registration_band(np.asarray(gt), self.band, name),
            )
        )
        return _wrap_flow(gt, self.reference, flow)

    def get_config(self) -> dict[str, Any]:
        return {
            "reference": {
                "shape": list(np.asarray(self.reference).shape),
                "dtype": str(np.asarray(self.reference).dtype),
            },
            "band": self.band,
        }


class OpticalFlowILK(OpticalFlowTVL1):
    """Dense per-pixel displacement via iterative Lucas-Kanade optical flow.

    Same contract as :class:`OpticalFlowTVL1` (both carriers supported),
    delegating to :func:`skimage.registration.optical_flow_ilk`.
    """

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        name = type(self).__name__
        flow = np.asarray(
            optical_flow_ilk(
                _registration_band(
                    np.asarray(self.reference), self.band, f"{name} reference"
                ),
                _registration_band(np.asarray(gt), self.band, name),
            )
        )
        return _wrap_flow(gt, self.reference, flow)


class Resize(Operator):
    """Resize a `GeoTensor` to a target spatial shape.

    Delegates to :meth:`georeader.geotensor.GeoTensor.resize`, which
    re-bases the affine transform so the geographic extent is preserved
    while the pixel grid is densified or coarsened. ``anti_aliasing``
    applies a Gaussian pre-filter when downscaling (recommended).

    Geo-dependent (transform-updating): requires a georeferenced
    ``GeoTensor`` input; plain arrays raise ``TypeError``.

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
        _require_geotensor(gt, "Resize")
        return _resize_to_shape(
            gt,
            self.shape,
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

    The output shape is derived per axis from the ratio of the input
    pixel size to the requested resolution,
    ``(round(H * res_y / dst_y), round(W * res_x / dst_x))``, and the
    resize is delegated to :meth:`georeader.geotensor.GeoTensor.resize`.
    The geographic extent is preserved, so the achieved pixel size
    matches ``resolution`` exactly only when it divides the extent.

    Geo-dependent: the target resolution is expressed in CRS units, so
    a georeferenced ``GeoTensor`` input is required; plain arrays raise
    ``TypeError``.

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
        _require_geotensor(gt, "Resample")
        res_x, res_y = gt.res
        dst_x, dst_y = self.resolution
        height, width = gt.shape[-2:]
        # Derive the shape per axis here instead of passing
        # ``resolution_dst`` to ``GeoTensor.resize``, which scales the
        # height by the *x* resolution ratio (wrong for non-square pixels).
        shape = (
            max(round(height * res_y / abs(dst_y)), 1),
            max(round(width * res_x / abs(dst_x)), 1),
        )
        return _resize_to_shape(
            gt,
            shape,
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

    Pixel-space padding: plain ``np.ndarray`` inputs are supported and
    padded with :func:`numpy.pad` (returned as plain arrays); the
    transform bookkeeping only happens for ``GeoTensor`` input.

    Args:
        shape: Target minimum spatial shape ``(H, W)``.
        mode: Numpy pad mode. ``"constant"`` (default), ``"edge"``,
            ``"reflect"``, ``"symmetric"``, ...
        fill: Constant value when ``mode == "constant"``. ``None``
            falls back to the carrier's ``fill_value_default``, or ``0``
            when the carrier has none (and for plain-array input).

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

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        height, width = gt.shape[-2:]
        pad_y = max(self.shape[0] - height, 0)
        pad_x = max(self.shape[1] - width, 0)
        if pad_y == 0 and pad_x == 0:
            return gt
        pad_yx = (
            (pad_y // 2, pad_y - pad_y // 2),
            (pad_x // 2, pad_x - pad_x // 2),
        )
        if hasattr(gt, "transform"):
            kwargs: dict[str, Any] = {}
            if self.mode == "constant":
                fill = self.fill
                if fill is None:
                    # ``GeoTensor.pad`` raises when neither a constant nor a
                    # carrier fill is set; fall back to 0 like the ndarray path.
                    fill = 0 if gt.fill_value_default is None else None
                kwargs["constant_values"] = fill
            padded = gt.pad({"y": pad_yx[0], "x": pad_yx[1]}, mode=self.mode, **kwargs)
            return adopt_attrs(gt, padded)
        arr = np.asarray(gt)
        pad_width = [(0, 0)] * (arr.ndim - 2) + list(pad_yx)
        if self.mode == "constant":
            return np.pad(
                arr,
                pad_width,
                mode="constant",
                constant_values=0 if self.fill is None else self.fill,
            )
        # `mode` is user-supplied (numpy pad-mode name); numpy's stubs only
        # accept the literal names, so widen the check away for ty.
        return np.pad(arr, pad_width, mode=cast("Any", self.mode))

    def get_config(self) -> dict[str, Any]:
        return {"shape": list(self.shape), "mode": self.mode, "fill": self.fill}


class CropTo(Operator):
    """Crop a `GeoTensor` to a target spatial shape.

    Anchors either at the centre (default) or upper-left corner of the
    input. The affine transform is updated through
    :meth:`georeader.geotensor.GeoTensor.read_from_window` so the
    geographic extent of the surviving pixels is preserved.

    Pixel-space cropping: plain ``np.ndarray`` inputs are supported
    (sliced and returned as plain arrays); the transform bookkeeping
    only happens for ``GeoTensor`` input.

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

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        height, width = gt.shape[-2:]
        if self.shape[0] > height or self.shape[1] > width:
            raise ValueError(f"Cannot crop shape {(height, width)} to {self.shape}.")
        if self.anchor == "center":
            row_off, col_off = center_offsets((height, width), self.shape)
        elif self.anchor == "upper_left":
            row_off, col_off = 0, 0
        else:
            raise ValueError("anchor must be 'center' or 'upper_left'.")
        if not hasattr(gt, "transform"):
            return np.asarray(gt)[
                ...,
                row_off : row_off + self.shape[0],
                col_off : col_off + self.shape[1],
            ]
        window = rasterio.windows.Window(
            col_off=col_off,
            row_off=row_off,
            width=self.shape[1],
            height=self.shape[0],
        )
        return adopt_attrs(gt, gt.read_from_window(window, boundless=False))

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

    Geo-dependent: requires a georeferenced ``GeoTensor`` input
    (transform + CRS); plain arrays raise ``TypeError``.

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
        _require_geotensor(gt, "CropToBounds")
        bounds = self.bounds
        if self.crs is not None and CRS.from_user_input(
            self.crs
        ) != CRS.from_user_input(gt.crs):
            bounds = transform_bounds(self.crs, gt.crs, *bounds)
        window = rasterio.windows.from_bounds(*bounds, transform=gt.transform)
        window = window.round_offsets().round_lengths()
        return adopt_attrs(gt, gt.read_from_window(window, boundless=False))

    def get_config(self) -> dict[str, Any]:
        return {"bounds": list(self.bounds), "crs": self.crs}


class Tile(Operator):
    """Split a `GeoTensor` into spatial tiles.

    Wraps :func:`georeader.slices.create_windows` (with
    ``trim_incomplete=False``) and then reads each window with
    ``boundless=True`` so the trailing-edge tiles match ``size``
    exactly. Out-of-bounds areas are padded with the carrier's
    ``fill_value_default``; use :func:`Stitch` (which masks the
    sentinel) to recover the original extent. A ``GeoTensor`` with
    ``fill_value_default=None`` is zero-padded instead — ``Stitch``
    cannot tell that padding from data, so pin its ``target_shape`` /
    ``target_transform`` to the source grid to drop it.

    Pixel-space tiling: plain ``np.ndarray`` inputs are supported and
    yield a list of plain arrays (out-of-bounds areas zero-padded, since
    a plain array carries no ``fill_value_default``). Note that
    :class:`Stitch` cannot reassemble plain-array tiles — it needs each
    tile's transform to place it.

    Args:
        size: Tile spatial shape ``(H, W)``.
        stride: Step between consecutive tile origins ``(sy, sx)``. The
            implied overlap is ``size - stride``. ``None`` (default)
            means non-overlapping tiles (``stride == size``).
        include_incomplete: Whether to keep edge tiles that don't fully
            fit (still emitted at ``size`` shape, padded as above).
            ``False`` drops them, so the trailing rows / columns that no
            full tile covers are not emitted at all.

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

    def _apply(self, gt: GeoTensor | np.ndarray) -> list[GeoTensor] | list[np.ndarray]:
        overlap: tuple[int, int] | None = None
        if self.stride is not None:
            overlap = (self.size[0] - self.stride[0], self.size[1] - self.stride[1])
            if overlap[0] < 0 or overlap[1] < 0:
                raise ValueError("stride must be less than or equal to size.")
        # ``trim_incomplete=False`` keeps every window at ``size`` so the
        # trailing tiles extend past the raster and get padded below
        # (georeader's default trims them to the raster bounds instead).
        windows = slices.create_windows(
            gt.shape[-2:],
            self.size,
            overlap=overlap,
            include_incomplete=self.include_incomplete,
            trim_incomplete=False,
        )
        if not hasattr(gt, "transform"):
            arr = np.asarray(gt)
            return [_read_window_boundless(arr, window) for window in windows]
        if gt.fill_value_default is not None:
            return [
                adopt_attrs(gt, gt.read_from_window(window, boundless=True))
                for window in windows
            ]
        # ``GeoTensor.read_from_window(boundless=True)`` refuses to pad
        # without a fill value; zero-pad like the plain-array path.
        return [
            wrap_like(
                gt,
                _read_window_boundless(np.asarray(gt.values), window),
                transform=rasterio.windows.transform(window, gt.transform),
            )
            for window in windows
        ]

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
    - ``"feather"`` — overlap-add with triangular edge weights, kernel from
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

    Unless ``target_shape`` is pinned, trailing bottom rows / right
    columns with no valid contributor (the sentinel padding
    :class:`Tile` adds to its edge tiles) are cropped off, so a
    ``Tile`` → ``Stitch`` round trip recovers the source extent. Pin
    ``target_shape`` to keep them.

    Geo-dependent: every tile must be a georeferenced ``GeoTensor`` —
    the tile transforms place each tile on the output grid. Plain-array
    tiles raise ``TypeError``. Every tile must be north-up
    (``b == d == 0``, ``a > 0``, ``e < 0``) and share the first tile's
    pixel size; otherwise ``ValueError`` is raised.

    Args:
        blend: Blend mode (see above). Default ``"average"``.
        feather_width: Pixel ramp width for ``"feather"`` blending.
        target_shape: Optional ``(H, W)`` override for the output grid.
        target_transform: Optional :class:`affine.Affine` override for
            the output grid origin, or its first six coefficients
            ``[a, b, c, d, e, f]`` (the form ``get_config`` emits). Must
            be north-up with the tiles' pixel size.
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
        target_transform: Affine | Sequence[float] | None = None,
        target_crs: str | None = None,
        fill: float | int | None = None,
    ) -> None:
        self.blend = blend
        self.feather_width = feather_width
        self.target_shape = as_tuple(target_shape)
        if target_transform is not None and not isinstance(target_transform, Affine):
            target_transform = Affine(*target_transform[:6])
        self.target_transform = target_transform
        self.target_crs = target_crs
        self.fill = fill

    def _apply(self, tiles: list[GeoTensor]) -> GeoTensor:
        if len(tiles) == 0:
            raise ValueError("Stitch requires at least one tile.")
        if self.blend not in {"average", "feather", "max", "first"}:
            raise ValueError("blend must be 'average', 'feather', 'max', or 'first'.")
        first = tiles[0]
        for index, tile in enumerate(tiles):
            if not hasattr(tile, "transform"):
                raise TypeError(
                    f"Stitch requires georeferenced GeoTensor tiles; tile "
                    f"{index} is a plain array ({type(tile).__name__})."
                )
            if not is_north_up(tile.transform):
                raise ValueError(
                    f"Stitch only supports north-up, non-rotated GeoTensors; "
                    f"tile {index} has a rotated/sheared or flipped transform "
                    f"({tuple(tile.transform)[:6]})."
                )
            # ``target_slices`` places tiles by origin only and assumes one
            # pixel size, so a coarser/finer tile would be pasted at the
            # wrong scale.
            if not np.allclose(tile.res, first.res):
                raise ValueError(
                    f"Stitch requires every tile to share the first tile's "
                    f"resolution {tuple(first.res)}; tile {index} has "
                    f"{tuple(tile.res)}."
                )
        if self.target_transform is not None and not (
            is_north_up(self.target_transform)
            and np.allclose(
                (self.target_transform.a, -self.target_transform.e), first.res
            )
        ):
            raise ValueError(
                f"target_transform must be north-up with the tiles' resolution "
                f"{tuple(first.res)}; got {tuple(self.target_transform)[:6]}."
            )
        transform, shape = self._target_grid(tiles)
        fill = first.fill_value_default if self.fill is None else self.fill
        dtype = first.dtype if self.blend in {"first", "max"} else np.float32
        out_shape = first.shape[:-2] + shape
        if self.blend == "max":
            values = np.full(out_shape, fill, dtype=dtype)
            covered = np.zeros(shape, dtype=bool)
            self._stitch_max(tiles, transform, values, covered, fill)
        elif self.blend == "first":
            values = np.full(out_shape, fill, dtype=dtype)
            covered = np.zeros(shape, dtype=bool)
            self._stitch_first(tiles, transform, values, covered, fill)
        else:  # average / feather
            values = np.zeros(out_shape, dtype=dtype)
            weights = np.zeros(shape, dtype=np.float32)
            self._stitch_average(tiles, transform, values, weights, fill)
            covered = weights > 0
            normalize_overlap_add(values, weights, fill)
        if self.target_shape is None:
            # Drop the trailing rows / columns that only :class:`Tile`'s
            # sentinel padding reached, recovering the source extent.
            rows = np.flatnonzero(covered.any(axis=1))
            cols = np.flatnonzero(covered.any(axis=0))
            if rows.size and cols.size:
                values = values[..., : rows[-1] + 1, : cols[-1] + 1]
        return GeoTensor(
            values,
            transform,
            self.target_crs or first.crs,
            fill_value_default=fill,
            attrs=_copied_attrs(first, values),
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
            mask_tile = valid_pixel_mask(arr, fill)
            kernel = (
                feather_weights(tile.shape[-2:], self.feather_width)
                if self.blend == "feather"
                else np.ones(tile.shape[-2:], dtype=np.float32)
            )
            overlap_add(
                values, weights, arr, out_slices, tile_slices, kernel, mask_tile
            )

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


class BowtieCorrection(Operator):
    """Resample scan-edge pixels to reduce bowtie overlap.

    The correction uses the sensor cross-track scan angle to estimate the
    scan-angle IFOV expansion (``1 / cos(θ)``), then samples each scan row
    from a compressed cross-track coordinate. ``scan_angle_max_deg=0`` is an
    exact identity.

    Pixel-space resampling (driven entirely by the scan-geometry
    constructor arguments): both ``GeoTensor`` and plain ``np.ndarray``
    inputs are supported and returned in kind.

    Args:
        scan_angle_max_deg: Maximum off-nadir scan angle in degrees.
        pixels_per_scan: Cross-track sample count.
        scans_per_granule: Along-track scan count.
        method: ``"nearest"`` or ``"bilinear"``.

    Examples:
        >>> import geotoolz as gz
        >>> debowtie = gz.geom.BowtieCorrection(
        ...     scan_angle_max_deg=55.0,
        ...     pixels_per_scan=1354,
        ...     scans_per_granule=203,
        ... )
    """

    def __init__(
        self,
        *,
        scan_angle_max_deg: float,
        pixels_per_scan: int,
        scans_per_granule: int,
        method: str = "nearest",
    ) -> None:
        if not 0.0 <= scan_angle_max_deg < 70.0:
            raise ValueError(
                "scan_angle_max_deg must be >= 0.0 and < 70.0 for stable "
                "cosine-based IFOV correction."
            )
        self.scan_angle_max_deg = scan_angle_max_deg
        self.pixels_per_scan = pixels_per_scan
        self.scans_per_granule = scans_per_granule
        self.method = method

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        if self.method not in {"nearest", "bilinear"}:
            raise ValueError("method must be 'nearest' or 'bilinear'.")
        height, width = gt.shape[-2:]
        if width != self.pixels_per_scan:
            raise ValueError(
                f"Expected {self.pixels_per_scan} cross-track pixels, got {width}."
            )
        if height != self.scans_per_granule:
            raise ValueError(
                f"Expected {self.scans_per_granule} along-track scans, got {height}."
            )
        if self.scan_angle_max_deg == 0:
            return gt
        rows = np.broadcast_to(
            np.arange(height, dtype=np.float64)[:, None], (height, width)
        )
        centre = (width - 1) / 2.0
        cols = np.arange(width, dtype=np.float64)
        angles = np.deg2rad(
            np.linspace(-self.scan_angle_max_deg, self.scan_angle_max_deg, width)
        )
        expansion = 1.0 / np.clip(np.cos(angles), _MIN_BOWTIE_COS, None)
        src_cols = centre + (cols - centre) / expansion
        sampled = _sample_array(
            np.asarray(gt),
            rows,
            np.broadcast_to(src_cols[None, :], (height, width)),
            method=self.method,
            fill=getattr(gt, "fill_value_default", None),
        )
        return wrap_like(gt, sampled)


class AntimeridianSplit(Operator):
    """Split a geographic swath at a longitude wrap.

    Longitudes are read from ``gt.attrs["lons"]`` / ``gt.attrs["lon"]`` when
    present; otherwise they are derived from the affine grid for geographic
    CRSs. If no jump larger than ``tolerance_deg`` is found, the output is a
    single-item list containing the input unchanged.

    Geo-dependent: requires a georeferenced ``GeoTensor`` input (attrs /
    geographic grid); plain arrays raise ``TypeError``.

    Args:
        crs: Geographic CRS used to interpret longitudes.
        tolerance_deg: Absolute longitude jump threshold in degrees.
    """

    def __init__(self, *, crs: str = "EPSG:4326", tolerance_deg: float = 90.0) -> None:
        self.crs = crs
        self._parsed_crs = CRS.from_user_input(crs)
        self.tolerance_deg = tolerance_deg

    def _apply(self, gt: GeoTensor) -> list[GeoTensor]:
        _require_geotensor(gt, "AntimeridianSplit")
        lons = _longitude_grid(gt, self._parsed_crs)
        if lons is None:
            raise ValueError(
                "AntimeridianSplit requires geographic longitudes in attrs['lons'] "
                "or an affine geographic grid."
            )
        diffs = np.abs(np.diff(lons, axis=1))
        if diffs.size == 0 or np.nanmax(diffs) <= self.tolerance_deg:
            return [gt]
        # Reject swaths with more than one longitude discontinuity (e.g. polar
        # passes that wrap twice). The single-split heuristic below cannot
        # represent multi-jump cases correctly.
        per_col_max = np.nanmax(diffs, axis=0)
        n_jumps = int(np.sum(per_col_max > self.tolerance_deg))
        if n_jumps > 1:
            raise ValueError(
                f"AntimeridianSplit found {n_jumps} longitude jumps above "
                f"tolerance_deg={self.tolerance_deg}; only single-discontinuity "
                "swaths are supported."
            )
        # The diff at column i is the jump between i and i + 1.
        split_col = int(np.nanargmax(per_col_max)) + 1
        left = gt.isel({"x": slice(0, split_col)})
        right = gt.isel({"x": slice(split_col, gt.shape[-1])})
        left = _with_sliced_longitudes(gt, left, lons[:, :split_col])
        right = _with_sliced_longitudes(gt, right, lons[:, split_col:])
        left_mean = float(np.nanmean(_normalise_longitudes(lons[:, :split_col])))
        right_mean = float(np.nanmean(_normalise_longitudes(lons[:, split_col:])))
        return [left, right] if left_mean < right_mean else [right, left]


class GeostationaryParallaxCorrect(Operator):
    """Shift pixels from apparent geostationary view position to nadir.

    A spherical Earth ray-intersection model is used: for each output ground
    pixel, the operator finds where an elevated target would appear to a
    geostationary satellite and samples the input there, treating each
    input value as located at its pixel centre. ``target_height_m=0``
    is an exact identity. Off-limb pixels (whose viewing ray misses the Earth
    sphere) are filled with ``gt.fill_value_default``.

    Geo-dependent: requires a georeferenced ``GeoTensor`` input on an
    EPSG:4326 grid; plain arrays raise ``TypeError``.

    Args:
        satellite_lon_deg: Sub-satellite longitude.
        satellite_height_m: Satellite height above the equator.
        target_height_m: Scalar height, same-grid array, or same-grid GeoTensor.
        earth_radius_m: Earth radius (spherical model).
        method: ``"nearest"`` or ``"bilinear"``.
    """

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self,
        *,
        satellite_lon_deg: float,
        satellite_height_m: float = 35_786_023.0,
        target_height_m: float | np.ndarray | GeoTensor = 0.0,
        earth_radius_m: float = 6_378_137.0,
        method: str = "bilinear",
    ) -> None:
        self.satellite_lon_deg = satellite_lon_deg
        self.satellite_height_m = satellite_height_m
        self.target_height_m = target_height_m
        self.earth_radius_m = earth_radius_m
        self.method = method

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        _require_geotensor(gt, "GeostationaryParallaxCorrect")
        if self.method not in {"nearest", "bilinear"}:
            raise ValueError("method must be 'nearest' or 'bilinear'.")
        heights = _height_array(self.target_height_m, gt.shape[-2:])
        if np.all(heights == 0):
            return gt
        lons_lats = _lonlat_centres(gt)
        if lons_lats is None:
            raise ValueError("GeostationaryParallaxCorrect requires EPSG:4326 grids.")
        lons, lats = lons_lats
        apparent_lon, apparent_lat, off_limb = _apparent_lonlat(
            lons,
            lats,
            heights,
            satellite_lon_deg=self.satellite_lon_deg,
            satellite_height_m=self.satellite_height_m,
            earth_radius_m=self.earth_radius_m,
        )
        inv = ~gt.transform
        src_cols, src_rows = inv * (apparent_lon, apparent_lat)
        # ``~transform`` yields continuous pixel coordinates in which pixel
        # ``(r, c)`` is centred at ``(r + 0.5, c + 0.5)``; the samplers index
        # ``arr[r]`` at coordinate ``r``, so shift to centre-indexed space.
        sampled = _sample_array(
            np.asarray(gt),
            np.asarray(src_rows) - 0.5,
            np.asarray(src_cols) - 0.5,
            method=self.method,
            fill=gt.fill_value_default,
        )
        if np.any(off_limb):
            sampled = np.where(
                off_limb,
                np.asarray(gt.fill_value_default, dtype=sampled.dtype),
                sampled,
            )
        return wrap_like(gt, sampled)

    def get_config(self) -> dict[str, Any]:
        target: float | str
        if np.isscalar(self.target_height_m):
            target = float(self.target_height_m)
        else:
            target = f"<{type(self.target_height_m).__name__}>"
        return {
            "satellite_lon_deg": self.satellite_lon_deg,
            "satellite_height_m": self.satellite_height_m,
            "target_height_m": target,
            "earth_radius_m": self.earth_radius_m,
            "method": self.method,
        }


class SegmentStitch(Operator):
    """Assemble indexed sensor segments into one contiguous ``GeoTensor``.

    Segment order is read from each input's
    ``attrs["__geotoolz_segment_meta__"]`` dictionary with
    ``segment_index`` and ``n_segments`` keys. Both zero-based
    (``0..n_segments-1``) and one-based (``1..n_segments``) indexing are
    accepted. One-based indexing is selected only when an index equal to
    ``n_segments`` is present; ambiguous missing-edge cases default to
    zero-based. Missing segments are filled with ``fill``.

    Geo-dependent: every segment must be a georeferenced ``GeoTensor``
    (attrs metadata + transform bookkeeping); plain arrays raise
    ``TypeError``.

    Args:
        axis: ``"scan"`` / ``"y"`` for along-track, or ``"sample"`` /
            ``"x"`` for cross-track.
        fill: Value used for missing segments. ``None`` (default) means
            ``NaN``; a ``NaN`` argument is stored as ``None`` so the config
            stays strict JSON.
    """

    def __init__(self, *, axis: str = "scan", fill: float | None = None) -> None:
        self.axis = axis
        self.fill = None if fill is None or np.isnan(fill) else fill

    def _apply(self, segments: list[GeoTensor]) -> GeoTensor:
        if len(segments) == 0:
            raise ValueError("SegmentStitch requires at least one segment.")
        for segment in segments:
            _require_geotensor(segment, "SegmentStitch")
        axis_num = _segment_axis(self.axis)
        metas = [_segment_meta(segment) for segment in segments]
        n_segments = metas[0][1]
        if any(n != n_segments for _, n in metas):
            raise ValueError("All segments must declare the same n_segments.")
        index_offset = 1 if max(index for index, _ in metas) == n_segments else 0
        by_index: dict[int, GeoTensor] = {}
        for (index, _), segment in zip(metas, segments, strict=True):
            normalised = index - index_offset
            if normalised in by_index:
                raise ValueError(
                    f"Duplicate segment_index {index} after index_offset "
                    f"normalisation; each segment must declare a unique index."
                )
            by_index[normalised] = segment
        if any(index < 0 or index >= n_segments for index in by_index):
            raise ValueError("segment_index is outside the declared n_segments range.")
        first_index = min(by_index)
        first = by_index[first_index]
        for segment in segments:
            _validate_segment_compatible(first, segment, axis_num)
        # Choose a fill that is representable in the segment dtype: integer
        # tensors cannot hold NaN, so fall back to the first segment's own
        # fill (or zero) when the requested fill is non-finite.
        fill = np.nan if self.fill is None else self.fill
        segment_fill = _coerce_segment_fill(fill, first)
        pieces: list[np.ndarray] = []
        segment_shape = first.shape
        for index in range(n_segments):
            if index in by_index:
                pieces.append(np.asarray(by_index[index]))
            else:
                pieces.append(np.full(segment_shape, segment_fill, dtype=first.dtype))
        values = np.concatenate(pieces, axis=axis_num)
        row_before = first_index * segment_shape[-2] if axis_num == -2 else 0
        col_before = first_index * segment_shape[-1] if axis_num == -1 else 0
        transform = first.transform * Affine.translation(-col_before, -row_before)
        out = wrap_like(
            first, values, transform=transform, fill_value_default=segment_fill
        )
        out.attrs["__geotoolz_segment_meta__"] = {
            "segment_index": 0,
            "n_segments": 1,
            "source_n_segments": n_segments,
        }
        return out


class Mosaic(Operator):
    r"""Mosaic multiple `GeoTensor`s onto a single grid.

    For ``method="first"`` this is a thin wrapper around
    :func:`georeader.mosaic.spatial_mosaic`: rasters are processed in
    order and the first valid pixel wins. For all other methods the
    *frame* is the union of the input bounds on the first raster's pixel
    grid (its CRS / dtype), every input is reprojected onto that frame
    (its fill sentinel becoming NaN) and the per-pixel reduction is
    applied:

    .. math::

        \hat{x}_{ij} \;=\; \mathrm{agg}\!\left(
            \{ x^{(k)}_{ij} \;|\; x^{(k)}_{ij} \neq \text{fill} \}
        \right)

    with ``agg`` chosen from ``mean``, ``median``, ``max``, ``min``.
    Pixels no input covers get the first raster's ``fill_value_default``;
    when that is ``None`` they get NaN (float dtypes) or ``0`` (integer
    dtypes), and ``"first"`` also takes this stacked path because
    ``spatial_mosaic`` cannot pad fill-less inputs.

    Geo-dependent: every input must be a georeferenced ``GeoTensor``
    (transform + CRS drive the mosaic frame); plain arrays raise
    ``TypeError``.

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
        "first": _nan_first,
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
        for gt in gts:
            _require_geotensor(gt, "Mosaic")
        if self.method == "first" and all(
            gt.fill_value_default is not None for gt in gts
        ):
            return adopt_attrs(
                gts[0],
                mosaic.spatial_mosaic(
                    gts, resampling=resolve_resampling(self.resampling)
                ),
            )
        # ``spatial_mosaic`` cannot pad inputs that carry no fill value, so
        # ``"first"`` over such inputs goes through the stacked path below.
        if self.method not in self._REDUCERS:
            raise ValueError(
                "method must be 'first', 'mean', 'median', 'max', or 'min'."
            )
        first = gts[0]
        transform, window = _mosaic_frame(gts)
        arrays = [
            np.asarray(
                read.read_reproject(
                    _nan_masked(gt),
                    dst_crs=first.crs,
                    dst_transform=transform,
                    window_out=window,
                    resampling=resolve_resampling(self.resampling),
                    dst_nodata=np.nan,
                )
            )
            for gt in gts
        ]
        stack = np.stack(arrays, axis=0)
        # Pixels no input covers are all-NaN slices; numpy warns about
        # them, but they are expected here and refilled below.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            values = self._REDUCERS[self.method](stack, axis=0)
        fill = first.fill_value_default
        if fill is None:
            fill = np.nan if np.issubdtype(first.dtype, np.floating) else 0
        values = np.where(np.isnan(values), fill, values)
        return wrap_like(
            first,
            values.astype(first.dtype, copy=False),
            transform=transform,
            fill_value_default=fill,
        )


class Georeference(Operator):
    """Georeference swath data using a `georeader` GLT `GeoTensor`.

    Thin wrapper around :func:`georeader.griddata.georreference` (note
    the upstream spelling). The GLT (Geolocation Lookup Table) maps
    output-grid pixels to source-sensor pixels for fast, exact
    orthorectification with no resampling artifacts.

    The input is sensor-space swath data, so plain ``np.ndarray`` input
    is accepted alongside ``GeoTensor``; the output is always a
    georeferenced ``GeoTensor`` whose geometry comes from the GLT.

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

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor:
        return griddata.georreference(
            self.glt,
            np.asarray(gt),
            valid_glt=self.valid_glt,
            fill_value_default=getattr(gt, "fill_value_default", None),
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

    Geo-dependent: requires a georeferenced ``GeoTensor`` input (its
    transform + CRS define the burn grid); plain arrays raise
    ``TypeError``.

    Note:
        Carries Python-level geometry objects (shapely / geopandas),
        so flagged ``forbid_in_yaml = True``.

    Args:
        geometries: Shapely geometries or a `GeoDataFrame`.
        column: For `GeoDataFrame` inputs, the column with burn-in
            values. ``None`` means "burn ``1``".
        all_touched: Whether to mark every pixel the geometry touches
            (vs. only those whose centre is inside).
        fill: Background value. Default ``0``. For a list of geometries
            the output is ``uint8`` when ``fill`` is an integer in
            ``[0, 255]`` and ``float32`` otherwise (e.g. ``np.nan``).

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
        _require_geotensor(gt, "Rasterize")
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

    Geo-dependent: requires a georeferenced ``GeoTensor`` input (the
    transform maps pixels to polygon coordinates); plain arrays raise
    ``TypeError``. The mask must be single-band, ``(H, W)`` or
    ``(1, H, W)``; other shapes raise ``ValueError``.

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

    _terminal: ClassVar[bool] = True

    def __init__(
        self,
        *,
        min_area: float = 25.5,
        simplify_tolerance: float | None = None,
    ) -> None:
        self.min_area = min_area
        self.simplify_tolerance = simplify_tolerance

    def _apply(self, gt: GeoTensor) -> list[BaseGeometry]:
        _require_geotensor(gt, "Vectorize")
        # ``get_polygons`` asserts a 2-D mask; squeeze the canonical
        # ``(1, H, W)`` band axis and pass the transform explicitly.
        mask = single_band(np.asarray(gt.values), name="Vectorize")
        polygons = vectorize.get_polygons(
            mask, min_area=self.min_area, tolerance=0.0, transform=gt.transform
        )
        if self.simplify_tolerance is None:
            return polygons
        return [polygon.simplify(self.simplify_tolerance) for polygon in polygons]


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
    # georeader defaults to ``uint8``, which cannot hold a NaN / negative /
    # fractional / >255 background; widen to ``float32`` for those.
    fill_value = float(fill)
    fits_uint8 = fill_value.is_integer() and 0 <= fill_value <= 255
    return rasterize.rasterize_geometry_like(
        geometry,
        like,
        value=1,
        dtype=np.uint8 if fits_uint8 else np.float32,
        fill=fill,
        all_touched=all_touched,
    )


def _sample_array(
    arr: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    method: str,
    fill: float | int | None,
) -> np.ndarray:
    if method == "nearest":
        return _sample_nearest(arr, rows, cols, fill)
    return _sample_bilinear(arr, rows, cols, fill)


def _sample_nearest(
    arr: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    fill: float | int | None,
) -> np.ndarray:
    height, width = arr.shape[-2:]
    row_idx = np.rint(rows).astype(int)
    col_idx = np.rint(cols).astype(int)
    valid = (row_idx >= 0) & (row_idx < height) & (col_idx >= 0) & (col_idx < width)
    safe_rows = np.clip(row_idx, 0, height - 1)
    safe_cols = np.clip(col_idx, 0, width - 1)
    sampled = arr[..., safe_rows, safe_cols]
    return _apply_invalid_fill(sampled, valid, fill)


def _sample_bilinear(
    arr: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    fill: float | int | None,
) -> np.ndarray:
    height, width = arr.shape[-2:]
    # ``rows`` / ``cols`` are centre-indexed: ``arr[r]`` sits at coordinate
    # ``r`` and pixel ``r`` covers ``[r - 0.5, r + 0.5]``. A sample is valid
    # whenever it lies inside that raster footprint; in the outer half-pixel
    # ring (between the edge centres and the footprint boundary) the
    # coordinate is clamped to the edge centre, returning the edge value.
    valid = (
        (rows >= -0.5) & (rows <= height - 0.5) & (cols >= -0.5) & (cols <= width - 0.5)
    )
    rows = np.clip(rows, 0, height - 1)
    cols = np.clip(cols, 0, width - 1)
    row0 = np.floor(rows).astype(int)
    col0 = np.floor(cols).astype(int)
    row1 = row0 + 1
    col1 = col0 + 1
    # Clamping the upper neighbour below keeps the bilinear formula well
    # defined at the trailing edge — its weight is zero there, so the edge
    # value is returned exactly.
    safe_row0 = np.clip(row0, 0, height - 1)
    safe_row1 = np.clip(row1, 0, height - 1)
    safe_col0 = np.clip(col0, 0, width - 1)
    safe_col1 = np.clip(col1, 0, width - 1)
    row_weight = (
        (rows - row0).astype(arr.dtype, copy=False)
        if np.issubdtype(arr.dtype, np.floating)
        else rows - row0
    )
    col_weight = (
        (cols - col0).astype(arr.dtype, copy=False)
        if np.issubdtype(arr.dtype, np.floating)
        else cols - col0
    )
    one = arr.dtype.type(1) if np.issubdtype(arr.dtype, np.floating) else 1.0
    top_left = arr[..., safe_row0, safe_col0]
    top_right = arr[..., safe_row0, safe_col1]
    bottom_left = arr[..., safe_row1, safe_col0]
    bottom_right = arr[..., safe_row1, safe_col1]
    sampled = (
        top_left * (one - row_weight) * (one - col_weight)
        + top_right * (one - row_weight) * col_weight
        + bottom_left * row_weight * (one - col_weight)
        + bottom_right * row_weight * col_weight
    )
    if np.issubdtype(arr.dtype, np.floating) and sampled.dtype != arr.dtype:
        sampled = sampled.astype(arr.dtype, copy=False)
    return _apply_invalid_fill(sampled, valid, fill)


def _apply_invalid_fill(
    sampled: np.ndarray,
    valid: np.ndarray,
    fill: float | int | None,
) -> np.ndarray:
    """Fill samples that fall outside the input grid.

    ``GeoTensor.fill_value_default`` can be ``None``; in that case this
    internal resampler uses ``0`` for invalid samples, matching georeader's
    constructor default.
    """
    if valid.all():
        return sampled
    fill_value = 0 if fill is None else fill
    # ``valid`` is a 2D (H, W) mask; ``sampled`` may have leading band
    # dimensions. ``np.where`` broadcasts the mask across them, while
    # boolean indexing with a 2D mask collapses those trailing axes.
    return np.where(valid, sampled, np.asarray(fill_value, dtype=sampled.dtype))


def _longitude_grid(gt: GeoTensor, expected_crs: str | CRS) -> np.ndarray | None:
    attrs = gt.attrs or {}
    spatial_shape = gt.shape[-2:]
    for key in ("lons", "lon", "longitude"):
        if key in attrs:
            lons = np.asarray(attrs[key], dtype=np.float64)
            if lons.ndim == 1:
                if lons.shape[0] != spatial_shape[1]:
                    raise ValueError(
                        f"attrs['{key}'] has length {lons.shape[0]} but the "
                        f"GeoTensor has width {spatial_shape[1]}."
                    )
                return np.broadcast_to(lons[None, :], spatial_shape)
            if lons.shape != spatial_shape:
                raise ValueError(
                    f"attrs['{key}'] has shape {lons.shape} but the GeoTensor "
                    f"spatial shape is {spatial_shape}."
                )
            return lons
    parsed_expected = (
        expected_crs
        if isinstance(expected_crs, CRS)
        else CRS.from_user_input(expected_crs)
    )
    if CRS.from_user_input(gt.crs) != parsed_expected:
        return None
    centres = _lonlat_centres(gt)
    if centres is None:
        return None
    return centres[0]


def _lonlat_centres(gt: GeoTensor) -> tuple[np.ndarray, np.ndarray] | None:
    if CRS.from_user_input(gt.crs) != CRS.from_epsg(4326):
        return None
    height, width = gt.shape[-2:]
    rows, cols = np.meshgrid(
        np.arange(height, dtype=np.float64) + 0.5,
        np.arange(width, dtype=np.float64) + 0.5,
        indexing="ij",
    )
    xs, ys = gt.transform * (cols, rows)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def _normalise_longitudes(lons: np.ndarray) -> np.ndarray:
    return ((lons + 180.0) % 360.0) - 180.0


def _with_sliced_longitudes(
    ref: GeoTensor, part: GeoTensor, lons: np.ndarray
) -> GeoTensor:
    out = adopt_attrs(ref, part)
    out.attrs["lons"] = _normalise_longitudes(lons)
    return out


def _height_array(
    target_height_m: float | np.ndarray | GeoTensor, shape: tuple[int, int]
) -> np.ndarray:
    if isinstance(target_height_m, GeoTensor):
        heights = np.asarray(target_height_m, dtype=np.float64)
    elif np.isscalar(target_height_m):
        return np.full(shape, float(target_height_m), dtype=np.float64)
    else:
        heights = np.asarray(target_height_m, dtype=np.float64)
    if heights.shape[-2:] != shape:
        raise ValueError(
            f"target_height_m shape {heights.shape[-2:]} does not match {shape}."
        )
    if heights.ndim > 2:
        heights = heights.reshape(-1, *shape)[0]
    return heights


def _apparent_lonlat(
    lons: np.ndarray,
    lats: np.ndarray,
    heights: np.ndarray,
    *,
    satellite_lon_deg: float,
    satellite_height_m: float,
    earth_radius_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon_rad = np.deg2rad(lons)
    lat_rad = np.deg2rad(lats)
    sat_lon = np.deg2rad(satellite_lon_deg)
    surface = _spherical_xyz(lon_rad, lat_rad, earth_radius_m)
    # Scale each surface radial vector outward by the target height.
    elevated = surface * ((earth_radius_m + heights)[None, ...] / earth_radius_m)
    satellite = np.array(
        [
            (earth_radius_m + satellite_height_m) * np.cos(sat_lon),
            (earth_radius_m + satellite_height_m) * np.sin(sat_lon),
            0.0,
        ],
        dtype=np.float64,
    )[:, None, None]
    direction = elevated - satellite
    a = np.sum(direction * direction, axis=0)
    b = 2.0 * np.sum(satellite * direction, axis=0)
    c = np.sum(satellite * satellite, axis=0) - earth_radius_m**2
    discriminant = b * b - 4.0 * a * c
    off_limb = discriminant < -_RAY_DISCRIMINANT_TOLERANCE
    near = (-b - np.sqrt(np.maximum(discriminant, 0.0))) / (2.0 * a)
    apparent = satellite + direction * near[None, ...]
    apparent_lon = np.rad2deg(np.arctan2(apparent[1], apparent[0]))
    apparent_lat = np.rad2deg(
        np.arctan2(apparent[2], np.hypot(apparent[0], apparent[1]))
    )
    return _normalise_longitudes(apparent_lon), apparent_lat, off_limb


def _spherical_xyz(
    lon_rad: np.ndarray,
    lat_rad: np.ndarray,
    radius: float,
) -> np.ndarray:
    cos_lat = np.cos(lat_rad)
    return np.stack(
        [
            radius * cos_lat * np.cos(lon_rad),
            radius * cos_lat * np.sin(lon_rad),
            radius * np.sin(lat_rad),
        ],
        axis=0,
    )


def _segment_axis(axis: str) -> int:
    if axis in {"scan", "y"}:
        return -2
    if axis in {"sample", "x"}:
        return -1
    raise ValueError("axis must be 'scan'/'y' or 'sample'/'x'.")


def _coerce_segment_fill(fill: float | int, first: GeoTensor) -> float | int:
    """Return a fill value that is representable in ``first.dtype``.

    Integer-typed sensor segments cannot hold ``NaN``; in that case fall back
    to the segment's own ``fill_value_default`` (when finite) or ``0``.
    """
    dtype = np.asarray(first).dtype
    if np.issubdtype(dtype, np.floating):
        return fill
    try:
        is_nan = bool(np.isnan(fill))
    except TypeError:
        is_nan = False
    if not is_nan:
        return fill
    existing = first.fill_value_default
    if existing is not None:
        try:
            if not bool(np.isnan(existing)):
                return existing
        except TypeError:
            return existing
    return 0


def _segment_meta(segment: GeoTensor) -> tuple[int, int]:
    attrs = segment.attrs or {}
    meta = attrs.get("__geotoolz_segment_meta__")
    if not isinstance(meta, dict):
        raise ValueError("Each segment must define attrs['__geotoolz_segment_meta__'].")
    try:
        return int(meta["segment_index"]), int(meta["n_segments"])
    except KeyError as exc:
        raise ValueError(
            "Segment metadata requires segment_index and n_segments."
        ) from exc


def _validate_segment_compatible(
    first: GeoTensor,
    segment: GeoTensor,
    axis_num: int,
) -> None:
    if CRS.from_user_input(segment.crs) != CRS.from_user_input(first.crs):
        raise ValueError("All segments must share a CRS.")
    if axis_num == -2 and (
        segment.shape[:-2] != first.shape[:-2] or segment.shape[-1] != first.shape[-1]
    ):
        raise ValueError("All scan segments must share band and sample dimensions.")
    if axis_num == -1 and (
        segment.shape[:-2] != first.shape[:-2] or segment.shape[-2] != first.shape[-2]
    ):
        raise ValueError("All sample segments must share band and scan dimensions.")
    if not np.isclose(segment.transform.a, first.transform.a) or not np.isclose(
        segment.transform.e,
        first.transform.e,
    ):
        raise ValueError("All segments must share pixel size.")
