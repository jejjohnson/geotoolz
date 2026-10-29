"""Tier-B Operators — carrier-aware geometry transforms.

Each Operator wraps either a primitive in
:mod:`geoops.geom._src.array`, a `geoarray` geometry function (slicing,
``read_window``, ``pad``), or a `georeader` engine function
(``read.read_to_crs``, ``mosaic.spatial_mosaic``, ``rasterize.*``,
``vectorize.get_polygons``, ``griddata.georreference``). Every engine call
goes through :mod:`geoops._src.engine` (``to_engine`` in, ``from_engine`` /
``adopt_attrs`` out), so no georeader ``GeoTensor`` ever leaves an
operator. The two-tier discipline (``_src/array.py`` for numpy
primitives, ``_src/operators.py`` for ``GeoArray`` wrappers) mirrors the
rest of geotoolz.

Operators that carry concrete ``GeoArray`` / ``GeoDataFrame`` /
``Affine`` references in their constructor — `ReprojectLike`,
`ResampleLike`, `RasterizeLike`, `Rasterize`, `Georeference`, and
`Stitch` with a pinned target grid — set
``forbid_in_yaml = True`` because those objects are not JSON-safe and
hydra-zen ``builds()`` cannot recreate them from ``get_config()``.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, cast

import geoarray
import numpy as np
from affine import Affine
from geoarray import DIMS, FrozenAttrs, GeoArray, GridMismatchError
from pipekit import Operator, get_adapter
from pipekit_array import array_namespace
from pyproj import CRS
from scipy.ndimage import shift as ndi_shift

from geoops._src.bands import band_count
from geoops._src.blending import normalize_overlap_add, overlap_add
from geoops._src.config import as_tuple
from geoops._src.dtype import as_float
from geoops._src.engine import from_engine, to_engine
from geoops._src.geo import (
    affine_of,
    as_pixel_window,
    grid_matches,
    pixel_xy,
    reject_georeader,
    reject_geotensor,
    require_geoarray,
    require_grid_match,
)
from geoops._src.shape import BAND_AXIS, over_frames, require_ndim, single_band
from geoops._src.valid import (
    carried_fill,
    invalid_pixel_mask,
    mask_invalid_to_nan,
    valid_pixels,
    wrap_filled,
)
from geoops._src.wrap import adopt_attrs, rewrap_attrs, rewrap_dims, wrap_like
from geoops.geom._src.array import (
    adaptive_gradient,
    bowtie_detector_positions,
    center_offsets,
    feather_weights,
    gradient,
    is_north_up,
    resolve_interpolation,
    resolve_resampling,
    scan_angles,
    scan_pixel_growth,
    target_slices,
    valid_pixel_mask,
)


_RAY_DISCRIMINANT_TOLERANCE = 1.0e-6

#: attrs keys `AntimeridianSplit` reads longitudes from, in order.
_LON_KEYS: tuple[str, ...] = ("lons", "lon", "longitude")

#: attrs key holding a `SegmentStitch` segment's ``segment_index`` /
#: ``n_segments`` mapping.
_SEGMENT_META = "__geotoolz_segment_meta__"


if TYPE_CHECKING:
    import geopandas as gpd
    from shapely.geometry.base import BaseGeometry

from geoops._src.optional import import_optional, optional_name, optional_root


if TYPE_CHECKING:
    import rasterio.windows


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


def _mean_imputed(band: np.ndarray) -> np.ndarray:
    """``band`` with its ``NaN`` (nodata) pixels set to the valid mean.

    A fill value such as ``-9999`` dominates a cross-correlation; the mean
    is neutral in the mean-subtracted spectrum.
    """
    finite = np.isfinite(band)
    if finite.all():
        return band
    return np.where(finite, band, band[finite].mean() if finite.any() else 0.0)


def _read_window_boundless(
    arr: np.ndarray, window: rasterio.windows.Window
) -> np.ndarray:
    """Zero-padded boundless window read of a plain array.

    Slices the window from ``arr`` and zero-pads any part that extends
    past the bottom/right edges (a plain array has no
    ``fill_value`` to use as the pad sentinel).
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


def _require_carrier_list(items: Any, op_name: str, what: str) -> None:
    """Refuse a single carrier where a list of them is expected.

    Raises:
        TypeError: ``items`` is a georeader ``GeoTensor`` (naming
            ``from_georeader``) or a single `GeoArray` / array rather than
            a sequence of `GeoArray`s.
    """
    reject_geotensor(items, op_name)
    if isinstance(items, GeoArray | np.ndarray) or not isinstance(items, Sequence):
        raise TypeError(
            f"{op_name} takes a list of GeoArray {what}; got {type(items).__name__}."
        )


def _same_fill(a: float | int, b: float | int) -> bool:
    """Whether two fill values are the same (``NaN`` matches ``NaN``)."""
    return bool(a == b) or bool(np.isnan(a) and np.isnan(b))


def _resize_to_shape(
    gt: GeoArray,
    shape: tuple[int, int],
    *,
    anti_aliasing: bool,
    interpolation: str,
) -> GeoArray:
    """Resize through georeader's engine with a per-axis transform.

    georeader's ``GeoTensor.resize`` scales the transform's *x* axis by the
    *height* ratio, which is only right when both axes are scaled by the
    same factor. Rebuild the transform from the per-axis pixel-space scale
    so the output keeps the input's geographic extent.
    """
    height, width = gt.shape[-2:]
    out_h, out_w = int(shape[0]), int(shape[1])
    feature = "geoops.geom.Resize / Resample"
    engine = to_engine(gt, feature)
    # georeader's resize is skimage.transform.resize underneath.
    import_optional("skimage.transform", "morph", feature=feature)
    resized = from_engine(
        engine.resize(
            output_shape=(out_h, out_w),
            anti_aliasing=anti_aliasing,
            interpolation=interpolation,
        )
    )
    transform = affine_of(gt) * Affine.scale(width / out_w, height / out_h)
    return wrap_like(
        gt,
        np.asarray(resized),
        transform=transform,
        fill_value=resized.fill_value,
    )


def _copied_attrs(ref: GeoArray, values: np.ndarray) -> dict[str, Any]:
    """Fresh attrs for a new-grid output built from ``ref`` (see ``rewrap_attrs``)."""
    return rewrap_attrs(
        ref.attrs, in_bands=band_count(ref.shape), out_bands=band_count(values.shape)
    )


def _mosaic_frame(gts: Sequence[GeoArray]) -> tuple[Affine, rasterio.windows.Window]:
    """Union-of-bounds frame on the first input's pixel grid.

    Mirrors the frame :func:`georeader.mosaic.spatial_mosaic` builds
    (first raster's CRS + pixel grid, outer-rounded to cover every
    input) without warping any data. Returns the frame transform and the
    ``(0, 0, W, H)`` output window.
    """
    rasterio = optional_root("rasterio.windows", "raster", feature="geoops.geom")
    transform_bounds = optional_name(
        "rasterio.warp",
        "transform_bounds",
        "raster",
        feature="geoops.geom",
    )
    window_utils = optional_name(
        "georeader",
        "window_utils",
        "sensors",
        feature="geoops.geom",
    )
    first = gts[0]
    first_transform = affine_of(first)
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
        rasterio.windows.from_bounds(*union, transform=first_transform)
    )
    transform = rasterio.windows.transform(window, first_transform)
    return transform, rasterio.windows.Window(
        col_off=0, row_off=0, width=window.width, height=window.height
    )


def _nan_masked(gt: GeoArray) -> GeoArray:
    """``float32`` copy of ``gt`` with its fill sentinel replaced by NaN.

    NaN then serves as both source and destination nodata when warping
    onto the mosaic frame, which also covers inputs whose
    ``fill_value`` is ``None``.
    """
    arr = np.asarray(gt, dtype=np.float32)
    fill = gt.fill_value
    if fill is not None and not np.isnan(fill):
        arr = np.where(arr == fill, np.float32(np.nan), arr)
    return gt.with_data(arr, fill_value=np.nan)


def _nan_first(stack: np.ndarray, axis: int = 0) -> np.ndarray:
    """First non-NaN value along ``axis`` (NaN where every entry is NaN)."""
    index = np.expand_dims(np.argmax(~np.isnan(stack), axis=axis), axis)
    return np.take_along_axis(stack, index, axis=axis).squeeze(axis)


class Reproject(Operator):
    """Reproject a `GeoArray` to a destination CRS (and optional resolution).

    Thin Operator wrapping :func:`georeader.read.read_to_crs`. The
    destination grid is anchored to the input's bounding box reprojected
    into ``dst_crs``; the resolution defaults to the input's pixel size
    expressed in the destination CRS.

    Geo-dependent: requires a georeferenced ``GeoArray`` input
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
        >>> import geoops as gz
        >>> # Reproject a UTM scene into geographic coords for plotting.
        >>> reproj = gz.geom.Reproject(
        ...     dst_crs="EPSG:4326", resampling="bilinear"
        ... )
        >>> geographic = reproj(utm_scene)
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

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

    def _apply(self, gt: GeoArray) -> GeoArray:
        read = optional_name(
            "georeader", "read", "sensors", feature="geoops.geom.Reproject"
        )
        require_geoarray(gt, "Reproject")
        resampling = resolve_resampling(self.resampling)
        engine_gt = to_engine(gt)
        if self.resolution is None:
            return adopt_attrs(
                gt, read.read_to_crs(engine_gt, self.dst_crs, resampling=resampling)
            )
        # ``read_to_crs`` returns its input untouched when the CRS already
        # matches, silently dropping ``resolution``; build the destination
        # grid explicitly so a same-CRS call still resamples.
        window, dst_transform = read.calculate_transform_window(
            engine_gt, self.dst_crs, self.resolution
        )
        return adopt_attrs(
            gt,
            read.read_reproject(
                engine_gt,
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
    """Reproject a `GeoArray` onto another `GeoArray`'s CRS + grid.

    Wraps :func:`georeader.read.read_reproject_like`: the output has the
    same CRS, transform, and spatial shape as ``like``. Useful for
    aligning auxiliary data (DEMs, masks, predictions from another
    sensor) onto a reference scene grid before downstream pipelines.

    Geo-dependent: requires a georeferenced ``GeoArray`` input
    (transform + CRS); plain arrays raise ``TypeError``.

    Note:
        ``like`` is a concrete ``GeoArray``, so this Operator is
        flagged ``forbid_in_yaml = True``.

    Args:
        like: The reference `GeoArray` whose grid (CRS + transform +
            spatial shape) is matched.
        resampling: See :class:`Reproject`.

    Examples:
        >>> import geoops as gz
        >>> # Align a coarse cloud mask onto the scene grid.
        >>> aligned_mask = gz.geom.ReprojectLike(
        ...     like=scene, resampling="nearest"
        ... )(coarse_mask)
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(self, *, like: GeoArray, resampling: str = "bilinear") -> None:
        self.like = like
        self.resampling = resampling

    def _apply(self, gt: GeoArray) -> GeoArray:
        read = optional_name(
            "georeader", "read", "sensors", feature="geoops.geom.ReprojectLike"
        )
        name = type(self).__name__
        require_geoarray(gt, name)
        reject_georeader(self.like, f"{name} (like)")
        return adopt_attrs(
            gt,
            read.read_reproject_like(
                to_engine(gt),
                to_engine(self.like),
                resampling=resolve_resampling(self.resampling),
            ),
        )

    def get_config(self) -> dict[str, Any]:
        return {"like": repr(self.like), "resampling": self.resampling}


class ResampleLike(ReprojectLike):
    """Resample a `GeoArray` onto another `GeoArray`'s spatial grid.

    Alias of :class:`ReprojectLike`. The intent of the rename is purely
    pipeline-readability: in workflows where both inputs already share a
    CRS, ``ResampleLike`` reads as "match this grid" rather than
    "warp into a new projection".

    Examples:
        >>> import geoops as gz
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
    ``(dy, dx)`` shift (nodata pixels of either registration band set to
    that band's valid mean, so a fill value never drives the
    correlation) is applied with linear interpolation
    (``scipy.ndimage.shift``) when ``apply=True``.

    The registration math is pixel-space, so plain ``np.ndarray`` inputs
    are supported: they come back as plain shifted arrays. For
    ``GeoArray`` input the shifted array keeps the input's affine
    transform (which must share the reference's grid), so a feature at
    pixel ``(r, c)`` of the output sits at the same world coordinate as
    pixel ``(r, c)`` of the reference. Nodata pixels (fill or
    non-finite) are shifted as nodata, and edge pixels the shift uncovers
    have no data either: every output pixel interpolated from one holds
    the input's fill value (``NaN`` for a fill-less float input), so a
    sub-pixel shift grows a nodata region by up to one pixel. An integer
    input without a ``fill_value`` cannot mark nodata; its
    uncovered edge repeats the nearest edge pixel.

    Called as ``op(moving, reference)``: ``reference`` is the fixed scene
    the input is registered against, on the input's pixel grid
    (``ValueError`` otherwise).

    Args:
        upsample_factor: Sub-pixel upsampling factor passed to
            :func:`skimage.registration.phase_cross_correlation`.
        band: Band index (along the band axis) used for registration on
            3-D inputs.
        apply: If ``True`` (default), return the shifted image; if
            ``False``, return the raw ``(shift_y, shift_x, error)``
            tuple instead; the operator is then terminal (last step
            only) in a ``Sequential``.
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        upsample_factor: int = 10,
        band: int = 0,
        apply: bool = True,
    ) -> None:
        self.upsample_factor = upsample_factor
        self.band = band
        self.apply = apply

    @property
    def _terminal(self) -> bool:  # ty: ignore[invalid-attribute-override]
        """The ``apply=False`` tuple breaks carrier-in / carrier-out."""
        return not self.apply

    def _apply(
        self, gt: GeoArray | np.ndarray, reference: GeoArray | np.ndarray
    ) -> GeoArray | np.ndarray | tuple[float, float, float]:
        phase_cross_correlation = optional_name(
            "skimage.registration",
            "phase_cross_correlation",
            "morph",
            feature="geoops.geom.PhaseAlign",
        )
        require_grid_match(gt, reference, "PhaseAlign", names=("moving", "reference"))
        arr = np.asarray(gt)
        mov_band = _mean_imputed(
            _registration_band(
                mask_invalid_to_nan(gt, dtype=np.float64), self.band, "PhaseAlign"
            )
        )
        ref_band = _mean_imputed(
            _registration_band(
                mask_invalid_to_nan(reference, dtype=np.float64),
                self.band,
                "PhaseAlign reference",
            )
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
        # Shift the validity alongside the values so the linear
        # interpolation never blends a fill value into valid data: an
        # output pixel that draws any weight from a nodata pixel -- or from
        # beyond the edge -- is nodata. An integer carrier without a fill
        # value cannot mark nodata, so its uncovered edge repeats the edge
        # pixels instead.
        inexact = np.issubdtype(arr.dtype, np.inexact)
        fill = carried_fill(gt, arr.dtype)
        in_valid = valid_pixels(gt)
        values = np.where(in_valid, arr.astype(np.float64), 0.0)
        spatial_shift = (shift_y, shift_x)
        shifted = ndi_shift(
            values,
            shift=(0.0, *spatial_shift) if arr.ndim == 3 else spatial_shift,
            order=1,
            mode="nearest",
        )
        weight = ndi_shift(
            in_valid.astype(np.float64),
            shift=spatial_shift,
            order=1,
            mode="constant" if inexact or fill is not None else "nearest",
            cval=0.0,
        )
        valid = weight > 1.0 - 1e-9
        out = (
            shifted.astype(arr.dtype) if inexact else np.rint(shifted).astype(arr.dtype)
        )
        # The array shift already moves the content onto the reference
        # grid; keep the original transform (translating it as well would
        # cancel the alignment in world coordinates, #118).
        return wrap_filled(gt, out, fill_value=fill, valid=valid)


def _wrap_flow(gt: Any, reference: Any, flow: np.ndarray) -> Any:
    """Rewrap a ``(2, H, W)`` displacement field with ``NaN`` as its nodata.

    A displacement is a new quantity: pixels that are nodata in either
    scene (non-finite or equal to that scene's ``fill_value``)
    hold ``NaN`` and the output declares ``fill_value=NaN``.
    """
    valid = valid_pixels(gt) & valid_pixels(reference)
    return wrap_filled(gt, flow, fill_value=np.nan, valid=valid)


class OpticalFlowTVL1(Operator):
    """Dense per-pixel displacement via TV-L1 optical flow.

    Wraps :func:`skimage.registration.optical_flow_tvl1` on the selected
    registration band. The output is a ``(2, H, W)`` displacement field
    (``[dy, dx]`` per pixel) on the input's carrier: pixel-space math,
    so both ``GeoArray`` and plain ``np.ndarray`` inputs are supported
    and returned in kind. Pixels that are nodata in either scene are
    ``NaN`` (``fill_value=NaN``). Inputs are ``(H, W)`` or
    ``(C, H, W)``; a 4-D ``(T, C, H, W)`` stack raises ``ValueError``
    (compute the flow of each frame separately).

    Called as ``op(moving, reference)``: ``reference`` is the fixed scene
    the input is registered against, on the input's pixel grid
    (``ValueError`` otherwise).

    Args:
        band: Band index (along the band axis) used for registration on
            3-D inputs.
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(self, *, band: int = 0) -> None:
        self.band = band

    def _apply(
        self, gt: GeoArray | np.ndarray, reference: GeoArray | np.ndarray
    ) -> GeoArray | np.ndarray:
        optical_flow_tvl1 = optional_name(
            "skimage.registration",
            "optical_flow_tvl1",
            "morph",
            feature="geoops.geom.OpticalFlowTVL1",
        )
        name = type(self).__name__
        require_grid_match(gt, reference, name, names=("moving", "reference"))
        flow = np.asarray(
            optical_flow_tvl1(
                _registration_band(
                    np.asarray(reference), self.band, f"{name} reference"
                ),
                _registration_band(np.asarray(gt), self.band, name),
            )
        )
        return _wrap_flow(gt, reference, flow)


class OpticalFlowILK(OpticalFlowTVL1):
    """Dense per-pixel displacement via iterative Lucas-Kanade optical flow.

    Same contract as :class:`OpticalFlowTVL1` (both carriers supported),
    delegating to :func:`skimage.registration.optical_flow_ilk`.
    """

    def _apply(
        self, gt: GeoArray | np.ndarray, reference: GeoArray | np.ndarray
    ) -> GeoArray | np.ndarray:
        optical_flow_ilk = optional_name(
            "skimage.registration",
            "optical_flow_ilk",
            "morph",
            feature="geoops.geom.OpticalFlowILK",
        )
        name = type(self).__name__
        require_grid_match(gt, reference, name, names=("moving", "reference"))
        flow = np.asarray(
            optical_flow_ilk(
                _registration_band(
                    np.asarray(reference), self.band, f"{name} reference"
                ),
                _registration_band(np.asarray(gt), self.band, name),
            )
        )
        return _wrap_flow(gt, reference, flow)


class Resize(Operator):
    """Resize a `GeoArray` to a target spatial shape.

    Delegates to georeader's ``GeoTensor.resize`` (through the engine
    boundary), re-basing the affine transform so the geographic extent is preserved
    while the pixel grid is densified or coarsened. ``anti_aliasing``
    applies a Gaussian pre-filter when downscaling (recommended).

    Geo-dependent (transform-updating): requires a georeferenced
    ``GeoArray`` input; plain arrays raise ``TypeError``.

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
        >>> import geoops as gz
        >>> tile = gz.geom.Resize(shape=(256, 256))(scene)
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

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

    def _apply(self, gt: GeoArray) -> GeoArray:
        require_geoarray(gt, "Resize")
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
    """Resample a `GeoArray` to a target spatial resolution.

    The output shape is derived per axis from the ratio of the input
    pixel size to the requested resolution,
    ``(round(H * res_y / dst_y), round(W * res_x / dst_x))``, and the
    resize is delegated to georeader's ``GeoTensor.resize`` (through the
    engine boundary).
    The geographic extent is preserved, so the achieved pixel size
    matches ``resolution`` exactly only when it divides the extent.

    Geo-dependent: the target resolution is expressed in CRS units, so
    a georeferenced ``GeoArray`` input is required; plain arrays raise
    ``TypeError``.

    Args:
        resolution: Target ``(pixel_size_x, pixel_size_y)`` in
            input-CRS units.
        resampling: Interpolation mode. Default ``"bilinear"``.
        anti_aliasing: Whether to anti-alias before downscaling.
            Default ``True``.

    Examples:
        >>> import geoops as gz
        >>> # Downsample a 10 m raster to 30 m.
        >>> coarse = gz.geom.Resample(resolution=(30.0, 30.0))(fine_10m)
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

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

    def _apply(self, gt: GeoArray) -> GeoArray:
        require_geoarray(gt, "Resample")
        res_x, res_y = gt.res
        dst_x, dst_y = self.resolution
        height, width = gt.shape[-2:]
        # Derive the shape per axis here instead of passing
        # ``resolution_dst`` to georeader's ``resize``, which scales the
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
    """Pad a `GeoArray` up to a target spatial shape.

    Pads the spatial axes (``"y"``, ``"x"``) symmetrically so the output
    has *at least* ``shape`` pixels along each spatial axis. If the
    input already exceeds the target on an axis, that axis is left
    alone (no truncation — pair with :class:`CropTo` for that).

    The affine transform is re-based (``geoarray.pad`` for a constant
    pad with the carrier's own fill) so the *geographic* position of the
    existing data does not move; new rows appear to the top/bottom and
    new columns to the left/right. The output keeps the input's
    ``fill_value``.

    Pixel-space padding: plain ``np.ndarray`` inputs are supported and
    padded with :func:`numpy.pad` (returned as plain arrays); the
    transform bookkeeping only happens for ``GeoArray`` input.

    Args:
        shape: Target minimum spatial shape ``(H, W)``.
        mode: Numpy pad mode. ``"constant"`` (default), ``"edge"``,
            ``"reflect"``, ``"symmetric"``, ...
        fill_value: Constant value when ``mode == "constant"``. ``None``
            falls back to the carrier's ``fill_value``, or ``0``
            when the carrier has none (and for plain-array input).

    Examples:
        >>> import geoops as gz
        >>> # Pad to a multiple of 256 before tiling.
        >>> snug = gz.geom.PadTo(shape=(2048, 2048), fill_value=0)(scene)
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy", "jax")

    def __init__(
        self,
        *,
        shape: tuple[int, int],
        mode: str = "constant",
        fill_value: float | None = None,
    ) -> None:
        self.shape = shape
        self.mode = mode
        self.fill_value = fill_value

    def _apply(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        reject_geotensor(gt, "PadTo")
        height, width = gt.shape[-2:]
        pad_y = max(self.shape[0] - height, 0)
        pad_x = max(self.shape[1] - width, 0)
        if pad_y == 0 and pad_x == 0:
            return gt
        pad_yx = (
            (pad_y // 2, pad_y - pad_y // 2),
            (pad_x // 2, pad_x - pad_x // 2),
        )
        carrier_fill = gt.fill_value if isinstance(gt, GeoArray) else None
        constant = self.fill_value
        if constant is None:
            # No explicit constant: the carrier's fill, else 0.
            constant = 0 if carrier_fill is None else carrier_fill
        if (
            isinstance(gt, GeoArray)
            and self.mode == "constant"
            and carrier_fill is not None
            and _same_fill(constant, carrier_fill)
        ):
            # Padding with the carrier's own fill is geoarray's ``pad``.
            return geoarray.pad(gt, pad_yx)
        data = gt.data if isinstance(gt, GeoArray) else np.asarray(gt)
        xp = gt.xp if isinstance(gt, GeoArray) else np
        pad_width = [(0, 0)] * (data.ndim - 2) + list(pad_yx)
        if self.mode == "constant":
            padded = xp.pad(data, pad_width, mode="constant", constant_values=constant)
        else:
            # `mode` is user-supplied (a numpy / jax.numpy pad-mode name);
            # the stubs only accept the literal names, so widen it for ty.
            padded = xp.pad(data, pad_width, mode=cast("Any", self.mode))
        if not isinstance(gt, GeoArray):
            return padded
        # geoarray's pad moves the transform on either backend (and under
        # jax.jit); the pixels are the ones padded above
        moved = geoarray.pad(gt.replace(fill_value=None), pad_yx, fill_value=0)
        return moved.with_data(padded, fill_value=carrier_fill)

    def get_config(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "mode": self.mode,
            "fill_value": self.fill_value,
        }


class CropTo(Operator):
    """Crop a `GeoArray` to a target spatial shape.

    Anchors either at the centre (default) or upper-left corner of the
    input. The affine transform is updated by slicing the `GeoArray`, so
    the geographic extent of the surviving pixels is preserved.

    Pixel-space cropping: plain ``np.ndarray`` inputs are supported
    (sliced and returned as plain arrays); the transform bookkeeping
    only happens for ``GeoArray`` input.

    Args:
        shape: Target spatial shape ``(H, W)``. Must be ``<=`` the
            input shape; oversize crops raise ``ValueError``.
        anchor: ``"center"`` (default) or ``"upper_left"``.

    Examples:
        >>> import geoops as gz
        >>> # Centre-crop a 1024×1024 scene to a 512×512 chip.
        >>> chip = gz.geom.CropTo(shape=(512, 512))(scene)
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy", "jax")

    def __init__(self, *, shape: tuple[int, int], anchor: str = "center") -> None:
        self.shape = shape
        self.anchor = anchor

    def _apply(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        reject_geotensor(gt, "CropTo")
        height, width = gt.shape[-2:]
        if self.shape[0] > height or self.shape[1] > width:
            raise ValueError(f"Cannot crop shape {(height, width)} to {self.shape}.")
        if self.anchor == "center":
            row_off, col_off = center_offsets((height, width), self.shape)
        elif self.anchor == "upper_left":
            row_off, col_off = 0, 0
        else:
            raise ValueError("anchor must be 'center' or 'upper_left'.")
        rows = slice(row_off, row_off + self.shape[0])
        cols = slice(col_off, col_off + self.shape[1])
        if not isinstance(gt, GeoArray):
            return np.asarray(gt)[..., rows, cols]
        return gt[..., rows, cols]

    def get_config(self) -> dict[str, Any]:
        return {"shape": list(self.shape), "anchor": self.anchor}


class CropToBounds(Operator):
    """Crop a `GeoArray` to a geographic bounding box.

    Thin wrapper over :func:`georeader.read.read_from_bounds` with
    ``boundless=False``: ``bounds`` are reprojected into the carrier's
    CRS if ``crs`` differs from it, the pixel window is rounded
    *outward* (every pixel the bounds touch, even partially, is kept —
    matching georeader's readers), and the intersection with the carrier
    is read. Non-intersecting bounds raise
    :class:`rasterio.windows.WindowError`.

    Geo-dependent: requires a georeferenced ``GeoArray`` input
    (transform + CRS); plain arrays raise ``TypeError``.

    Args:
        bounds: ``(minx, miny, maxx, maxy)`` in CRS units of ``crs``.
        crs: CRS of ``bounds``. ``None`` (default) means "same as the
            carrier".

    Examples:
        >>> import geoops as gz
        >>> # Crop a global mosaic to a small AOI in geographic coords.
        >>> aoi = gz.geom.CropToBounds(
        ...     bounds=(-10.0, 35.0, -8.0, 36.5), crs="EPSG:4326"
        ... )
        >>> patch = aoi(global_scene)
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        bounds: tuple[float, float, float, float],
        crs: str | None = None,
    ) -> None:
        self.bounds = bounds
        self.crs = crs

    def _apply(self, gt: GeoArray) -> GeoArray:
        rasterio = optional_root(
            "rasterio.windows", "raster", feature="geoops.geom.CropToBounds"
        )
        read = optional_name(
            "georeader", "read", "sensors", feature="geoops.geom.CropToBounds"
        )
        require_geoarray(gt, "CropToBounds")
        cropped = read.read_from_bounds(
            to_engine(gt), tuple(self.bounds), crs_bounds=self.crs, boundless=False
        )
        if cropped is None:
            # georeader signals an empty intersection with ``None``.
            raise rasterio.windows.WindowError(
                f"CropToBounds: bounds {tuple(self.bounds)!r} (crs={self.crs!r}) "
                f"do not intersect the carrier's bounds {gt.bounds!r}."
            )
        return adopt_attrs(gt, cropped)

    def get_config(self) -> dict[str, Any]:
        return {"bounds": list(self.bounds), "crs": self.crs}


class Tile(Operator):
    """Split a `GeoArray` into spatial tiles.

    Wraps :func:`georeader.slices.create_windows` (with
    ``trim_incomplete=False``) and then reads each window with
    ``boundless=True`` so the trailing-edge tiles match ``size``
    exactly. Out-of-bounds areas are padded with the carrier's
    ``fill_value``; use :func:`Stitch` (which masks the
    sentinel) to recover the original extent. A ``GeoArray`` with
    ``fill_value=None`` is zero-padded instead — ``Stitch``
    cannot tell that padding from data, so pin its ``target_shape`` /
    ``target_transform`` to the source grid to drop it.

    Pixel-space tiling: plain ``np.ndarray`` inputs are supported and
    yield a list of plain arrays (out-of-bounds areas zero-padded, since
    a plain array carries no ``fill_value``). Note that
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
        >>> import geoops as gz
        >>> # 512x512 tiles with 64-pixel overlap.
        >>> tiles = gz.geom.Tile(size=(512, 512), stride=(448, 448))(scene)
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

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

    def _apply(self, gt: GeoArray | np.ndarray) -> list[GeoArray] | list[np.ndarray]:
        rasterio = optional_root(
            "rasterio.windows", "raster", feature="geoops.geom.Tile"
        )
        slices = optional_name(
            "georeader",
            "slices",
            "sensors",
            feature="geoops.geom.Tile",
        )
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
        reject_geotensor(gt, type(self).__name__)
        if not isinstance(gt, GeoArray):
            arr = np.asarray(gt)
            return [_read_window_boundless(arr, window) for window in windows]
        if gt.fill_value is not None:
            return [
                gt.read_window(as_pixel_window(window), boundless=True)
                for window in windows
            ]
        # ``read_window(boundless=True)`` refuses to pad without a fill
        # value; zero-pad like the plain-array path (the fill stays None).
        transform = affine_of(gt)
        return [
            wrap_like(
                gt,
                _read_window_boundless(np.asarray(gt), window),
                transform=rasterio.windows.transform(window, transform),
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
    """Split a `GeoArray` into overlapping sliding-window tiles.

    Convenience subclass of :class:`Tile` that takes a scalar
    ``overlap`` (in pixels) and derives the stride. Equivalent to
    ``Tile(size=size, stride=(size[0] - overlap, size[1] - overlap))``.

    Args:
        size: Tile spatial shape ``(H, W)``.
        overlap: Overlap in pixels along both axes. Default ``0``.

    Examples:
        >>> import geoops as gz
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
    r"""Stitch georeferenced tiles back into one `GeoArray`.

    Blends a ``list[GeoArray]`` (typically the output of :class:`Tile`)
    onto a unioning north-up grid, masking out the
    ``fill_value`` sentinels that :class:`Tile` introduces
    along the bottom / right edges with ``boundless=True``.

    Supported blend modes:

    - ``"average"`` — unweighted mean of overlapping pixels.
    - ``"feather"`` — overlap-add with triangular edge weights, kernel from
      :func:`geoops.geom._src.array.feather_weights`.
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
    ``fill_value`` (default: the first tile's ``fill_value``).

    Unless ``target_shape`` is pinned, trailing bottom rows / right
    columns with no valid contributor (the sentinel padding
    :class:`Tile` adds to its edge tiles) are cropped off, so a
    ``Tile`` → ``Stitch`` round trip recovers the source extent. Pin
    ``target_shape`` to keep them.

    Geo-dependent: every tile must be a georeferenced ``GeoArray`` —
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
        fill_value: Output fill value. ``None`` (default) inherits from the
            first tile.

    Examples:
        >>> import geoops as gz
        >>> # Round-trip a tiled inference pipeline.
        >>> tiles = gz.geom.Tile(size=(256, 256))(scene)
        >>> predictions = [model(tile) for tile in tiles]
        >>> stitched = gz.geom.Stitch(blend="feather", feather_width=16)(
        ...     predictions
        ... )
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        blend: str = "average",
        feather_width: int = 16,
        target_shape: tuple[int, int] | None = None,
        target_transform: Affine | Sequence[float] | None = None,
        target_crs: str | None = None,
        fill_value: float | int | None = None,
    ) -> None:
        self.blend = blend
        self.feather_width = feather_width
        self.target_shape = as_tuple(target_shape)
        if target_transform is not None and not isinstance(target_transform, Affine):
            target_transform = Affine(*target_transform[:6])
        self.target_transform = target_transform
        self.target_crs = target_crs
        self.fill_value = fill_value

    def _apply(self, tiles: list[GeoArray]) -> GeoArray:
        _require_carrier_list(tiles, "Stitch", "tiles")
        if len(tiles) == 0:
            raise ValueError("Stitch requires at least one tile.")
        if self.blend not in {"average", "feather", "max", "first"}:
            raise ValueError("blend must be 'average', 'feather', 'max', or 'first'.")
        first = tiles[0]
        for index, tile in enumerate(tiles):
            require_geoarray(tile, "Stitch", arg=f"tile at index {index}")
            if not is_north_up(affine_of(tile)):
                raise ValueError(
                    f"Stitch only supports north-up, non-rotated GeoArrays; "
                    f"tile {index} has a rotated/sheared or flipped transform "
                    f"({tuple(np.asarray(tile.transform).tolist())})."
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
        fill = first.fill_value if self.fill_value is None else self.fill_value
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
        return GeoArray(
            values,
            transform,
            self.target_crs or first.crs,
            fill_value=fill,
            # a carrier's attrs, carried: the 0.2 per-band keys stay as they are
            attrs=FrozenAttrs(_copied_attrs(first, values)),
            dims=DIMS[-values.ndim :],
        )

    def _target_grid(
        self,
        tiles: list[GeoArray],
    ) -> tuple[Affine, tuple[int, int]]:
        first = tiles[0]
        if self.target_transform is not None:
            transform = self.target_transform
        else:
            bounds = [tile.bounds for tile in tiles]
            minx = min(bound[0] for bound in bounds)
            maxy = max(bound[3] for bound in bounds)
            first_transform = affine_of(first)
            transform = Affine(
                first_transform.a, 0.0, minx, 0.0, first_transform.e, maxy
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
        tiles: list[GeoArray],
        transform: Affine,
        values: np.ndarray,
        weights: np.ndarray,
        fill: float | int | None,
    ) -> None:
        for tile in tiles:
            arr = np.asarray(tile)
            out_slices, tile_slices = target_slices(
                affine_of(tile), tile.shape[-2:], transform, weights.shape
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
        tiles: list[GeoArray],
        transform: Affine,
        values: np.ndarray,
        filled: np.ndarray,
        fill: float | int | None,
    ) -> None:
        for tile in tiles:
            arr = np.asarray(tile)
            out_slices, tile_slices = target_slices(
                affine_of(tile), tile.shape[-2:], transform, filled.shape
            )
            mask_tile = valid_pixel_mask(arr, fill)[tile_slices]
            mask = (~filled[out_slices]) & mask_tile
            values_window = values[..., out_slices[0], out_slices[1]]
            tile_window = arr[..., tile_slices[0], tile_slices[1]]
            values_window[..., mask] = tile_window[..., mask]
            filled[out_slices] |= mask

    def _stitch_max(
        self,
        tiles: list[GeoArray],
        transform: Affine,
        values: np.ndarray,
        filled: np.ndarray,
        fill: float | int | None,
    ) -> None:
        for tile in tiles:
            arr = np.asarray(tile)
            out_slices, tile_slices = target_slices(
                affine_of(tile), tile.shape[-2:], transform, filled.shape
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
            "fill_value": self.fill_value,
        }


#: Mean Earth radius R₁ (km), IUGG / Moritz (2000), "Geodetic Reference
#: System 1980", J. Geodesy 74:128-133.
_EARTH_MEAN_RADIUS_KM: float = 6371.0088

# MODIS: 10 / 20 / 40 detectors per scan at 1 km / 500 m / 250 m; 1354 /
# 2708 / 5416 frames over ±55°; Terra/Aqua nominal altitude 705 km.
# Barnes, Pagano & Salomonson (1998), IEEE TGRS 36(4):1088-1100;
# Wolfe et al. (2002), Remote Sens. Environ. 83:31-49; MCST MODIS Level 1B
# Product User's Guide.
_MODIS_DETECTORS: dict[int, int] = {1000: 10, 500: 20, 250: 40}
_MODIS_MAX_SCAN_ANGLE_DEG = 55.0
_MODIS_ALTITUDE_KM = 705.0

# VIIRS: 16 M-band / 32 I-band detectors per scan over ±56.28°, S-NPP /
# JPSS nominal altitude 824 km, three along-scan aggregation zones per
# half-scan (nadir → 31.72°: 3 native samples per pixel; → 44.86°: 2;
# → 56.28°: 1), i.e. 1776 + 736 + 640 native M-band samples giving
# 592 + 368 + 640 pixels (3200 per scan; I-band doubles every count, 6400).
# NOAA Technical Report NESDIS 142, "VIIRS Sensor Data Record (SDR) User's
# Guide" (Cao et al.), §2.1-2.2 and Fig. 3.
_VIIRS_DETECTORS: dict[str, int] = {"M": 16, "I": 32}
_VIIRS_MAX_SCAN_ANGLE_DEG = 56.28
_VIIRS_ALTITUDE_KM = 824.0
_VIIRS_M_ZONES: tuple[tuple[int, int], ...] = ((1776, 3), (736, 2), (640, 1))


class BowtieCorrection(Operator):
    """Remove whiskbroom bowtie overlap by resampling each scan along track.

    A whiskbroom scanner (MODIS, VIIRS) images ``N`` detector rows per
    mirror scan. The along-track footprint of a detector grows with scan
    angle θ, while the satellite advances by the same distance every scan,
    so consecutive scans overlap along track towards the swath edges: the
    same ground appears in the last rows of scan ``j`` and the first rows of
    scan ``j + 1`` (the "bowtie" effect). This operator removes that
    duplication, per column, from the scan geometry alone.

    **Geometry** (spherical Earth, radius ``R``; altitude ``h``; see
    :func:`~geoops.geom._src.array.scan_pixel_growth`)::

        β(θ) = arcsin(((R + h)/R)·sin θ) − θ          Earth central angle
        s(θ) = (R + h)·cos θ − √(R² − (R + h)²·sin² θ)  slant range
        g(θ) = s / (h·cos β)                          along-track growth

    ``g`` is the along-track size of a detector, measured in orbit angle
    (the coordinate in which every scan advances by the same amount),
    relative to nadir; ``g(0) = 1`` and the flat-Earth limit is
    ``1 / cos θ``. The column scan angles ``θ_c`` follow the constant
    mirror rate (see :func:`~geoops.geom._src.array.scan_angles`),
    including VIIRS along-scan aggregation when ``aggregation_zones`` is
    given.

    **De-overlap.** In units of one nadir detector, scans are contiguous
    at nadir (scan ``j`` spans rows ``j·N … j·N + N − 1``) and detector
    ``k`` of scan ``j`` lies at ``y = j·N + (N − 1)/2 + (k − (N − 1)/2)·g``.
    Output row ``r = j·N + m`` is the ground at ``y = r`` — a uniform
    along-track grid, identical to the input grid at nadir — sampled from
    the scan whose centre is nearest (scan ``j``) at fractional detector
    ``k₀ = (N − 1)/2 + (m − (N − 1)/2)/g`` (nearest or linear in ``k``).
    Each scan is therefore shrunk about its centre by ``1/g`` so the rows
    it shares with a neighbour are taken once, from the nearer scan. If
    that sample is nodata, the overlapping neighbour scan ``j ± 1`` is
    used instead when its footprint covers ``y = r``; otherwise the output
    is fill.

    **Output grid.** Same shape, transform and CRS as the input; only
    along-track values change. Columns are never moved (cross-track pixel
    growth is left for a swath-to-grid resampler) and the along-track
    extent is conserved.

    **Assumptions / limitations.** Spherical Earth, circular orbit at a
    constant altitude, no terrain; scans exactly contiguous at nadir;
    detectors equally spaced along track (first order in the detector
    angle, error ~ (N·IFOV)² ≈ 10⁻⁴ of a row); the along-track shift of a
    scan line caused by satellite motion and Earth rotation during a scan
    is common to all scans of a column and does not change the overlap, so
    it is ignored. Input rows must be whole scans (``H % N == 0``) with
    row 0 the first detector of a scan.

    **VIIRS.** On-board aggregation makes the columns non-uniform in angle;
    pass ``aggregation_zones`` (the ``viirs`` preset does). On-board
    *bowtie deletion* (1 row per scan edge in the 2-sample zone, 2 in the
    unaggregated zone, M-band; doubled for I-bands) delivers the duplicated
    rows as fill: they are nodata here, so the neighbouring scan's copy of
    that ground is used and no gap remains.

    Nodata: a pixel is invalid when any band is non-finite or equals the
    carrier's fill; invalid samples never contribute to an output pixel,
    and output pixels without a valid source hold the output's fill
    (``carried_fill``: the input's fill for ``"nearest"``; ``NaN`` for
    ``"linear"`` on integer input, which is promoted to float).
    ``(T, C, H, W)`` stacks are corrected frame by frame.

    Pixel-space resampling (driven entirely by the scan-geometry
    constructor arguments): both ``GeoArray`` and plain ``np.ndarray``
    inputs are supported and returned in kind.

    Args:
        detectors_per_scan: Detector rows per scan ``N``.
        max_scan_angle_deg: Scan-edge angle ``θ_max`` in degrees (outer edge
            of the outermost native sample); ``0`` is the identity.
        altitude_km: Orbit altitude ``h`` (km).
        earth_radius_km: Earth radius ``R`` (km); defaults to the mean
            radius 6371.0088 km.
        aggregation_zones: Optional along-scan aggregation table,
            ``(native_samples, factor)`` per zone from nadir outward for one
            half-scan (VIIRS); ``None`` means one column per native sample.
        method: ``"nearest"`` (dtype-preserving, safe for masks / QA) or
            ``"linear"`` interpolation between detectors.

    Examples:
        >>> import geoops as gz
        >>> debowtie = gz.geom.BowtieCorrection.modis()  # 1 km bands
        >>> viirs = gz.geom.BowtieCorrection.viirs(band="M", method="linear")
        >>> custom = gz.geom.BowtieCorrection(
        ...     detectors_per_scan=10, max_scan_angle_deg=55.0, altitude_km=705.0
        ... )
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        detectors_per_scan: int,
        max_scan_angle_deg: float,
        altitude_km: float,
        earth_radius_km: float = _EARTH_MEAN_RADIUS_KM,
        aggregation_zones: Sequence[Sequence[int]] | None = None,
        method: str = "nearest",
    ) -> None:
        if detectors_per_scan < 1:
            raise ValueError("detectors_per_scan must be >= 1.")
        if altitude_km <= 0 or earth_radius_km <= 0:
            raise ValueError("altitude_km and earth_radius_km must be > 0.")
        horizon = np.rad2deg(
            np.arcsin(earth_radius_km / (earth_radius_km + altitude_km))
        )
        if not 0.0 <= max_scan_angle_deg < horizon:
            raise ValueError(
                f"max_scan_angle_deg must be in [0, {horizon:.2f}) — the horizon "
                f"seen from {altitude_km} km."
            )
        if method not in {"nearest", "linear"}:
            raise ValueError("method must be 'nearest' or 'linear'.")
        self.detectors_per_scan = int(detectors_per_scan)
        self.max_scan_angle_deg = max_scan_angle_deg
        self.altitude_km = altitude_km
        self.earth_radius_km = earth_radius_km
        self.aggregation_zones = (
            None
            if aggregation_zones is None
            else tuple(
                (int(native), int(factor)) for native, factor in aggregation_zones
            )
        )
        self.method = method

    @classmethod
    def modis(
        cls, resolution_m: int = 1000, *, method: str = "nearest"
    ) -> BowtieCorrection:
        """MODIS (Terra / Aqua) preset.

        10 / 20 / 40 detectors per scan at 1 km / 500 m / 250 m, ±55°,
        705 km (Barnes et al. 1998; Wolfe et al. 2002).

        Args:
            resolution_m: Band resolution: ``1000``, ``500`` or ``250``.
            method: ``"nearest"`` or ``"linear"``.

        Returns:
            The configured operator.

        Examples:
            >>> BowtieCorrection.modis().detectors_per_scan
            10
            >>> BowtieCorrection.modis(250).detectors_per_scan
            40
        """
        if resolution_m not in _MODIS_DETECTORS:
            raise ValueError(f"resolution_m must be one of {sorted(_MODIS_DETECTORS)}.")
        return cls(
            detectors_per_scan=_MODIS_DETECTORS[resolution_m],
            max_scan_angle_deg=_MODIS_MAX_SCAN_ANGLE_DEG,
            altitude_km=_MODIS_ALTITUDE_KM,
            method=method,
        )

    @classmethod
    def viirs(cls, band: str = "M", *, method: str = "nearest") -> BowtieCorrection:
        """VIIRS (S-NPP / JPSS) preset for aggregated SDR swaths.

        16 M-band (3200 columns) or 32 I-band (6400 columns) detectors per
        scan, ±56.28°, 824 km, with the three on-board aggregation zones
        (NOAA NESDIS 142, VIIRS SDR User's Guide). Bowtie-deleted rows are
        expected as fill and are refilled from the overlapping scan.

        Args:
            band: ``"M"`` (750 m) or ``"I"`` (375 m).
            method: ``"nearest"`` or ``"linear"``.

        Returns:
            The configured operator.

        Examples:
            >>> BowtieCorrection.viirs().detectors_per_scan
            16
            >>> BowtieCorrection.viirs("I").aggregation_zones[0]
            (3552, 3)
        """
        if band not in _VIIRS_DETECTORS:
            raise ValueError(f"band must be one of {sorted(_VIIRS_DETECTORS)}.")
        scale = 2 if band == "I" else 1
        return cls(
            detectors_per_scan=_VIIRS_DETECTORS[band],
            max_scan_angle_deg=_VIIRS_MAX_SCAN_ANGLE_DEG,
            altitude_km=_VIIRS_ALTITUDE_KM,
            aggregation_zones=tuple(
                (native * scale, factor) for native, factor in _VIIRS_M_ZONES
            ),
            method=method,
        )

    @over_frames
    def _apply(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        require_ndim(gt, (2, 3), "BowtieCorrection")
        n = self.detectors_per_scan
        height, width = gt.shape[-2:]
        if height % n:
            raise ValueError(
                f"BowtieCorrection needs whole scans: {height} rows is not a "
                f"multiple of detectors_per_scan={n}."
            )
        theta = scan_angles(width, self.max_scan_angle_deg, self.aggregation_zones)
        _, growth, _ = scan_pixel_growth(theta, self.altitude_km, self.earth_radius_km)
        k0, k1, step = bowtie_detector_positions(growth, n)

        values = np.asarray(gt)
        if self.method == "linear":
            values = as_float(values)
        valid = valid_pixels(gt)
        rows = np.arange(height)
        scan = rows // n
        m = rows % n
        n_scans = height // n
        # Primary source: the scan whose nadir rows contain the output row.
        out, ok = self._sample_scan(values, valid, scan, k0[m], n)
        # Fallback: the overlapping neighbour scan, where the primary is nodata.
        neighbour = scan + step[m]
        in_granule = (neighbour >= 0) & (neighbour < n_scans)
        alt, alt_ok = self._sample_scan(
            values, valid, np.clip(neighbour, 0, n_scans - 1), k1[m], n
        )
        use_alt = ~ok & alt_ok & in_granule[:, None]
        out = np.where(use_alt, alt, out)
        filled = ok | use_alt
        fill = carried_fill(gt, out.dtype)
        return wrap_filled(gt, out, fill_value=fill, valid=filled)

    def _sample_scan(
        self,
        values: np.ndarray,
        valid: np.ndarray,
        scan: np.ndarray,
        k: np.ndarray,
        n: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample ``values`` at fractional detector ``k`` of scan ``scan``.

        Returns the ``(..., H, W)`` samples and the ``(H, W)`` mask of
        samples inside the scan footprint (``−½ ≤ k ≤ N − ½``) whose
        contributing input pixels are all valid.
        """
        cols = np.arange(k.shape[-1])[None, :]
        base = (scan * n)[:, None]
        inside = (k >= -0.5) & (k <= n - 0.5)
        kc = np.clip(k, 0.0, n - 1.0)
        if self.method == "nearest":
            row = base + np.rint(kc).astype(int)
            return values[..., row, cols], inside & valid[row, cols]
        # Linear in k: v = v[k₀] + w·(v[k₀ + 1] − v[k₀]), w = k − k₀ (exact when
        # the two detectors agree, so a cross-track ramp maps to itself).
        lo = np.minimum(np.floor(kc).astype(int), max(n - 2, 0))
        hi = np.minimum(lo + 1, n - 1)
        weight = (kc - lo).astype(values.dtype)
        row_lo, row_hi = base + lo, base + hi
        valid_lo, valid_hi = valid[row_lo, cols], valid[row_hi, cols]
        # Zero-weight neighbours may be nodata without spoiling the sample.
        ok = inside & (valid_lo | (weight == 1)) & (valid_hi | (weight == 0))
        v_lo = np.where(valid_lo, values[..., row_lo, cols], 0)
        v_hi = np.where(valid_hi, values[..., row_hi, cols], 0)
        return v_lo + weight * (v_hi - v_lo), ok


class AntimeridianSplit(Operator):
    """Split a geographic swath at a longitude wrap.

    Longitudes are read from ``gt.attrs["lons"]`` / ``gt.attrs["lon"]`` /
    ``gt.attrs["longitude"]`` when present -- one value per column (a 1-D
    tuple) or one per pixel (nested ``(H, W)`` tuples such as
    ``lons.tolist()``; attrs cannot hold a 2-D array) -- otherwise they are
    derived from the affine grid for geographic CRSs. If no jump larger
    than ``tolerance_deg`` is found, the output is a single-item list
    containing the input unchanged.

    Each part carries its normalised longitudes (in ``[-180, 180)``) under
    ``attrs["lons"]``, in the form they were read: per column when the
    input gave one value per column (or the grid-derived longitudes are
    constant along rows), else per pixel as nested tuples.

    Geo-dependent: requires a georeferenced ``GeoArray`` input (attrs /
    geographic grid); plain arrays raise ``TypeError``.

    Args:
        crs: Geographic CRS used to interpret longitudes.
        tolerance_deg: Absolute longitude jump threshold in degrees.
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(self, *, crs: str = "EPSG:4326", tolerance_deg: float = 90.0) -> None:
        self.crs = crs
        self._parsed_crs = CRS.from_user_input(crs)
        self.tolerance_deg = tolerance_deg

    def _apply(self, gt: GeoArray) -> list[GeoArray]:
        require_geoarray(gt, "AntimeridianSplit")
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
        per_column = _per_column(gt, lons)
        left = _with_sliced_longitudes(
            gt[..., :, :split_col], lons[:, :split_col], per_column
        )
        right = _with_sliced_longitudes(
            gt[..., :, split_col:], lons[:, split_col:], per_column
        )
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
    sphere) are filled with ``gt.fill_value`` (``0`` when it is ``None``).

    Geo-dependent: requires a georeferenced ``GeoArray`` input on an
    EPSG:4326 grid; plain arrays raise ``TypeError``.

    Args:
        satellite_lon_deg: Sub-satellite longitude.
        satellite_height_m: Satellite height above the equator.
        target_height_m: Scalar height, same-grid array, or same-grid GeoArray.
        earth_radius_m: Earth radius (spherical model).
        method: ``"nearest"`` or ``"bilinear"``.
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self,
        *,
        satellite_lon_deg: float,
        satellite_height_m: float = 35_786_023.0,
        target_height_m: float | np.ndarray | GeoArray = 0.0,
        earth_radius_m: float = 6_378_137.0,
        method: str = "bilinear",
    ) -> None:
        self.satellite_lon_deg = satellite_lon_deg
        self.satellite_height_m = satellite_height_m
        self.target_height_m = target_height_m
        self.earth_radius_m = earth_radius_m
        self.method = method

    def _apply(self, gt: GeoArray) -> GeoArray:
        require_geoarray(gt, "GeostationaryParallaxCorrect")
        if self.method not in {"nearest", "bilinear"}:
            raise ValueError("method must be 'nearest' or 'bilinear'.")
        reject_geotensor(
            self.target_height_m, "GeostationaryParallaxCorrect (target_height_m)"
        )
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
        inv = ~affine_of(gt)
        src_cols, src_rows = inv * (apparent_lon, apparent_lat)
        # ``~transform`` yields continuous pixel coordinates in which pixel
        # ``(r, c)`` is centred at ``(r + 0.5, c + 0.5)``; the samplers index
        # ``arr[r]`` at coordinate ``r``, so shift to centre-indexed space.
        sampled = _sample_array(
            np.asarray(gt),
            np.asarray(src_rows) - 0.5,
            np.asarray(src_cols) - 0.5,
            method=self.method,
            fill=gt.fill_value,
        )
        if np.any(off_limb):
            sampled = _apply_invalid_fill(sampled, ~off_limb, gt.fill_value)
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
    """Assemble indexed sensor segments into one contiguous ``GeoArray``.

    Segment order is read from each input's
    ``attrs["__geotoolz_segment_meta__"]`` mapping (a nested dict, frozen
    by the carrier) with ``segment_index`` and ``n_segments`` keys. Both zero-based
    (``0..n_segments-1``) and one-based (``1..n_segments``) indexing are
    accepted. One-based indexing is selected only when an index equal to
    ``n_segments`` is present; ambiguous missing-edge cases default to
    zero-based. Missing segments are filled with ``fill_value``.

    Geo-dependent: every segment must be a georeferenced ``GeoArray``
    (attrs metadata + transform bookkeeping); plain arrays raise
    ``TypeError``.

    Args:
        direction: Stitching direction: ``"scan"`` / ``"y"`` for
            along-track, or ``"sample"`` / ``"x"`` for cross-track.
        fill_value: Value used for missing segments. ``None`` (default) means
            ``NaN``; a ``NaN`` argument is stored as ``None`` so the config
            stays strict JSON.
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self, *, direction: str = "scan", fill_value: float | None = None
    ) -> None:
        self.direction = direction
        self.fill_value = (
            None if fill_value is None or np.isnan(fill_value) else fill_value
        )

    def _apply(self, segments: list[GeoArray]) -> GeoArray:
        _require_carrier_list(segments, "SegmentStitch", "segments")
        if len(segments) == 0:
            raise ValueError("SegmentStitch requires at least one segment.")
        for segment in segments:
            require_geoarray(segment, "SegmentStitch")
        axis_num = _segment_axis(self.direction)
        metas = [_segment_meta(segment) for segment in segments]
        n_segments = metas[0][1]
        if any(n != n_segments for _, n in metas):
            raise ValueError("All segments must declare the same n_segments.")
        index_offset = 1 if max(index for index, _ in metas) == n_segments else 0
        by_index: dict[int, GeoArray] = {}
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
        fill = np.nan if self.fill_value is None else self.fill_value
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
        transform = affine_of(first) * Affine.translation(-col_before, -row_before)
        out = wrap_like(first, values, transform=transform, fill_value=segment_fill)
        return out.with_attrs(
            **{
                _SEGMENT_META: {
                    "segment_index": 0,
                    "n_segments": 1,
                    "source_n_segments": n_segments,
                }
            }
        )


class Mosaic(Operator):
    r"""Mosaic multiple `GeoArray`s onto a single grid.

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
    Pixels no input covers get the first raster's ``fill_value``;
    when that is ``None`` they get NaN (float dtypes) or ``0`` (integer
    dtypes), and ``"first"`` also takes this stacked path because
    ``spatial_mosaic`` cannot pad fill-less inputs.

    Geo-dependent: every input must be a georeferenced ``GeoArray``
    (transform + CRS drive the mosaic frame); plain arrays raise
    ``TypeError``.

    Args:
        method: ``"first"`` (default), ``"mean"`` / ``"average"``,
            ``"median"``, ``"max"``, ``"min"``.
        resampling: Resampling mode used for reprojection onto the
            mosaic frame.

    Examples:
        >>> import geoops as gz
        >>> # Median-composite a stack of overlapping S2 scenes.
        >>> composite = gz.geom.Mosaic(method="median")(s2_scene_list)
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

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

    def _apply(self, gts: list[GeoArray]) -> GeoArray:
        mosaic = optional_name(
            "georeader",
            "mosaic",
            "sensors",
            feature="geoops.geom.Mosaic",
        )
        read = optional_name(
            "georeader", "read", "sensors", feature="geoops.geom.Mosaic"
        )
        _require_carrier_list(gts, "Mosaic", "rasters")
        for gt in gts:
            require_geoarray(gt, "Mosaic")
        if self.method == "first" and all(gt.fill_value is not None for gt in gts):
            return adopt_attrs(
                gts[0],
                mosaic.spatial_mosaic(
                    [to_engine(gt) for gt in gts],
                    resampling=resolve_resampling(self.resampling),
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
                    to_engine(_nan_masked(gt)),
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
        fill = first.fill_value
        if fill is None:
            fill = np.nan if np.issubdtype(first.dtype, np.floating) else 0
        values = np.where(np.isnan(values), fill, values)
        return wrap_like(
            first,
            values.astype(first.dtype, copy=False),
            transform=transform,
            fill_value=fill,
        )


class Georeference(Operator):
    """Georeference swath data using a GLT (geolocation lookup table) `GeoArray`.

    Thin wrapper around :func:`georeader.griddata.georreference` (note
    the upstream spelling), called through the engine boundary. The GLT (Geolocation Lookup Table) maps
    output-grid pixels to source-sensor pixels for fast, exact
    orthorectification with no resampling artifacts.

    The input is sensor-space swath data, so plain ``np.ndarray`` input
    is accepted alongside ``GeoArray``; the output is always a
    georeferenced ``GeoArray`` whose geometry comes from the GLT.

    Args:
        glt: GLT `GeoArray` of shape ``(2, H_out, W_out)``. ``glt[0]``
            holds source columns; ``glt[1]`` holds source rows.
        valid_glt: Optional boolean mask of valid GLT pixels.

    Note:
        Carries a concrete `GeoArray` and (optionally) a numpy array
        reference, so flagged ``forbid_in_yaml = True``.

    Examples:
        >>> import geoops as gz
        >>> # Orthorectify a hyperspectral scene predicted in sensor space.
        >>> ortho = gz.geom.Georeference(glt=glt)(prediction_swath)
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(self, *, glt: GeoArray, valid_glt: np.ndarray | None = None) -> None:
        self.glt = glt
        self.valid_glt = valid_glt

    def _apply(self, gt: GeoArray | np.ndarray) -> GeoArray:
        griddata = optional_name(
            "georeader",
            "griddata",
            "sensors",
            feature="geoops.geom.Georeference",
        )
        reject_geotensor(gt, "Georeference")
        reject_georeader(self.glt, "Georeference (glt)")
        reject_geotensor(self.valid_glt, "Georeference (valid_glt)")
        fill = gt.fill_value if isinstance(gt, GeoArray) else None
        return from_engine(
            griddata.georreference(
                to_engine(self.glt),
                np.asarray(gt),
                valid_glt=self.valid_glt,
                fill_value_default=fill,
            )
        )

    def get_config(self) -> dict[str, Any]:
        return {"glt": repr(self.glt), "valid_glt": None}


class Rasterize(Operator):
    """Rasterize geometries onto the input `GeoArray`'s grid.

    Burns ``geometries`` into a fresh array aligned with the input's
    CRS and transform. Accepts either:

    - a ``list`` of shapely geometries (each gets value ``1``); or
    - a :class:`geopandas.GeoDataFrame` (the ``column`` argument
      selects which attribute supplies the burn-in value; without one
      every geometry burns ``1``, as for a list).

    Delegates to :func:`georeader.rasterize.rasterize_geopandas_like`
    or :func:`georeader.rasterize.rasterize_geometry_like`.

    Geo-dependent: requires a georeferenced ``GeoArray`` input (its
    transform + CRS define the burn grid); plain arrays raise
    ``TypeError``.

    The output is one ``(H, W)`` raster whatever the input's rank: every
    band and every frame of a ``(T, C, H, W)`` stack share the grid, so
    one burn serves them all. Its ``fill_value`` is ``fill_value``
    in the output dtype (``0`` for the default ``uint8`` burn), and
    pixels that are nodata in the input (in any band or frame) hold it.

    Note:
        Carries Python-level geometry objects (shapely / geopandas),
        so flagged ``forbid_in_yaml = True``.

    Args:
        geometries: Shapely geometries or a `GeoDataFrame`.
        column: For `GeoDataFrame` inputs, the column with burn-in
            values. ``None`` means "burn ``1``".
        all_touched: Whether to mark every pixel the geometry touches
            (vs. only those whose centre is inside).
        fill_value: Background value. Default ``0``. Without a
            ``column`` the output is ``uint8`` when ``fill_value`` is an
            integer in ``[0, 255]`` and ``float32`` otherwise (e.g.
            ``np.nan``).

    Examples:
        >>> import geoops as gz
        >>> from shapely.geometry import box
        >>> aoi_mask = gz.geom.Rasterize(geometries=[box(*aoi_bounds)])(scene)
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self,
        *,
        geometries: list[BaseGeometry] | gpd.GeoDataFrame,
        column: str | None = None,
        all_touched: bool = False,
        fill_value: float = 0.0,
    ) -> None:
        self.geometries = geometries
        self.column = column
        self.all_touched = all_touched
        self.fill_value = fill_value

    def _apply(self, gt: GeoArray) -> GeoArray:
        require_geoarray(gt, "Rasterize")
        return _burn_onto(
            self.geometries,
            gt,
            column=self.column,
            fill=self.fill_value,
            all_touched=self.all_touched,
            nodata_from=(gt,),
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "geometries": f"<{type(self.geometries).__name__} n={len(self.geometries)}>",
            "column": self.column,
            "all_touched": self.all_touched,
            "fill_value": self.fill_value,
        }


class RasterizeLike(Operator):
    """Rasterize geometries onto a stored reference grid.

    Same as :class:`Rasterize` but the reference grid is pinned at
    construction (``like``) rather than supplied per call. The
    operator takes *no* positional input (it is a producer) -- useful as
    the head of a `Sequential` pipeline that consumes the burned mask --
    or a `GeoArray` on ``like``'s grid (``ValueError`` otherwise), whose
    nodata pixels then hold the fill as in :class:`Rasterize` (``like``
    only supplies the grid). The output is one ``(H, W)`` raster whatever
    the input's rank.

    Note:
        Flagged ``forbid_in_yaml = True``.

    Args:
        like: Reference `GeoArray` whose CRS + transform + shape define
            the output grid.
        geometries: Shapely geometries or a `GeoDataFrame`.
        column: Attribute column for the burn-in value, or ``None``.
        all_touched: As in :class:`Rasterize`.
        fill_value: Background value.

    Examples:
        >>> import geoops as gz
        >>> burner = gz.geom.RasterizeLike(
        ...     like=ref_scene, geometries=polygons_gdf, column="class_id"
        ... )
        >>> labels = burner()
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self,
        *,
        like: GeoArray,
        geometries: gpd.GeoDataFrame,
        column: str | None = None,
        all_touched: bool = False,
        fill_value: float = 0.0,
    ) -> None:
        self.like = like
        self.geometries = geometries
        self.column = column
        self.all_touched = all_touched
        self.fill_value = fill_value

    def _apply(self, gt: GeoArray | None = None) -> GeoArray:
        reject_georeader(self.like, "RasterizeLike (like)")
        if gt is not None:
            reject_geotensor(gt, "RasterizeLike")
        if gt is not None and not grid_matches(gt, self.like):
            raise ValueError(
                "RasterizeLike: the input is not on `like`'s grid (spatial "
                "shape, transform, CRS); call it with no input, or use "
                "Rasterize to burn onto the input's own grid."
            )
        return _burn_onto(
            self.geometries,
            self.like,
            column=self.column,
            fill=self.fill_value,
            all_touched=self.all_touched,
            nodata_from=() if gt is None else (gt,),
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "like": repr(self.like),
            "geometries": f"<{type(self.geometries).__name__} n={len(self.geometries)}>",
            "column": self.column,
            "all_touched": self.all_touched,
            "fill_value": self.fill_value,
        }


class Vectorize(Operator):
    """Vectorize non-zero regions of a mask `GeoArray` into polygons.

    Wraps :func:`georeader.vectorize.get_polygons`. The carrier's
    transform is used to express polygons in geographic coordinates.
    Polygons with area below ``min_area_px`` (in square pixels of the
    mask) are dropped.

    Geo-dependent: requires a georeferenced ``GeoArray`` input (the
    transform maps pixels to polygon coordinates); plain arrays raise
    ``TypeError``. The mask must be single-band, ``(H, W)`` or
    ``(1, H, W)``; other shapes raise ``ValueError``.

    Args:
        min_area_px: Minimum polygon area in square *pixels*. Default
            ``25.5`` (~5×5 px), matching the georeader default.
        simplify_tolerance: Optional post-hoc :meth:`shapely.simplify`
            tolerance (in CRS units). ``None`` (default) leaves the
            staircase pixel boundaries intact.

    Examples:
        >>> import geoops as gz
        >>> polys = gz.geom.Vectorize(min_area_px=100.0)(water_mask)
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    _terminal: ClassVar[bool] = True

    def __init__(
        self,
        *,
        min_area_px: float = 25.5,
        simplify_tolerance: float | None = None,
    ) -> None:
        self.min_area_px = min_area_px
        self.simplify_tolerance = simplify_tolerance

    def _apply(self, gt: GeoArray) -> list[BaseGeometry]:
        vectorize = optional_name(
            "georeader",
            "vectorize",
            "sensors",
            feature="geoops.geom.Vectorize",
        )
        require_geoarray(gt, "Vectorize")
        # ``get_polygons`` asserts a 2-D mask; squeeze the canonical
        # ``(1, H, W)`` band axis and pass the transform explicitly.
        mask = single_band(np.asarray(gt), name="Vectorize")
        polygons = vectorize.get_polygons(
            mask, min_area=self.min_area_px, tolerance=0.0, transform=affine_of(gt)
        )
        if self.simplify_tolerance is None:
            return polygons
        return [polygon.simplify(self.simplify_tolerance) for polygon in polygons]


def _rasterize_like(
    geometries: list[BaseGeometry] | gpd.GeoDataFrame,
    like: GeoArray,
    *,
    column: str | None,
    fill: float,
    all_touched: bool,
) -> GeoArray:
    """Dispatch onto the right `georeader.rasterize` API for the inputs.

    Picks :func:`rasterize_geopandas_like` for `GeoDataFrame` inputs
    (synthesising a unit-value column when none is provided) and falls
    back to :func:`rasterize_geometry_like` over the union of shapely
    geometries otherwise.
    """
    # A unit burn is ``uint8``, which cannot hold a NaN / negative /
    # fractional / >255 background; widen to ``float32`` for those.
    rasterize = optional_name(
        "georeader",
        "rasterize",
        "sensors",
        feature="geoops.geom",
    )
    unary_union = optional_name(
        "shapely.ops",
        "unary_union",
        "vector",
        feature="geoops.geom",
    )
    fill_value = float(fill)
    fits_uint8 = fill_value.is_integer() and 0 <= fill_value <= 255
    unit_dtype = np.uint8 if fits_uint8 else np.float32
    if hasattr(geometries, "geometry"):
        dataframe = geometries
        column_name = column or "__geotoolz_value__"
        if column is None:
            dataframe = geometries.assign(
                **{column_name: np.ones(len(geometries), dtype=unit_dtype)}
            )
        return from_engine(
            rasterize.rasterize_geopandas_like(
                dataframe,
                to_engine(like),
                column=column_name,
                fill=fill,
                all_touched=all_touched,
            )
        )
    return from_engine(
        rasterize.rasterize_geometry_like(
            unary_union(geometries),
            to_engine(like),
            value=1,
            dtype=unit_dtype,
            fill=fill,
            all_touched=all_touched,
        )
    )


def _burn_onto(
    geometries: list[BaseGeometry] | gpd.GeoDataFrame,
    like: GeoArray,
    *,
    column: str | None,
    fill: float,
    all_touched: bool,
    nodata_from: Sequence[Any],
) -> GeoArray:
    """Burn ``geometries`` onto ``like``'s grid with a dtype-true background fill.

    The background ``fill`` doubles as the output's nodata, declared in the
    burn raster's dtype (``0``, not ``0.0``, for a ``uint8`` raster), and
    every pixel that is nodata in any of ``nodata_from`` (in any band or
    frame) holds it too: a burn over a scene says nothing where the scene
    has no data.
    """
    burned = _rasterize_like(
        geometries, like, column=column, fill=fill, all_touched=all_touched
    )
    values = np.asarray(burned)
    fill_value: float | int = float(fill)
    if not np.issubdtype(values.dtype, np.inexact) and float(fill).is_integer():
        fill_value = int(fill)
    valid = np.ones(values.shape[-2:], dtype=bool)
    for carrier in nodata_from:
        valid &= valid_pixels(carrier)
    return wrap_filled(burned, values, fill_value=fill_value, valid=valid)


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

    ``GeoArray.fill_value`` can be ``None``; in that case this
    internal resampler uses ``0`` for invalid samples.
    """
    if valid.all():
        return sampled
    fill_value = 0 if fill is None else fill
    # ``valid`` is a 2D (H, W) mask; ``sampled`` may have leading band
    # dimensions. ``np.where`` broadcasts the mask across them, while
    # boolean indexing with a 2D mask collapses those trailing axes.
    return np.where(valid, sampled, np.asarray(fill_value, dtype=sampled.dtype))


def _longitude_grid(gt: GeoArray, expected_crs: str | CRS) -> np.ndarray | None:
    attrs = gt.attrs
    spatial_shape = gt.shape[-2:]
    for key in _LON_KEYS:
        if key in attrs:
            lons = np.asarray(attrs[key], dtype=np.float64)
            if lons.ndim == 1:
                if lons.shape[0] != spatial_shape[1]:
                    raise ValueError(
                        f"attrs['{key}'] has length {lons.shape[0]} but the "
                        f"GeoArray has width {spatial_shape[1]}."
                    )
                return np.broadcast_to(lons[None, :], spatial_shape)
            if lons.shape != spatial_shape:
                raise ValueError(
                    f"attrs['{key}'] has shape {lons.shape} but the GeoArray "
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


def _lonlat_centres(gt: GeoArray) -> tuple[np.ndarray, np.ndarray] | None:
    if CRS.from_user_input(gt.crs) != CRS.from_epsg(4326):
        return None
    height, width = gt.shape[-2:]
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    return pixel_xy(gt, rows, cols)


def _normalise_longitudes(lons: np.ndarray) -> np.ndarray:
    return ((lons + 180.0) % 360.0) - 180.0


def _per_column(gt: GeoArray, lons: np.ndarray) -> bool:
    """Whether a part's ``attrs["lons"]`` is written per column (1-D).

    Per column when the input's longitudes were given that way, or when
    they come from the grid and are constant along rows (a north-up
    geographic grid); otherwise per pixel.
    """
    for key in _LON_KEYS:
        if key in gt.attrs:
            return np.ndim(gt.attrs[key]) == 1
    return bool(np.all(lons == lons[:1]))


def _with_sliced_longitudes(
    part: GeoArray, lons: np.ndarray, per_column: bool
) -> GeoArray:
    """``part`` with its normalised longitudes under ``attrs["lons"]``.

    attrs hold hashable values only, so per-column longitudes are a 1-D
    tuple and per-pixel ones nested ``(H, W)`` tuples.
    """
    normalised = _normalise_longitudes(lons)
    return part.with_attrs(
        lons=tuple(normalised[0].tolist()) if per_column else normalised.tolist()
    )


def _height_array(
    target_height_m: float | np.ndarray | GeoArray, shape: tuple[int, int]
) -> np.ndarray:
    if isinstance(target_height_m, GeoArray):
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


def _segment_axis(direction: str) -> int:
    if direction in {"scan", "y"}:
        return -2
    if direction in {"sample", "x"}:
        return -1
    raise ValueError("direction must be 'scan'/'y' or 'sample'/'x'.")


def _coerce_segment_fill(fill: float | int, first: GeoArray) -> float | int:
    """Return a fill value that is representable in ``first.dtype``.

    Integer-typed sensor segments cannot hold ``NaN``; in that case fall back
    to the segment's own ``fill_value`` (when finite) or ``0``.
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
    existing = first.fill_value
    if existing is not None:
        try:
            if not bool(np.isnan(existing)):
                return existing
        except TypeError:
            return existing
    return 0


def _segment_meta(segment: GeoArray) -> tuple[int, int]:
    meta = segment.attrs.get(_SEGMENT_META)
    if not isinstance(meta, Mapping):
        raise ValueError("Each segment must define attrs['__geotoolz_segment_meta__'].")
    try:
        return int(meta["segment_index"]), int(meta["n_segments"])
    except KeyError as exc:
        raise ValueError(
            "Segment metadata requires segment_index and n_segments."
        ) from exc


def _validate_segment_compatible(
    first: GeoArray,
    segment: GeoArray,
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
    segment_transform, first_transform = affine_of(segment), affine_of(first)
    if not np.isclose(segment_transform.a, first_transform.a) or not np.isclose(
        segment_transform.e,
        first_transform.e,
    ):
        raise ValueError("All segments must share pixel size.")


class Gradient(Operator):
    r"""Diagnostic gradient of a gridded field in map axes, NaN- and mask-aware (OP-O7).

    **Masks.** ``nan_policy`` decides what a masked cell costs:

    - ``"propagate"`` (default): the plain stencil; a masked cell (non-finite,
      the carrier's fill, or ``True`` in ``mask``) makes every derivative
      whose stencil reads it ``NaN``, a halo of about ``accuracy`` cells;
    - ``"adaptive"``: per point, the widest stencil lying wholly on valid
      data (central, then forward, then backward, at decreasing accuracy),
      so valid cells next to land keep a one-order-lower estimate and only
      masked cells are ``NaN``.

    The arithmetic is the public L1 primitives, so the same maths is
    available without a carrier: :func:`geoops.geom.gradient` /
    :func:`geoops.geom.adaptive_gradient` over :func:`geoops.geom.difference`
    / :func:`geoops.geom.adaptive_difference`, whose stencils come from
    :func:`geoops.geom.difference_offsets` and :func:`geoops.geom.difference_coeffs`.

    **Geometry**, derived from the carrier. The row and column derivatives
    map to the CRS's ``x`` and ``y`` through the transform's Jacobian
    (:math:`x = a\,j + b\,i + c,\; y = d\,j + e\,i + f`),

    .. math::

        \begin{pmatrix} \partial_x f \\ \partial_y f \end{pmatrix}
        = \frac{1}{ae - bd}
          \begin{pmatrix} e & -d \\ -b & a \end{pmatrix}
          \begin{pmatrix} \partial_j f \\ \partial_i f \end{pmatrix},

    exact for a linear field on north-up, flipped and rotated grids. On a
    geographic CRS degrees become metres on the CRS's ellipsoid at each
    pixel's latitude :math:`\varphi`:
    :math:`\partial_{\mathrm{north}} f = \partial_\varphi f / (M(\varphi)\,\pi/180)`,
    :math:`\partial_{\mathrm{east}} f = \partial_\lambda f / (N(\varphi)\cos\varphi\,\pi/180)`,
    with :math:`M`, :math:`N` the meridional and prime-vertical radii.

    The result gains a ``component`` dim just before ``y x`` holding
    ``(∂f/∂y, ∂f/∂x)``, per CRS unit (per metre on a geographic CRS), and
    declares ``fill_value=NaN``. A plain array is differentiated in pixel
    units, rows then columns. A diagnostic: model-grade calculus on
    staggered grids stays in finitevolX. A geometry operator, so a
    ``DataArray`` is refused: lift it with ``xrops.lift``, or call the L1
    primitives with coordinate spacings.

    Args:
        accuracy: Accuracy order of the one-sided stencils (the centred one
            is one order higher); ``1`` is ``numpy.gradient``'s scheme.
        nan_policy: ``"propagate"`` or ``"adaptive"`` (above).
        periodic_x: The columns wrap, for a global longitude grid.

    Examples:
        >>> import geoops as gz
        >>> grad = gz.Gradient()(ssh_map)                 # (component, y, x)
        >>> grad = gz.Gradient(nan_policy="adaptive")(ssh_map, land)  # keeps the coast
        >>> step = jax.jit(gz.Gradient())                 # one trace across transforms
        >>> d_y, d_x = gz.geom.adaptive_gradient(ssh, dy, dx, mask=land)  # arrays only
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy", "jax")

    def __init__(
        self,
        *,
        accuracy: int = 1,
        nan_policy: str = "propagate",
        periodic_x: bool = False,
    ) -> None:
        if nan_policy not in ("propagate", "adaptive"):
            raise ValueError(
                f"nan_policy must be 'propagate' or 'adaptive'; got {nan_policy!r}"
            )
        if accuracy < 1:
            raise ValueError(f"accuracy must be at least 1; got {accuracy}")
        self.accuracy = accuracy
        self.nan_policy = nan_policy
        self.periodic_x = periodic_x

    def _apply(self, gt: Any, mask: Any = None) -> Any:
        reject_geotensor(gt, "Gradient")
        if type(gt).__module__.startswith("xarray."):
            raise TypeError(
                "Gradient is a geometry operator and takes a GeoArray: lift it "
                "over a DataArray cube with xrops.lift(Gradient(), dim=...), or "
                "call geoops.geom.gradient / adaptive_gradient with coordinate "
                "spacings."
            )
        if mask is not None and isinstance(gt, GeoArray) and isinstance(mask, GeoArray):
            # geoarray's own grid check: exact, and deferred to run time under jit
            try:
                get_adapter(gt).check_compatible(gt, mask)
            except GridMismatchError as exc:
                raise GridMismatchError(
                    f"Gradient: the mask is not on the field's pixel grid ({exc})"
                ) from exc
        if not isinstance(gt, GeoArray):
            # a plain array (NumPy or JAX): pixel units, on its own backend
            field = as_float(gt if hasattr(gt, "dtype") else np.asarray(gt))
            xp = array_namespace(field)
            grad = self._pixel_gradient(field, xp, _mask_data(mask, xp))
            return xp.moveaxis(grad, 0, -3)
        if gt.batch_dims:
            raise ValueError(
                f"Gradient: the carrier is batched over {gt.batch_dims} (one "
                "transform per item); map it per item with "
                f"geoarray.map_over(x, Gradient(...), {gt.batch_dims[0]!r})."
            )
        xp = gt.xp
        data = as_float(gt.data)
        # geoops' pixel nodata rule (any invalid band), on the carrier's backend
        invalid = invalid_pixel_mask(gt, mask=_aligned_mask(mask, gt, xp))
        mask_data = invalid
        field = xp.where(invalid, xp.nan, data)
        d_row, d_col = self._pixel_gradient(field, xp, mask_data)
        d_y, d_x = _map_axes(gt, d_row, d_col)
        # an invalid input pixel has no derivative, whatever its neighbours hold
        grad = xp.where(invalid[..., None, :, :], xp.nan, xp.stack([d_y, d_x], axis=-3))
        dims = (*gt.dims[:-2], "component", *gt.dims[-2:])
        return rewrap_dims(gt, grad, dims, fill_value=np.nan)

    def _pixel_gradient(self, field: Any, xp: Any, mask: Any) -> Any:
        """``(∂f/∂i, ∂f/∂j)`` per pixel step, through the L1 primitives."""
        if self.nan_policy == "adaptive":
            return adaptive_gradient(
                field, mask=mask, accuracy=self.accuracy, periodic_x=self.periodic_x
            )
        if mask is not None:
            field = xp.where(mask, xp.nan, field)
        out = gradient(field, accuracy=self.accuracy, periodic_x=self.periodic_x)
        if mask is not None:
            out = xp.where(mask, xp.nan, out)
        return out


def _aligned_mask(mask: Any, gt: GeoArray, xp: Any) -> Any:
    """``mask`` as booleans aligned to ``gt``'s dims: by name for a carrier.

    A `GeoArray` mask's dims must be among the field's; each missing dim
    becomes a size-1 axis, so a ``(time, y, x)`` mask drops whole pixels of
    a ``(time, band, y, x)`` field frame by frame. A plain array broadcasts
    from the right, as NumPy does (an ``(H, W)`` mask on any field).
    """
    data = _mask_data(mask, xp)
    if data is None or not isinstance(mask, GeoArray):
        return data
    extra = [d for d in mask.dims if d not in gt.dims]
    if extra:
        raise ValueError(
            f"Gradient: the mask has dims {extra} the field lacks; its dims are "
            f"{mask.dims}, the field's {gt.dims}"
        )
    present = [d for d in gt.dims if d in mask.dims]
    data = xp.permute_dims(data, tuple(mask.dims.index(d) for d in present))
    for axis, dim in enumerate(gt.dims):
        if dim not in mask.dims:
            data = xp.expand_dims(data, axis=axis)
    return data


def _mask_data(mask: Any, xp: Any) -> Any:
    """The boolean array of a mask carrier (``True`` drops), or ``None``."""
    if mask is None:
        return None
    reject_geotensor(mask, "Gradient")
    data = mask.data if isinstance(mask, GeoArray) else mask
    data = data if hasattr(data, "dtype") else np.asarray(data)
    return data if data.dtype == bool else xp.astype(data, xp.bool)


def _map_axes(gt: GeoArray, d_row: Any, d_col: Any) -> tuple[Any, Any]:
    """``(∂f/∂y, ∂f/∂x)`` in the CRS's axes (metres on a geographic CRS)."""
    a, b, _, d, e, f0 = (gt.transform[k] for k in range(6))
    xp = gt.xp

    def times(coeff: Any, value: Any) -> Any:
        # a zero Jacobian entry must not pull in the other axis's NaN (0 * nan)
        return xp.where(coeff == 0, 0.0, coeff * value)

    det = a * e - b * d
    d_x = (times(e, d_col) - times(d, d_row)) / det
    d_y = (times(a, d_row) - times(b, d_col)) / det
    crs = gt.crs
    if crs is None or not CRS.from_user_input(crs).is_geographic:
        return d_y, d_x
    height, width = gt.shape[-2:]
    rows = xp.arange(height, dtype=xp.float64)[:, None] + 0.5
    cols = xp.arange(width, dtype=xp.float64)[None, :] + 0.5
    lat = xp.deg2rad(d * cols + e * rows + f0)
    ellipsoid = CRS.from_user_input(crs).ellipsoid
    semi_major = ellipsoid.semi_major_metre
    e2 = 1.0 - (ellipsoid.semi_minor_metre / semi_major) ** 2
    w = xp.sqrt(1.0 - e2 * xp.sin(lat) ** 2)
    meridional = semi_major * (1.0 - e2) / w**3
    prime_vertical = semi_major / w
    per_degree = np.pi / 180.0
    return d_y / (meridional * per_degree), d_x / (
        prime_vertical * xp.cos(lat) * per_degree
    )
