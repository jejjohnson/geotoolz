"""Carrier-aware compositing operators for co-registered GeoTensors.

All composites are metadata-independent per-pixel reductions, so they
accept sequences of plain ``np.ndarray`` frames as well as GeoTensors.
The frame-sequence composites (:class:`MedianComposite`,
:class:`MaxNDVIComposite`) also accept a single ``(T, C, H, W)`` time
stack -- georeader's ``("time", "band", "y", "x")`` GeoTensor -- which is
reduced over its time axis and rewrapped like ``stack.isel({"time": 0})``.
Grid checks (transform / CRS equality) apply only to frames that carry
georeferencing; plain arrays fall back to shape-equality checks. The
output carrier follows the first frame.

Nodata is judged *per frame* with :func:`geotoolz._src.valid.valid_pixels`
(non-finite values, or the frame's ``fill_value_default``, in any band):
an invalid frame never contributes to a reduction, a score or a
selection at that pixel, and pixels invalid in every frame come out
holding the output's fill value. Composites carry the frames' values, so
they keep the first frame's ``fill_value_default`` -- switching to
``NaN`` when integer frames are averaged into float (see
:func:`geotoolz._src.valid.carried_fill`) and using ``NaN`` when the first
frame has none. Auxiliary outputs follow their own meaning: contributor
counts declare ``fill_value_default=0`` (a pixel no frame covers has
count ``0``), float scores use ``NaN``, and the frame-index map of
:class:`MaxNDVIComposite` uses ``-1`` (``0`` is a valid frame index).
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from jaxtyping import Bool, Float, Int, Num, Shaped
from pipekit import Operator

from geotoolz._src.bands import BandRef, resolve_band
from geotoolz._src.geo import grid_matches
from geotoolz._src.valid import (
    carried_fill,
    invalid_values,
    restore_fill,
    valid_pixels,
)
from geotoolz._src.wrap import wrap_like


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor

NanPolicy = Literal["ignore", "propagate"]


def _validate_nan_policy(nan_policy: str) -> NanPolicy:
    if nan_policy not in {"ignore", "propagate"}:
        raise ValueError("nan_policy must be 'ignore' or 'propagate'.")
    return nan_policy  # type: ignore[return-value]


def _as_frames(
    frames: Sequence[GeoTensor | np.ndarray] | GeoTensor | np.ndarray, name: str
) -> Sequence[GeoTensor | np.ndarray]:
    """Accept a sequence of frames or a single ``(T, C, H, W)`` stack.

    A GeoTensor stack (dims ``time, band, y, x``) is split into its
    ``gt.isel({"time": t})`` frames, which keep its transform, CRS, fill
    value and attrs, so the composite is rewrapped like frame 0. A plain
    ndarray stack (``(T, C, H, W)``, or ``(T, H, W)`` for single-band
    frames) is split along axis 0. A 2-D / 3-D GeoTensor is not a time
    stack (its leading axis is bands) and is rejected.
    """
    if not isinstance(frames, np.ndarray):
        return frames
    isel = getattr(frames, "isel", None)
    if isel is not None and frames.ndim == 4:
        return [isel({"time": t}) for t in range(frames.shape[0])]
    if isel is None and frames.ndim in (3, 4):
        return list(frames)
    raise ValueError(
        f"{name} takes a sequence of co-registered frames or a (T, C, H, W) "
        f"time stack; got a {frames.ndim}-D "
        f"{'GeoTensor' if isel is not None else 'array'} of shape {frames.shape}"
    )


def _require_frames(
    frames: Sequence[GeoTensor | np.ndarray],
) -> GeoTensor | np.ndarray:
    if not frames:
        raise ValueError("At least one GeoTensor is required for compositing.")
    base = frames[0]
    for idx, frame in enumerate(frames[1:], start=1):
        # Full shape and exact affine: a per-pixel reduction over misaligned
        # grids (or mismatched band counts) silently produces garbage.
        if not grid_matches(base, frame, spatial_only=False):
            raise ValueError(
                "All input GeoTensors must share shape, transform, and CRS; "
                f"frame 0 has shape {base.shape}, frame {idx} has shape {frame.shape}."
            )
    return base


def _stack_frames(
    frames: Sequence[GeoTensor | np.ndarray],
) -> tuple[GeoTensor | np.ndarray, np.ndarray]:
    base = _require_frames(frames)
    return base, np.stack([np.asarray(frame) for frame in frames], axis=0)


def _frame_validity(
    frames: Sequence[GeoTensor | np.ndarray],
) -> Bool[np.ndarray, "t h w"]:
    """Per-frame ``(T, H, W)`` validity; each frame judged by its own fill.

    A ``(T, C, H, W)`` / ``(T, H, W)`` array carrier passed whole is judged
    against its own ``fill_value_default`` (slicing a frame out of it could
    drop the fill); a sequence is judged frame by frame.
    """
    if isinstance(frames, np.ndarray):
        invalid = invalid_values(frames)
        if invalid.ndim > 3:
            invalid = invalid.any(axis=tuple(range(1, invalid.ndim - 2)))
        return ~invalid
    return np.stack([valid_pixels(frame) for frame in frames], axis=0)


def _broadcast_frame_valid(
    valid: Bool[np.ndarray, "t h w"], shape: tuple[int, ...]
) -> Bool[np.ndarray, "t *dims h w"]:
    """Broadcast ``(T, H, W)`` frame validity against a ``(T, ..., H, W)`` stack."""
    extra = len(shape) - valid.ndim
    return np.broadcast_to(
        valid.reshape((valid.shape[0], *([1] * extra), *valid.shape[1:])), shape
    )


def _mask_frames(
    stack: Num[np.ndarray, "t *dims h w"], valid: Bool[np.ndarray, "t h w"]
) -> Float[np.ndarray, "t *dims h w"]:
    """Float copy of ``stack`` with every invalid frame-pixel set to NaN.

    Float stacks keep their dtype; integer stacks become ``float64`` --
    the dtype ``np.median`` / integer division already produced for them.
    """
    dtype = stack.dtype if np.issubdtype(stack.dtype, np.inexact) else np.float64
    out = stack.astype(dtype, copy=True)
    out[~_broadcast_frame_valid(valid, out.shape)] = np.nan
    return out


def _take_by_spatial_index(
    stack: Shaped[np.ndarray, "t *dims h w"], index: Int[np.ndarray, "h w"]
) -> Shaped[np.ndarray, "*dims h w"]:
    """Select one frame per pixel from ``(T, ..., H, W)`` stack data."""
    indexer = np.broadcast_to(index, stack.shape[1:]).reshape((1, *stack.shape[1:]))
    return np.take_along_axis(stack, indexer, axis=0)[0]


def _mask_array(
    mask: Any, target_shape: tuple[int, ...]
) -> Bool[np.ndarray, "*dims h w"]:
    mask_arr = np.asarray(mask, dtype=bool)
    spatial_shape = target_shape[-2:]
    if mask_arr.shape == spatial_shape:
        return np.broadcast_to(mask_arr, target_shape)
    if mask_arr.shape == (1, *spatial_shape):
        # For 2-D targets, a (1, H, W) mask is spatially equivalent to (H, W);
        # squeeze before broadcasting so we don't try to add a leading axis to
        # a 2-D target.
        squeezed = mask_arr[0]
        return np.broadcast_to(squeezed, target_shape)
    if mask_arr.shape == target_shape:
        return mask_arr
    raise ValueError(
        "Cloud masks must have spatial shape (H, W), (1, H, W), or match the "
        "GeoTensor shape; got "
        f"{mask_arr.shape}, expected compatible with {target_shape}."
    )


def _require_pairs(
    pairs: Sequence[tuple[GeoTensor | np.ndarray, Any]],
) -> tuple[GeoTensor | np.ndarray, np.ndarray, np.ndarray]:
    if not pairs:
        raise ValueError("At least one (GeoTensor, mask) pair is required.")
    frames = [scene for scene, _ in pairs]
    base, stack = _stack_frames(frames)
    masks = np.stack([_mask_array(mask, base.shape) for _, mask in pairs], axis=0)
    return base, stack, masks


def _metadata_value(
    metadata: Mapping[str, Any], *names: str, default: Any = None
) -> Any:
    for name in names:
        if name in metadata:
            return metadata[name]
    return default


def _score_array(
    value: Any, spatial_shape: tuple[int, int]
) -> Float[np.ndarray, "h w"]:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape == ():
        return np.full(spatial_shape, float(arr), dtype=np.float32)
    if arr.shape == spatial_shape:
        return arr
    if arr.shape == (1, *spatial_shape):
        return arr[0]
    raise ValueError(
        f"Score metadata must be scalar, (H, W), or (1, H, W); got {arr.shape}."
    )


_DAYS_PER_YEAR = 365


def _doy_distance(
    doy: Float[np.ndarray, "*dims"], target_doy: float
) -> Float[np.ndarray, "*dims"]:
    """Circular day-of-year distance to ``target_doy``, in days.

        Δ = |doy − target_doy| mod 365,   d = min(Δ, 365 − Δ)

    so DOY 360 and target DOY 5 are 10 days apart, not 355. A 365-day
    year is assumed (DOY 366 coincides with DOY 1).
    """
    delta = np.mod(
        np.abs(np.asarray(doy, dtype=np.float64) - target_doy), _DAYS_PER_YEAR
    )
    return np.minimum(delta, _DAYS_PER_YEAR - delta)


def _doy_score(
    doy: Float[np.ndarray, "*dims"], target_doy: float, sigma: float
) -> Float[np.ndarray, "*dims"]:
    """Gaussian day-of-year score (Griffiths et al., 2013), peak-normalised.

        S_doy = exp(−½ · (d / σ_doy)²),   d = circular DOY distance

    Griffiths et al. score DOY with a Gaussian centred on the target DOY;
    the ``1/(σ√(2π))`` density factor is dropped so ``S_doy ∈ (0, 1]``
    shares a scale with the other scores (``S_doy = e^(−½) ≈ 0.61`` at
    ``d = σ_doy``).
    """
    d = _doy_distance(doy, target_doy)
    return np.exp(-0.5 * (d / sigma) ** 2)


def _view_angle_score(
    view_angle: Float[np.ndarray, "*dims"], sigma: float
) -> Float[np.ndarray, "*dims"]:
    """Gaussian view-zenith-angle score, best at nadir.

        S_view = exp(−½ · (θ / σ_θ)²)

    ``θ`` (sign ignored) and ``σ_θ`` in degrees.
    """
    theta = np.asarray(view_angle, dtype=np.float64)
    return np.exp(-0.5 * (theta / sigma) ** 2)


def _cloud_distance_score(
    distance: Float[np.ndarray, "*dims"], *, d_req: float, d_min: float, slope: float
) -> Float[np.ndarray, "*dims"]:
    """Sigmoid distance-to-cloud score (Griffiths et al., 2013, eq. 2).

        S_cloud = 1 / (1 + exp(−k · (min(D, D_req) − (D_req − D_min) / 2)))

    ``D`` distance to the nearest cloud / shadow, ``D_req`` the distance
    at which the score saturates, ``D_min`` the minimum distance, ``k`` the
    slope; ``D``, ``D_req``, ``D_min`` share one unit and ``k`` is per it.
    """
    d = np.minimum(np.asarray(distance, dtype=np.float64), d_req)
    return 1.0 / (1.0 + np.exp(-slope * (d - (d_req - d_min) / 2.0)))


def _opacity_score(
    opacity: Float[np.ndarray, "*dims"], *, low: float, high: float
) -> Float[np.ndarray, "*dims"]:
    """Piecewise-linear atmospheric-opacity score (White et al., 2014).

        S_opacity = clip((τ_high − τ) / (τ_high − τ_low), 0, 1)

    i.e. 1 for ``τ ≤ τ_low``, 0 for ``τ ≥ τ_high``, linear between.
    """
    tau = np.asarray(opacity, dtype=np.float64)
    return np.clip((high - tau) / (high - low), 0.0, 1.0)


def _as_float_for_nan(values: Num[np.ndarray, "*dims"]) -> Float[np.ndarray, "*dims"]:
    return values.astype(np.result_type(values.dtype, np.float32), copy=False)


class MedianComposite(Operator):
    """Per-pixel median across a stack of co-registered GeoTensors.

    Metadata-independent: frames may also be plain ``np.ndarray`` maps, in
    which case the outputs are plain arrays and the grid check reduces to
    shape equality. Nodata frame-pixels (non-finite or the frame's fill)
    are treated like NaN; pixels invalid in every frame hold the output fill.

    Args:
        nan_policy: ``"ignore"`` skips NaN / nodata samples with
            ``np.nanmedian``; ``"propagate"`` uses ``np.median`` so any
            nodata sample makes the pixel NaN.
        return_count: When true, also return a carrier with the number
            of valid contributors per output pixel.
            The output is then a tuple, so the operator is terminal
            (last step only) in a ``Sequential``.
    """

    def __init__(
        self, *, nan_policy: NanPolicy = "ignore", return_count: bool = False
    ) -> None:
        self.nan_policy = _validate_nan_policy(nan_policy)
        self.return_count = return_count

    @property
    def _terminal(self) -> bool:  # ty: ignore[invalid-attribute-override]
        """A tuple output breaks carrier-in / carrier-out (last step only)."""
        return self.return_count

    def _apply(
        self, frames: Sequence[GeoTensor | np.ndarray]
    ) -> GeoTensor | np.ndarray | tuple[GeoTensor | np.ndarray, GeoTensor | np.ndarray]:
        frames = _as_frames(frames, type(self).__name__)
        base, stack = _stack_frames(frames)
        valid = _frame_validity(frames)
        masked = _mask_frames(stack, valid)
        with warnings.catch_warnings():
            # All-nodata pixels are expected; they get the output fill below.
            warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
            values = (
                np.nanmedian(masked, axis=0)
                if self.nan_policy == "ignore"
                else np.median(masked, axis=0)
            )
        fill = carried_fill(base, values.dtype)
        values = restore_fill(values, valid.any(axis=0), fill)
        out = wrap_like(base, values, fill_value_default=fill)
        if not self.return_count:
            return out
        count = np.sum(~np.isnan(masked), axis=0).astype(np.int64)
        return out, wrap_like(base, count, fill_value_default=0)


class MaxNDVIComposite(Operator):
    """Pick the frame with maximum NDVI per pixel and return its band values.

    Inputs must be multi-band (``(C, H, W)``); 2-D GeoTensors raise because
    NDVI needs distinct red and NIR bands. Nodata frame-pixels (non-finite
    or the frame's fill in any band) never win the selection. Pixels with
    no valid NDVI in any frame hold the first frame's
    ``fill_value_default`` (``NaN`` for float inputs without one), so the
    output dtype is the input dtype.

    The per-pixel math is metadata-independent, so plain ``np.ndarray``
    frames are accepted when ``red`` / ``nir`` are integer indices; named
    band references need GeoTensor ``attrs`` and raise ``TypeError`` for
    plain arrays.

    Args:
        red: Red band reference — integer band index or a band name
            resolved against the first frame's attrs.
        nir: NIR band reference, same conventions as ``red``.
        return_index: When true, also return a carrier holding the
            selected frame index per pixel (``int64``; ``-1``, the
            declared ``fill_value_default``, where no frame is valid).
            The output is then a tuple, so the operator is terminal
            (last step only) in a ``Sequential``.
        eps: Stabiliser added to the NDVI denominator.
    """

    def __init__(
        self,
        *,
        red: BandRef,
        nir: BandRef,
        return_index: bool = False,
        eps: float = 1e-10,
    ) -> None:
        self.red = red
        self.nir = nir
        self.return_index = return_index
        self.eps = eps

    @property
    def _terminal(self) -> bool:  # ty: ignore[invalid-attribute-override]
        """A tuple output breaks carrier-in / carrier-out (last step only)."""
        return self.return_index

    def _apply(
        self, frames: Sequence[GeoTensor | np.ndarray]
    ) -> GeoTensor | np.ndarray | tuple[GeoTensor | np.ndarray, GeoTensor | np.ndarray]:
        frames = _as_frames(frames, type(self).__name__)
        base, stack = _stack_frames(frames)
        # NDVI needs distinct red/nir bands; 2-D GeoTensors don't have a
        # band axis and would silently broadcast `:, red_idx, ...` into
        # nonsense. Fail loudly.
        if base.ndim < 3:
            raise ValueError(
                "MaxNDVIComposite requires multi-band GeoTensors (C, H, W); "
                f"got shape {base.shape}."
            )
        if (isinstance(self.red, str) or isinstance(self.nir, str)) and not hasattr(
            base, "attrs"
        ):
            raise TypeError(
                "MaxNDVIComposite with named band references requires "
                "GeoTensor inputs carrying band-name attrs; got a plain "
                "array. Pass integer band indices instead."
            )
        red_idx = resolve_band(base, self.red)
        nir_idx = resolve_band(base, self.nir)
        if red_idx == nir_idx:
            raise ValueError(
                "MaxNDVIComposite requires distinct red and NIR bands; both "
                f"resolved to band index {red_idx} (red={self.red!r}, "
                f"nir={self.nir!r})."
            )
        red = stack[:, red_idx, ...].astype(np.float32, copy=False)
        nir = stack[:, nir_idx, ...].astype(np.float32, copy=False)
        with np.errstate(invalid="ignore", divide="ignore"):
            ndvi = (nir - red) / (nir + red + self.eps)
        # A nodata frame (e.g. -9999 in both bands -> NDVI 0) must never win.
        frame_valid = _frame_validity(frames)
        scores = np.where(np.isnan(ndvi) | ~frame_valid, -np.inf, ndvi)
        index = np.argmax(scores, axis=0)
        values = _take_by_spatial_index(stack, index)
        all_invalid = np.all(~np.isfinite(scores), axis=0)
        fill = carried_fill(base, values.dtype)
        if np.any(all_invalid):
            if np.issubdtype(values.dtype, np.floating):
                values = restore_fill(values, ~all_invalid, fill)
            else:
                # Integer / unsigned inputs can't carry NaN. Fall back to the
                # input's fill_value_default so the dtype is preserved.
                if fill is None:
                    raise ValueError(
                        "MaxNDVIComposite: all NDVI scores are invalid for "
                        "some pixels and the integer-dtype input carries no "
                        "fill_value_default to mark them; use a GeoTensor "
                        "with fill_value_default or float frames."
                    )
                values[..., all_invalid] = fill
        out = wrap_like(base, values, fill_value_default=fill)
        if not self.return_index:
            return out
        # Frame 0 is a real index, so pixels no frame covers are -1.
        index = np.where(all_invalid, -1, index).astype(np.int64)
        return out, wrap_like(base, index, fill_value_default=-1)


class CloudFreeComposite(Operator):
    """Per-pixel mean over frames where the cloud mask is false.

    Consumes ``(frame, cloud_mask)`` pairs; masks may have spatial shape
    ``(H, W)``, ``(1, H, W)``, or match the frame shape exactly. Pixels
    with fewer than ``min_valid`` clear contributors come out as NaN.
    Nodata frame-pixels (non-finite or the frame's fill) never contribute;
    pixels invalid in every frame hold the output fill value.
    Metadata-independent: plain ``np.ndarray`` frames are accepted and
    yield plain-array outputs.

    Args:
        nan_policy: ``"ignore"`` (default) also drops NaN samples from
            clear pixels; ``"propagate"`` lets NaNs poison the mean.
        min_valid: Minimum number of clear contributors per pixel;
            pixels below the threshold are NaN. Must be at least 1.
        return_count: When true, also return a carrier with the number
            of clear contributors per pixel (``int64``).

    Raises:
            The output is then a tuple, so the operator is terminal
            (last step only) in a ``Sequential``.
        ValueError: If ``min_valid`` is below 1.
    """

    def __init__(
        self,
        *,
        nan_policy: NanPolicy = "ignore",
        min_valid: int = 1,
        return_count: bool = False,
    ) -> None:
        if min_valid < 1:
            raise ValueError("min_valid must be at least 1.")
        self.nan_policy = _validate_nan_policy(nan_policy)
        self.min_valid = min_valid
        self.return_count = return_count

    @property
    def _terminal(self) -> bool:  # ty: ignore[invalid-attribute-override]
        """A tuple output breaks carrier-in / carrier-out (last step only)."""
        return self.return_count

    def _apply(
        self, pairs: Sequence[tuple[GeoTensor | np.ndarray, Any]]
    ) -> GeoTensor | np.ndarray | tuple[GeoTensor | np.ndarray, GeoTensor | np.ndarray]:
        base, stack, cloudy = _require_pairs(pairs)
        frame_valid = _frame_validity([scene for scene, _ in pairs])
        stack = _mask_frames(stack, frame_valid)
        clear = ~cloudy
        valid = clear & ~np.isnan(stack) if self.nan_policy == "ignore" else clear
        count = np.sum(valid, axis=0)
        total = np.sum(np.where(valid, stack, 0), axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            values = total / count
        values = np.where(count >= self.min_valid, values, np.nan)
        values = _as_float_for_nan(values)
        fill = carried_fill(base, values.dtype)
        values = restore_fill(values, frame_valid.any(axis=0), fill)
        out = wrap_like(base, values, fill_value_default=fill)
        if not self.return_count:
            return out
        return out, wrap_like(base, count.astype(np.int64), fill_value_default=0)


class BAPComposite(Operator):
    """Best Available Pixel compositing from quality-score metadata.

    Each frame-pixel gets a weighted sum of per-criterion scores in
    ``[0, 1]`` and the highest-scoring valid frame wins (Griffiths et al.,
    2013; White et al., 2014):

        S = w_view · S_view + w_recency · S_doy
            + w_cloud · S_cloud + w_opacity · S_opacity

    Metadata may give precomputed ``view_angle_score``, ``recency_score``,
    ``cloud_distance_score`` and ``opacity_score`` values, or the raw
    values the scores are built from:

    * ``doy`` / ``day_of_year`` → Gaussian of the *circular* DOY distance
      (Griffiths et al.), so late December is close to early January::

          Δ = |doy − target_doy| mod 365,   d = min(Δ, 365 − Δ)
          S_doy = exp(−½ · (d / σ_doy)²)

    * ``view_angle`` (view zenith angle, degrees; sign ignored)::

          S_view = exp(−½ · (θ / σ_θ)²)

    * ``cloud_distance`` (distance to the nearest cloud / shadow, in the
      unit of ``cloud_distance_req``, e.g. pixels) → Griffiths et al.
      eq. 2 sigmoid::

          S_cloud = 1 / (1 + exp(−k · (min(D, D_req) − (D_req − D_min) / 2)))

    * ``opacity`` (unitless atmospheric opacity τ, e.g. LEDAPS
      ``atmos_opacity``) → White et al. (2014) ramp::

          S_opacity = clip((τ_high − τ) / (τ_high − τ_low), 0, 1)

    Missing raw values default to the best case (``doy = target_doy``,
    ``view_angle = 0``, ``opacity = 0``), except ``cloud_distance``, which
    defaults to ``0`` (next to a cloud). Every score lies in ``[0, 1]``, so
    raw and precomputed values may be mixed across frames. Each value may
    be a scalar or a per-pixel array. Frames may be GeoTensors or plain
    ``np.ndarray`` maps (the scores live in the metadata dicts, not in
    geo-metadata). A nodata frame-pixel (non-finite or the frame's fill)
    or a NaN score is never selected; pixels with no valid frame hold the
    output fill value (also in the score output). Griffiths et al. also
    score acquisition year and sensor; those have no raw input here.

    Args:
        target_doy: Day of year (1-366) the DOY score is centred on.
        w_view_angle: Weight of the view-angle score.
        w_recency: Weight of the DOY score.
        w_cloud_distance: Weight of the cloud-distance score.
        w_opacity: Weight of the opacity score.
        doy_sigma: ``σ_doy``, width of the DOY Gaussian in days. Default
            ``30`` (a scene 30 days off target scores e^(−½) ≈ 0.61).
        view_angle_sigma: ``σ_θ``, width of the view-angle Gaussian in
            degrees. Default ``15`` (7.5°, the Landsat swath edge, scores
            ≈ 0.88).
        cloud_distance_req: ``D_req``, distance at which the cloud
            score saturates. Default ``50`` (pixels; 1.5 km at 30 m).
        cloud_distance_min: ``D_min`` of the sigmoid. Default ``0``.
        cloud_distance_slope: Sigmoid slope ``k`` per distance unit.
            Default ``0.2``.
        opacity_low: ``τ_low``, opacity at or below which the score is 1.
            Default ``0.2``.
        opacity_high: ``τ_high``, opacity at or above which the score is
            0. Default ``0.3``.
        return_score: When true, also return a carrier holding the
            winning per-pixel score (``float32``).
            The output is then a tuple, so the operator is terminal
            (last step only) in a ``Sequential``.

    Raises:
        ValueError: If ``doy_sigma``, ``view_angle_sigma`` or
            ``cloud_distance_slope`` is not positive, or
            ``cloud_distance_req ≤ cloud_distance_min`` or
            ``opacity_high ≤ opacity_low``.

    References:
        Griffiths, P., van der Linden, S., Kuemmerle, T. & Hostert, P.
        (2013). A pixel-based Landsat compositing algorithm for large area
        land cover mapping. IEEE JSTARS 6(5), 2088-2101.

        White, J. C., Wulder, M. A., Hobart, G. W., et al. (2014).
        Pixel-based image compositing for large-area dense time series
        applications and science. Canadian Journal of Remote Sensing
        40(3), 192-212.
    """

    def __init__(
        self,
        *,
        target_doy: int,
        w_view_angle: float = 0.3,
        w_recency: float = 0.4,
        w_cloud_distance: float = 0.2,
        w_opacity: float = 0.1,
        doy_sigma: float = 30.0,
        view_angle_sigma: float = 15.0,
        cloud_distance_req: float = 50.0,
        cloud_distance_min: float = 0.0,
        cloud_distance_slope: float = 0.2,
        opacity_low: float = 0.2,
        opacity_high: float = 0.3,
        return_score: bool = False,
    ) -> None:
        if min(doy_sigma, view_angle_sigma, cloud_distance_slope) <= 0:
            raise ValueError(
                "doy_sigma, view_angle_sigma and cloud_distance_slope must be positive."
            )
        if cloud_distance_req <= cloud_distance_min:
            raise ValueError("cloud_distance_req must exceed cloud_distance_min.")
        if opacity_high <= opacity_low:
            raise ValueError("opacity_high must exceed opacity_low.")
        self.target_doy = target_doy
        self.w_view_angle = w_view_angle
        self.w_recency = w_recency
        self.w_cloud_distance = w_cloud_distance
        self.w_opacity = w_opacity
        self.doy_sigma = doy_sigma
        self.view_angle_sigma = view_angle_sigma
        self.cloud_distance_req = cloud_distance_req
        self.cloud_distance_min = cloud_distance_min
        self.cloud_distance_slope = cloud_distance_slope
        self.opacity_low = opacity_low
        self.opacity_high = opacity_high
        self.return_score = return_score

    @property
    def _terminal(self) -> bool:  # ty: ignore[invalid-attribute-override]
        """A tuple output breaks carrier-in / carrier-out (last step only)."""
        return self.return_score

    def _frame_scores(
        self, metadata: Mapping[str, Any], spatial_shape: tuple[int, int]
    ) -> Float[np.ndarray, "4 h w"]:
        """Stack one frame's (view, DOY, cloud-distance, opacity) scores."""

        def raw(*names: str, default: float) -> Float[np.ndarray, "h w"]:
            value = _metadata_value(metadata, *names, default=default)
            return _score_array(value, spatial_shape)

        view = _metadata_value(metadata, "view_angle_score")
        if view is None:
            view = _view_angle_score(
                raw("view_angle", default=0.0), self.view_angle_sigma
            )
        recency = _metadata_value(metadata, "recency_score")
        if recency is None:
            recency = _doy_score(
                raw("doy", "day_of_year", default=float(self.target_doy)),
                self.target_doy,
                self.doy_sigma,
            )
        cloud = _metadata_value(metadata, "cloud_distance_score")
        if cloud is None:
            cloud = _cloud_distance_score(
                raw("cloud_distance", default=0.0),
                d_req=self.cloud_distance_req,
                d_min=self.cloud_distance_min,
                slope=self.cloud_distance_slope,
            )
        opacity = _metadata_value(metadata, "opacity_score")
        if opacity is None:
            opacity = _opacity_score(
                raw("opacity", default=0.0),
                low=self.opacity_low,
                high=self.opacity_high,
            )
        return np.stack(
            [_score_array(s, spatial_shape) for s in (view, recency, cloud, opacity)]
        )

    def _apply(
        self, pairs: Sequence[tuple[GeoTensor | np.ndarray, Mapping[str, Any]]]
    ) -> GeoTensor | np.ndarray | tuple[GeoTensor | np.ndarray, GeoTensor | np.ndarray]:
        if not pairs:
            raise ValueError("At least one (GeoTensor, metadata) pair is required.")
        frames = [scene for scene, _ in pairs]
        base, stack = _stack_frames(frames)
        spatial_shape = base.shape[-2:]
        # (T, 4, H, W): view, DOY, cloud-distance, opacity score per frame.
        scores = np.stack(
            [self._frame_scores(metadata, spatial_shape) for _, metadata in pairs]
        )
        weights = np.array(
            [self.w_view_angle, self.w_recency, self.w_cloud_distance, self.w_opacity],
            dtype=np.float32,
        )
        score_stack = np.einsum("k,tkhw->thw", weights, scores).astype(
            np.float32, copy=False
        )
        # A nodata frame-pixel (or a NaN score) must never win the argmax.
        frame_valid = _frame_validity(frames)
        score_stack = np.where(
            frame_valid & ~np.isnan(score_stack), score_stack, -np.inf
        ).astype(np.float32, copy=False)
        index = np.argmax(score_stack, axis=0)
        any_valid = np.isfinite(score_stack).any(axis=0)
        values = _take_by_spatial_index(stack, index)
        fill = carried_fill(base, values.dtype)
        values = restore_fill(values, any_valid, fill)
        out = wrap_like(base, values, fill_value_default=fill)
        if not self.return_score:
            return out
        best_score = np.take_along_axis(score_stack, index[None, ...], axis=0)[0]
        best_score = restore_fill(best_score, any_valid, np.nan)
        return out, wrap_like(base, best_score, fill_value_default=np.nan)


class MinCloudComposite(Operator):
    """Pick each pixel from the scene with the lowest *global* cloud coverage.

    For every pixel, the selected frame is the one with the smallest
    scene-wide cloud fraction among those where the pixel is clear. Pixels
    cloudy in every frame fall back to the globally least-cloudy frame so
    the output is a complete composite. Nodata frame-pixels (non-finite or
    the frame's fill) are never picked and do not count towards a scene's
    cloud fraction; pixels invalid in every frame hold the output fill.

    This is a coarse cloud-aware composite, not a per-pixel
    cloud-distance composite — frames are ranked by their overall cloud
    coverage rather than by distance-to-nearest-cloud at each pixel. Use
    :class:`BAPComposite` with per-pixel ``cloud_distance`` metadata when
    that finer granularity is needed.

    Metadata-independent: plain ``np.ndarray`` frames are accepted and
    yield plain-array outputs.

    Args:
        return_count: When true, also return a carrier with the number
            of clear contributors per pixel (``int64``).
            The output is then a tuple, so the operator is terminal
            (last step only) in a ``Sequential``.
    """

    def __init__(self, *, return_count: bool = False) -> None:
        self.return_count = return_count

    @property
    def _terminal(self) -> bool:  # ty: ignore[invalid-attribute-override]
        """A tuple output breaks carrier-in / carrier-out (last step only)."""
        return self.return_count

    def _apply(
        self, pairs: Sequence[tuple[GeoTensor | np.ndarray, Any]]
    ) -> GeoTensor | np.ndarray | tuple[GeoTensor | np.ndarray, GeoTensor | np.ndarray]:
        base, stack, cloudy = _require_pairs(pairs)
        valid = _broadcast_frame_valid(
            _frame_validity([scene for scene, _ in pairs]), cloudy.shape
        )
        clear = ~cloudy & valid
        # Scene-wide cloud fraction over each frame's *valid* pixels only; a
        # frame with no valid pixel counts as fully cloudy.
        n_valid = valid.reshape((valid.shape[0], -1)).sum(axis=1)
        n_cloudy = (cloudy & valid).reshape((valid.shape[0], -1)).sum(axis=1)
        cloud_coverage = np.where(
            n_valid > 0, n_cloudy / np.maximum(n_valid, 1), 1.0
        ).reshape((-1, *([1] * (clear.ndim - 1))))
        # Clear pixels win; cloudy-but-valid pixels fall back to the least
        # cloudy valid frame (the +2 keeps them behind every clear option);
        # nodata frame-pixels are never picked.
        costs = np.where(
            clear, cloud_coverage, np.where(valid, cloud_coverage + 2.0, np.inf)
        )
        index = np.argmin(costs, axis=0)
        values = _take_by_spatial_index(stack, index)
        fill = carried_fill(base, values.dtype)
        values = restore_fill(values, valid.any(axis=0), fill)
        out = wrap_like(base, values, fill_value_default=fill)
        if not self.return_count:
            return out
        count = np.sum(clear, axis=0).astype(np.int64)
        return out, wrap_like(base, count, fill_value_default=0)


__all__ = [
    "BAPComposite",
    "CloudFreeComposite",
    "MaxNDVIComposite",
    "MedianComposite",
    "MinCloudComposite",
]
