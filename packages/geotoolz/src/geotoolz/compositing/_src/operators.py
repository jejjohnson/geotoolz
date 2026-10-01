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

Multi-source fusion: :class:`StackMatched` and :class:`BlendMatched` are
the siblings of the temporal composites for a *matched tuple* of tensors
from different sources (same grid, typically aligned by a
``geotoolz.geom.coregister`` operator first) rather than a temporal stack
of one sensor. "Matched" means matched *grids* -- unrelated to the
``geotoolz.matched_filter`` target-detection family.

The per-pixel maths lives in :mod:`geotoolz.compositing._src.array`.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from jaxtyping import Bool, Float, Shaped
from pipekit import Operator

from geotoolz._src.bands import (
    BandRef,
    concat_band_attrs,
    resolve_band,
    strip_band_attrs,
)
from geotoolz._src.geo import grid_matches
from geotoolz._src.valid import (
    carried_fill,
    invalid_values,
    is_fill,
    restore_fill,
    valid_pixels,
)
from geotoolz._src.wrap import wrap_like
from geotoolz.compositing._src.array import (
    BlendMethod,
    NanPolicy,
    bap_scores,
    blend_weighted,
    broadcast_frame_valid,
    cloud_distance_score,
    doy_score,
    mask_frames,
    mean_composite,
    median_composite,
    opacity_score,
    take_by_spatial_index,
    view_angle_score,
)
from geotoolz.indices._src.array import ndvi


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor


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
        masked = mask_frames(stack, valid)
        # All-nodata pixels come out NaN; they get the output fill below.
        values = median_composite(masked, nan_policy=self.nan_policy)
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
        with np.errstate(invalid="ignore", divide="ignore"):
            ndvi_map = ndvi(stack, nir_idx, red_idx, axis=1, eps=self.eps)
        # A nodata frame (e.g. -9999 in both bands -> NDVI 0) must never win.
        frame_valid = _frame_validity(frames)
        scores = np.where(np.isnan(ndvi_map) | ~frame_valid, -np.inf, ndvi_map)
        index = np.argmax(scores, axis=0)
        values = take_by_spatial_index(stack, index)
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
            The output is then a tuple, so the operator is terminal
            (last step only) in a ``Sequential``.

    Raises:
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
        values, count = mean_composite(
            mask_frames(stack, frame_valid),
            ~cloudy,
            min_valid=self.min_valid,
            nan_policy=self.nan_policy,
        )
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
            view = view_angle_score(
                raw("view_angle", default=0.0), self.view_angle_sigma
            )
        recency = _metadata_value(metadata, "recency_score")
        if recency is None:
            recency = doy_score(
                raw("doy", "day_of_year", default=float(self.target_doy)),
                self.target_doy,
                self.doy_sigma,
            )
        cloud = _metadata_value(metadata, "cloud_distance_score")
        if cloud is None:
            cloud = cloud_distance_score(
                raw("cloud_distance", default=0.0),
                d_req=self.cloud_distance_req,
                d_min=self.cloud_distance_min,
                slope=self.cloud_distance_slope,
            )
        opacity = _metadata_value(metadata, "opacity_score")
        if opacity is None:
            opacity = opacity_score(
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
        score_stack = bap_scores(scores, weights)
        # A nodata frame-pixel (or a NaN score) must never win the argmax.
        frame_valid = _frame_validity(frames)
        score_stack = np.where(
            frame_valid & ~np.isnan(score_stack), score_stack, -np.inf
        ).astype(np.float32, copy=False)
        index = np.argmax(score_stack, axis=0)
        any_valid = np.isfinite(score_stack).any(axis=0)
        values = take_by_spatial_index(stack, index)
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
        valid = broadcast_frame_valid(
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
        values = take_by_spatial_index(stack, index)
        fill = carried_fill(base, values.dtype)
        values = restore_fill(values, valid.any(axis=0), fill)
        out = wrap_like(base, values, fill_value_default=fill)
        if not self.return_count:
            return out
        count = np.sum(clear, axis=0).astype(np.int64)
        return out, wrap_like(base, count, fill_value_default=0)


def _normalize_to_sequence(
    tensors: Sequence[GeoTensor | np.ndarray] | Mapping[str, GeoTensor | np.ndarray],
    order: list[str] | None,
) -> tuple[list[GeoTensor | np.ndarray], list[str] | None]:
    """Accept either a Sequence or a Mapping; return a parallel sequence + names.

    When the input is a Mapping, ``order`` (if given) **must cover
    every key** — missing names raise, extra names raise. This is
    strict by design: silently dropping a key that's present in the
    input would mask configuration drift (e.g. a new source added to
    `MatchedPatch.members` without updating the stack config). Users
    who genuinely want a subset should slice the input dict before
    passing it in.

    A sequence input ignores ``order`` (no key→pos mapping to apply).
    """
    if isinstance(tensors, Mapping):
        if order is not None:
            order_set = set(order)
            input_set = set(tensors)
            missing = sorted(order_set - input_set)
            extra = sorted(input_set - order_set)
            if missing or extra:
                msgs = []
                if missing:
                    msgs.append(f"missing from input: {missing!r}")
                if extra:
                    msgs.append(f"extra in input but not in order: {extra!r}")
                raise KeyError(
                    "StackMatched.order must cover every input key exactly; "
                    + "; ".join(msgs)
                    + ". Slice the input dict first if you want a subset."
                )
            ordered = [tensors[k] for k in order]
            return ordered, list(order)
        return list(tensors.values()), list(tensors.keys())
    return list(tensors), None


def _as_band_first(
    values: Shaped[np.ndarray, "*bands h w"],
) -> Shaped[np.ndarray, "c h w"]:
    """Promote ``(H, W)`` to ``(1, H, W)``; leave ``(C, H, W)`` alone."""
    if values.ndim == 2:
        return values[None, :, :]
    if values.ndim == 3:
        return values
    raise ValueError(
        "StackMatched expects 2-D (H, W) or 3-D (C, H, W) tensors; "
        f"got ndim={values.ndim}."
    )


def _translate_fills(
    stacked: Shaped[np.ndarray, "c h w"],
    seq: Sequence[GeoTensor | np.ndarray],
    arrays: Sequence[np.ndarray],
    out_fill: object,
) -> Shaped[np.ndarray, "c h w"]:
    """Rewrite each input's nodata pixels to the output fill, in its own bands.

    The output keeps the first input's ``fill_value_default`` (``NaN``
    when concatenation promotes integer bands to float); an input
    with a different fill would otherwise leave nodata that the output's
    fill no longer marks. When the output fill cannot be represented in
    the concatenated dtype, the input's values are left untouched.
    """
    offset = 0
    for tensor, arr in zip(seq, arrays, strict=True):
        n = arr.shape[0]
        valid = valid_pixels(tensor)
        block = stacked[offset : offset + n]
        if not valid.all() and not is_fill(block[:, ~valid], out_fill).all():
            with contextlib.suppress(ValueError):
                stacked[offset : offset + n] = restore_fill(block, valid, out_fill)
        offset += n
    return stacked


class StackMatched(Operator):
    """Concatenate aligned tensors along the band axis.

    Inputs are either a `Sequence[GeoTensor]` or a
    ``Mapping[str, GeoTensor]`` — typical when called on
    `MatchedPatch.members`. All inputs must share spatial shape,
    transform, and CRS; the per-tensor band count may differ. Plain
    ``np.ndarray`` inputs are also accepted (the concatenation itself is
    metadata-free); grid verification then degrades to spatial-shape
    equality and the output carrier follows the first input.

    Args:
        order: When the input is a Mapping, this list fixes the
            stacking order. Must cover every key of the input
            mapping **exactly** — extra or missing names raise. (If
            you want a subset, slice the input dict before passing
            it in; this avoids silently dropping a key the user
            forgot to update.) Ignored for Sequence inputs.

    Examples:
        >>> import geotoolz as gz
        >>> stack = gz.compositing.StackMatched(order=["modis", "s2"])
        >>> fused = stack({"modis": modis_chip, "s2": s2_chip_aligned})
        >>> fused.shape  # (modis_bands + s2_bands, H, W)
        (5, 256, 256)

    Notes:
        Per-band attrs (``band_names``, ``descriptions``, ``wavelengths``,
        ...) are concatenated in stacking order when every input carries
        them; a key missing from any input is dropped. Other attrs follow
        the first input. NaN-fill padding on grid mismatch is tracked for
        a future revision; today the operator requires strict grid
        equality. Pre-coregister with
        ``geotoolz.geom.coregister.RasterToRasterLike`` if the
        inputs aren't already on the same grid. Nodata pixels of each
        input (non-finite or that input's fill) are rewritten to the
        output's fill (the first input's ``fill_value_default``) in that
        input's bands, when the output dtype can represent it.

        Where :meth:`georeader.geotensor.GeoTensor.concatenate` applies
        (3-D inputs with equal band counts and equal, non-NaN fills) the
        output matches it. The operator does not delegate to it because
        ``concatenate`` refuses differing band counts, 2-D inputs, and
        NaN fills (``NaN != NaN``), and it keeps only the first input's
        attrs.
    """

    def __init__(
        self,
        *,
        order: list[str] | None = None,
    ) -> None:
        self.order = list(order) if order is not None else None

    def _apply(
        self,
        tensors: Sequence[GeoTensor | np.ndarray]
        | Mapping[str, GeoTensor | np.ndarray],
    ) -> GeoTensor | np.ndarray:
        seq, _names = _normalize_to_sequence(tensors, self.order)
        if not seq:
            raise ValueError("StackMatched requires at least one input tensor.")

        # Validate grids exactly — silent affine drift on a per-pixel
        # fused stack is a real bug source; we'd rather fail loudly
        # than emit subtly misregistered output.
        base = seq[0]
        for idx, frame in enumerate(seq[1:], start=1):
            if not grid_matches(base, frame):
                raise ValueError(
                    "StackMatched inputs must share spatial shape, "
                    "transform, and CRS; "
                    f"input 0 has shape {base.shape[-2:]}, "
                    f"transform {getattr(base, 'transform', None)!r}; "
                    f"input {idx} has shape {frame.shape[-2:]}, "
                    f"transform {getattr(frame, 'transform', None)!r}."
                )

        arrays = [_as_band_first(np.asarray(t)) for t in seq]
        stacked = np.concatenate(arrays, axis=0)
        fill = carried_fill(base, stacked.dtype)
        stacked = _translate_fills(stacked, seq, arrays, fill)
        attrs = strip_band_attrs(getattr(base, "attrs", None))
        attrs.update(
            concat_band_attrs(
                [getattr(t, "attrs", None) for t in seq],
                [arr.shape[0] for arr in arrays],
            )
        )
        return wrap_like(base, stacked, fill_value_default=fill, attrs=attrs)


class BlendMatched(Operator):
    """Weighted mean across N aligned tensors.

    Three blending modes:

    * ``"mean"`` — equal-weight average across all inputs. Best for
      ensemble-style fusion where every source is equally trustworthy.
    * ``"weighted_mean"`` — per-source scalar weights from
      ``self.weights``. Useful when one source is known to be
      higher-quality (e.g. ground-truth vs satellite).
    * ``"ivw"`` — inverse-variance weighting. Each input is weighted by
      ``1 / variance``, so noisier sources contribute less. Requires
      a parallel ``variances`` sequence at call time, one
      per-source variance array (same spatial shape as the data).

    `nan_policy` controls per-pixel NaN handling:

    * ``"ignore"`` — exclude NaN samples from the blend; the
      surviving weights renormalise. If every input is NaN at a pixel,
      the output is NaN.
    * ``"propagate"`` — any NaN at a pixel poisons the output pixel.

    Nodata pixels of each input (non-finite, or that input's
    ``fill_value_default``, in any band) are treated as NaN; pixels that
    are nodata in every input hold the output fill: the first input's
    ``fill_value_default``, or ``NaN`` when it has none or is an integer
    input blended into float (see :func:`geotoolz._src.valid.carried_fill`).

    All inputs must share spatial shape, transform, and CRS. The
    band axis must also be uniform (use `StackMatched` if you want
    cross-source band concatenation; `BlendMatched` is the per-pixel
    averaging counterpart). Plain ``np.ndarray`` inputs are also
    accepted (the blend is metadata-free per-pixel math); grid
    verification then degrades to shape equality and the output
    carrier follows the first input.

    When ``tensors`` is a ``Mapping``, the per-source order used by
    ``weighted_mean`` / ``ivw`` follows the mapping's iteration order
    (insertion order on dicts). Likewise, ``weights`` and ``variances``
    are zipped positionally against ``tensors`` — pass an ``OrderedDict``
    or a plain ``list``/``tuple`` if you need a stable, explicit pairing.

    Args:
        method: One of ``"mean"`` / ``"weighted_mean"`` / ``"ivw"``.
        weights: Per-source scalar weights. Required when
            ``method="weighted_mean"``; must be ``None`` for the other
            methods (passing weights with another method raises).
            Length must equal the number of input tensors at call time.
        nan_policy: ``"ignore"`` (default) or ``"propagate"``.
    """

    def __init__(
        self,
        *,
        method: BlendMethod = "mean",
        weights: list[float] | None = None,
        nan_policy: NanPolicy = "ignore",
    ) -> None:
        if method not in {"mean", "weighted_mean", "ivw"}:
            raise ValueError(
                f"BlendMatched.method must be 'mean', 'weighted_mean', or "
                f"'ivw'; got {method!r}"
            )
        if nan_policy not in {"ignore", "propagate"}:
            raise ValueError(
                f"BlendMatched.nan_policy must be 'ignore' or 'propagate'; "
                f"got {nan_policy!r}"
            )
        if method == "weighted_mean" and weights is None:
            raise ValueError("BlendMatched(method='weighted_mean') requires `weights`.")
        if method != "weighted_mean" and weights is not None:
            raise ValueError(
                "BlendMatched `weights` only applies to "
                "method='weighted_mean'. For per-pixel variance "
                "weighting use method='ivw' with a `variances` argument."
            )
        self.method = method
        self.weights = list(weights) if weights is not None else None
        self.nan_policy = nan_policy

    def _apply(
        self,
        tensors: Sequence[GeoTensor | np.ndarray]
        | Mapping[str, GeoTensor | np.ndarray],
        variances: Sequence[np.ndarray] | None = None,
    ) -> GeoTensor | np.ndarray:
        if variances is not None and self.method != "ivw":
            raise ValueError(
                "BlendMatched: `variances` is only accepted when "
                f"method='ivw'; got method={self.method!r}."
            )
        seq, _names = _normalize_to_sequence(tensors, None)
        if not seq:
            raise ValueError("BlendMatched requires at least one input tensor.")

        # Strict grid + band-shape validation. BlendMatched is a
        # per-pixel reduction across inputs, so any shape difference
        # would mean we're averaging different physical quantities.
        base = seq[0]
        for idx, frame in enumerate(seq[1:], start=1):
            if not grid_matches(base, frame):
                raise ValueError(
                    "BlendMatched inputs must share spatial shape, "
                    "transform, and CRS; "
                    f"input 0 has shape {base.shape[-2:]}, "
                    f"transform {getattr(base, 'transform', None)!r}; "
                    f"input {idx} has shape {frame.shape[-2:]}, "
                    f"transform {getattr(frame, 'transform', None)!r}."
                )
            if frame.shape != base.shape:
                raise ValueError(
                    "BlendMatched inputs must share full shape (including "
                    f"band axis); input 0 has shape {base.shape}, "
                    f"input {idx} has shape {frame.shape}."
                )

        # Stack along a new "source" axis at position 0. Shape is now
        # (N, ...spatial...) for 2-D or (N, C, H, W) for 3-D. Each input's
        # nodata pixels (non-finite or its own fill) become NaN in every
        # band, so the NaN policy below excludes / propagates them.
        stack = np.stack([np.asarray(t).astype(np.float64) for t in seq], axis=0)
        source_valid = np.stack([valid_pixels(t) for t in seq], axis=0)
        if not source_valid.all():
            per_source = source_valid[:, None] if stack.ndim == 4 else source_valid
            stack[~np.broadcast_to(per_source, stack.shape)] = np.nan

        # Build the per-source weight broadcastable to `stack`.
        if self.method == "ivw":
            if variances is None:
                raise ValueError(
                    "BlendMatched(method='ivw') requires `variances` "
                    "(one array per source, same spatial shape as the data)."
                )
            var_list = list(variances)
            if len(var_list) != len(seq):
                raise ValueError(
                    f"BlendMatched(method='ivw'): got {len(var_list)} "
                    f"variance arrays for {len(seq)} input tensors."
                )
            # Each variance must be either a full-shape array matching
            # the data or a spatial-only (H, W) array that broadcasts
            # against the band axis. Validate shape explicitly so a
            # silent broadcast can't hide a mis-shaped variance.
            var_arrays: list[np.ndarray] = []
            spatial_shape = base.shape[-2:]
            for i, v in enumerate(var_list):
                arr = np.asarray(v, dtype=np.float64)
                if arr.shape != base.shape and arr.shape != spatial_shape:
                    raise ValueError(
                        f"BlendMatched(method='ivw'): variance {i} has "
                        f"shape {arr.shape}; expected {base.shape} "
                        f"(full) or {spatial_shape} (spatial-only)."
                    )
                if not np.all(np.isfinite(arr)):
                    raise ValueError(
                        f"BlendMatched(method='ivw'): variance {i} "
                        "contains non-finite values (NaN/Inf); "
                        "IVW weights are undefined."
                    )
                if np.any(arr <= 0):
                    raise ValueError(
                        f"BlendMatched(method='ivw'): variance {i} "
                        "contains non-positive values; variances must "
                        "be strictly positive."
                    )
                var_arrays.append(np.broadcast_to(arr, base.shape))
            var_stack = np.stack(var_arrays, axis=0)
            w = 1.0 / var_stack
        elif self.method == "weighted_mean":
            assert self.weights is not None  # guarded in __init__
            if len(self.weights) != len(seq):
                raise ValueError(
                    f"BlendMatched(weights=...): got {len(self.weights)} "
                    f"weights for {len(seq)} input tensors."
                )
            w_arr = np.asarray(self.weights, dtype=np.float64)
            # Keep weights as (N, 1, ..., 1) so they broadcast against
            # `stack` without materialising a full (N, *spatial) array.
            w_shape = (len(seq),) + (1,) * (stack.ndim - 1)
            w = w_arr.reshape(w_shape)
        else:  # "mean"
            w = np.ones((len(seq),) + (1,) * (stack.ndim - 1), dtype=np.float64)

        result = blend_weighted(stack, w, nan_policy=self.nan_policy)

        fill = carried_fill(base, result.dtype)
        result = restore_fill(result, source_valid.any(axis=0), fill)
        return wrap_like(base, result, fill_value_default=fill)


__all__ = [
    "BAPComposite",
    "BlendMatched",
    "CloudFreeComposite",
    "MaxNDVIComposite",
    "MedianComposite",
    "MinCloudComposite",
    "StackMatched",
]
