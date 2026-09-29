"""Carrier-aware compositing operators for co-registered GeoTensors.

All composites are metadata-independent per-pixel reductions, so they
accept sequences of plain ``np.ndarray`` frames as well as GeoTensors.
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


def _grid_matches(a: GeoTensor | np.ndarray, b: GeoTensor | np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    a_transform = getattr(a, "transform", None)
    b_transform = getattr(b, "transform", None)
    if a_transform is None or b_transform is None:
        # Plain arrays carry no georeferencing — shape equality is the only
        # co-registration check available (mirrors mask's hasattr guards).
        return True
    # Affine equality is exact, not tolerant: sub-pixel grid drift is a real
    # bug source and should fail loudly. Some other geotoolz modules use
    # ``np.allclose`` on transforms; compositing intentionally tightens that
    # because a per-pixel reduction over misaligned grids silently produces
    # garbage.
    return a_transform == b_transform and getattr(a, "crs", None) == getattr(
        b, "crs", None
    )


def _require_frames(
    frames: Sequence[GeoTensor | np.ndarray],
) -> GeoTensor | np.ndarray:
    if not frames:
        raise ValueError("At least one GeoTensor is required for compositing.")
    base = frames[0]
    for idx, frame in enumerate(frames[1:], start=1):
        if not _grid_matches(base, frame):
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


def _normalize_positive(
    stack: Float[np.ndarray, "t h w"],
) -> Float[np.ndarray, "t h w"]:
    """Normalize by the maximum finite positive value, or return zeros."""
    max_value = np.nanmax(stack)
    if not np.isfinite(max_value) or max_value <= 0:
        return np.zeros_like(stack, dtype=np.float32)
    return np.asarray(stack / max_value, dtype=np.float32)


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

    Metadata may provide precomputed ``*_score`` arrays, or raw ``view_angle``,
    ``doy``, ``cloud_distance``, and ``opacity`` values used to build simple
    scores. Each value may be a scalar or a per-pixel array. Frames may be
    GeoTensors or plain ``np.ndarray`` maps (the scores live in the metadata
    dicts, not in geo-metadata). A nodata frame-pixel (non-finite or the
    frame's fill) or a NaN score is never selected; pixels with no valid
    frame hold the output fill value (also in the score output).

    Args:
        target_doy: Day-of-year the recency score is anchored to.
        w_view_angle: Weight of the view-angle score.
        w_recency: Weight of the recency score.
        w_cloud_distance: Weight of the cloud-distance score.
        w_opacity: Weight of the opacity score.
        return_score: When true, also return a carrier holding the
            winning per-pixel score (``float32``).
            The output is then a tuple, so the operator is terminal
            (last step only) in a ``Sequential``.
    """

    def __init__(
        self,
        *,
        target_doy: int,
        w_view_angle: float = 0.3,
        w_recency: float = 0.4,
        w_cloud_distance: float = 0.2,
        w_opacity: float = 0.1,
        return_score: bool = False,
    ) -> None:
        self.target_doy = target_doy
        self.w_view_angle = w_view_angle
        self.w_recency = w_recency
        self.w_cloud_distance = w_cloud_distance
        self.w_opacity = w_opacity
        self.return_score = return_score

    @property
    def _terminal(self) -> bool:  # ty: ignore[invalid-attribute-override]
        """A tuple output breaks carrier-in / carrier-out (last step only)."""
        return self.return_score

    def _apply(
        self, pairs: Sequence[tuple[GeoTensor | np.ndarray, Mapping[str, Any]]]
    ) -> GeoTensor | np.ndarray | tuple[GeoTensor | np.ndarray, GeoTensor | np.ndarray]:
        if not pairs:
            raise ValueError("At least one (GeoTensor, metadata) pair is required.")
        frames = [scene for scene, _ in pairs]
        base, stack = _stack_frames(frames)
        spatial_shape = base.shape[-2:]
        view_scores = []
        recency_scores = []
        cloud_distance_scores = []
        raw_cloud_distance = []
        opacity_scores = []
        for _, metadata in pairs:
            view = _metadata_value(metadata, "view_angle_score")
            if view is None:
                view_angle = _score_array(
                    _metadata_value(metadata, "view_angle", default=0.0), spatial_shape
                )
                view = 1.0 / (1.0 + np.abs(view_angle))
            view_scores.append(_score_array(view, spatial_shape))

            recency = _metadata_value(metadata, "recency_score")
            if recency is None:
                doy = _score_array(
                    _metadata_value(
                        metadata, "doy", "day_of_year", default=self.target_doy
                    ),
                    spatial_shape,
                )
                recency = 1.0 / (1.0 + np.abs(doy - self.target_doy))
            recency_scores.append(_score_array(recency, spatial_shape))

            cloud_distance = _metadata_value(metadata, "cloud_distance_score")
            if cloud_distance is None:
                cloud_distance = _score_array(
                    _metadata_value(metadata, "cloud_distance", default=0.0),
                    spatial_shape,
                )
                raw_cloud_distance.append(True)
            else:
                raw_cloud_distance.append(False)
            cloud_distance_scores.append(_score_array(cloud_distance, spatial_shape))

            opacity = _metadata_value(metadata, "opacity_score")
            if opacity is None:
                opacity_value = _score_array(
                    _metadata_value(metadata, "opacity", default=0.0), spatial_shape
                )
                opacity = 1.0 - np.clip(opacity_value, 0.0, 1.0)
            opacity_scores.append(_score_array(opacity, spatial_shape))

        if any(raw_cloud_distance) and not all(raw_cloud_distance):
            # Raw ``cloud_distance`` values (typically pixel/meter scale) and
            # precomputed ``cloud_distance_score`` values (0-1) live on
            # incompatible scales. Silently mixing them would let raw
            # distances dominate the weighted sum, so refuse rather than
            # produce metadata-dependent rankings.
            raise ValueError(
                "BAPComposite received a mix of raw 'cloud_distance' and "
                "precomputed 'cloud_distance_score' metadata across frames. "
                "Provide the same representation for every frame so the "
                "values share a common scale."
            )
        cloud_distance_stack = np.stack(cloud_distance_scores, axis=0)
        if all(raw_cloud_distance):
            cloud_distance_stack = _normalize_positive(cloud_distance_stack)
        score_stack = (
            self.w_view_angle * np.stack(view_scores, axis=0)
            + self.w_recency * np.stack(recency_scores, axis=0)
            + self.w_cloud_distance * cloud_distance_stack
            + self.w_opacity * np.stack(opacity_scores, axis=0)
        ).astype(np.float32, copy=False)
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
