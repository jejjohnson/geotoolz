"""Carrier-aware compositing operators for co-registered GeoTensors.

All composites are metadata-independent per-pixel reductions, so they
accept sequences of plain ``np.ndarray`` frames as well as GeoTensors.
Grid checks (transform / CRS equality) apply only to frames that carry
georeferencing; plain arrays fall back to shape-equality checks. The
output carrier follows the first frame.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from jaxtyping import Bool, Float, Int, Num, Shaped
from pipekit import Operator

from geotoolz._src.wrap import wrap_like
from geotoolz.indices._src.bands import BandRef, resolve_band


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
    shape equality.

    Args:
        nan_policy: ``"ignore"`` skips NaNs with ``np.nanmedian``;
            ``"propagate"`` uses ``np.median``.
        return_count: When true, also return a carrier with the number
            of non-NaN contributors per output pixel.
    """

    def __init__(
        self, *, nan_policy: NanPolicy = "ignore", return_count: bool = False
    ) -> None:
        self.nan_policy = _validate_nan_policy(nan_policy)
        self.return_count = return_count

    def _apply(
        self, frames: Sequence[GeoTensor | np.ndarray]
    ) -> GeoTensor | np.ndarray | tuple[GeoTensor | np.ndarray, GeoTensor | np.ndarray]:
        base, stack = _stack_frames(frames)
        values = (
            np.nanmedian(stack, axis=0)
            if self.nan_policy == "ignore"
            else np.median(stack, axis=0)
        )
        out = wrap_like(base, values)
        if not self.return_count:
            return out
        count = np.sum(~np.isnan(stack), axis=0).astype(np.int64)
        return out, wrap_like(base, count)

    def get_config(self) -> dict[str, Any]:
        return {"nan_policy": self.nan_policy, "return_count": self.return_count}


class MaxNDVIComposite(Operator):
    """Pick the frame with maximum NDVI per pixel and return its band values.

    Inputs must be multi-band (``(C, H, W)``); 2-D GeoTensors raise because
    NDVI needs distinct red and NIR bands. The output dtype is the input
    dtype when all-invalid pixels can be represented in it (e.g. integer
    masks via ``fill_value_default``); for float inputs invalid pixels are
    set to NaN.

    The per-pixel math is metadata-independent, so plain ``np.ndarray``
    frames are accepted when ``red`` / ``nir`` are integer indices; named
    band references need GeoTensor ``attrs`` and raise ``TypeError`` for
    plain arrays.

    Args:
        red: Red band reference — integer band index or a band name
            resolved against the first frame's attrs.
        nir: NIR band reference, same conventions as ``red``.
        return_index: When true, also return a carrier holding the
            selected frame index per pixel (``int64``).
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
        ndvi = (nir - red) / (nir + red + self.eps)
        scores = np.where(np.isnan(ndvi), -np.inf, ndvi)
        index = np.argmax(scores, axis=0)
        values = _take_by_spatial_index(stack, index)
        all_invalid = np.all(~np.isfinite(scores), axis=0)
        if np.any(all_invalid):
            if np.issubdtype(values.dtype, np.floating):
                values[..., all_invalid] = np.nan
            else:
                # Integer / unsigned inputs can't carry NaN. Fall back to the
                # input's fill_value_default so the dtype is preserved.
                fill = getattr(base, "fill_value_default", None)
                if fill is None:
                    raise ValueError(
                        "MaxNDVIComposite: all NDVI scores are invalid for "
                        "some pixels and the integer-dtype input carries no "
                        "fill_value_default to mark them; use a GeoTensor "
                        "with fill_value_default or float frames."
                    )
                values[..., all_invalid] = fill
        out = wrap_like(base, values)
        if not self.return_index:
            return out
        return out, wrap_like(base, index.astype(np.int64))

    def get_config(self) -> dict[str, Any]:
        return {
            "red": self.red,
            "nir": self.nir,
            "return_index": self.return_index,
            "eps": self.eps,
        }


class CloudFreeComposite(Operator):
    """Per-pixel mean over frames where the cloud mask is false.

    Consumes ``(frame, cloud_mask)`` pairs; masks may have spatial shape
    ``(H, W)``, ``(1, H, W)``, or match the frame shape exactly. Pixels
    with fewer than ``min_valid`` clear contributors come out as NaN.
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

    def _apply(
        self, pairs: Sequence[tuple[GeoTensor | np.ndarray, Any]]
    ) -> GeoTensor | np.ndarray | tuple[GeoTensor | np.ndarray, GeoTensor | np.ndarray]:
        base, stack, cloudy = _require_pairs(pairs)
        clear = ~cloudy
        valid = clear & ~np.isnan(stack) if self.nan_policy == "ignore" else clear
        count = np.sum(valid, axis=0)
        total = np.sum(np.where(valid, stack, 0), axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            values = total / count
        values = np.where(count >= self.min_valid, values, np.nan)
        out = wrap_like(base, _as_float_for_nan(values))
        if not self.return_count:
            return out
        return out, wrap_like(base, count.astype(np.int64))

    def get_config(self) -> dict[str, Any]:
        return {
            "nan_policy": self.nan_policy,
            "min_valid": self.min_valid,
            "return_count": self.return_count,
        }


class BAPComposite(Operator):
    """Best Available Pixel compositing from quality-score metadata.

    Metadata may provide precomputed ``*_score`` arrays, or raw ``view_angle``,
    ``doy``, ``cloud_distance``, and ``opacity`` values used to build simple
    scores. Each value may be a scalar or a per-pixel array. Frames may be
    GeoTensors or plain ``np.ndarray`` maps (the scores live in the metadata
    dicts, not in geo-metadata).

    Args:
        target_doy: Day-of-year the recency score is anchored to.
        w_view_angle: Weight of the view-angle score.
        w_recency: Weight of the recency score.
        w_cloud_distance: Weight of the cloud-distance score.
        w_opacity: Weight of the opacity score.
        return_score: When true, also return a carrier holding the
            winning per-pixel score (``float32``).
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
        index = np.argmax(score_stack, axis=0)
        out = wrap_like(base, _take_by_spatial_index(stack, index))
        if not self.return_score:
            return out
        best_score = np.take_along_axis(score_stack, index[None, ...], axis=0)[0]
        return out, wrap_like(base, best_score)

    def get_config(self) -> dict[str, Any]:
        return {
            "target_doy": self.target_doy,
            "w_view_angle": self.w_view_angle,
            "w_recency": self.w_recency,
            "w_cloud_distance": self.w_cloud_distance,
            "w_opacity": self.w_opacity,
            "return_score": self.return_score,
        }


class MinCloudComposite(Operator):
    """Pick each pixel from the scene with the lowest *global* cloud coverage.

    For every pixel, the selected frame is the one with the smallest
    scene-wide cloud fraction among those where the pixel is clear. Pixels
    cloudy in every frame fall back to the globally least-cloudy frame so
    the output is a complete composite.

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
    """

    def __init__(self, *, return_count: bool = False) -> None:
        self.return_count = return_count

    def _apply(
        self, pairs: Sequence[tuple[GeoTensor | np.ndarray, Any]]
    ) -> GeoTensor | np.ndarray | tuple[GeoTensor | np.ndarray, GeoTensor | np.ndarray]:
        base, stack, cloudy = _require_pairs(pairs)
        clear = ~cloudy
        spatial_clear = clear.reshape((clear.shape[0], -1))
        cloud_coverage = 1.0 - spatial_clear.mean(axis=1)
        costs = np.where(
            clear,
            cloud_coverage.reshape((-1, *([1] * (clear.ndim - 1)))),
            np.inf,
        )
        fallback = int(np.argmin(cloud_coverage))
        index = np.argmin(costs, axis=0)
        all_cloudy = ~np.any(clear, axis=0)
        index = np.where(all_cloudy, fallback, index)
        out = wrap_like(base, _take_by_spatial_index(stack, index))
        if not self.return_count:
            return out
        count = np.sum(clear, axis=0).astype(np.int64)
        return out, wrap_like(base, count)

    def get_config(self) -> dict[str, Any]:
        return {"return_count": self.return_count}


__all__ = [
    "BAPComposite",
    "CloudFreeComposite",
    "MaxNDVIComposite",
    "MedianComposite",
    "MinCloudComposite",
]
