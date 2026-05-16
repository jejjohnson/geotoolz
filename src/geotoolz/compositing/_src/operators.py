"""Carrier-aware compositing operators for co-registered GeoTensors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from pipekit import Operator
from geotoolz.indices._src.bands import BandRef, resolve_band


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor

NanPolicy = Literal["ignore", "propagate"]
CompositeState = dict[str, list[Any] | Any]


def _validate_nan_policy(nan_policy: str) -> NanPolicy:
    if nan_policy not in {"ignore", "propagate"}:
        raise ValueError("nan_policy must be 'ignore' or 'propagate'.")
    return nan_policy  # type: ignore[return-value]


def _grid_matches(a: GeoTensor, b: GeoTensor) -> bool:
    return (
        a.shape == b.shape
        and np.allclose(tuple(a.transform), tuple(b.transform))
        and a.crs == b.crs
    )


def _require_frames(frames: Sequence[GeoTensor]) -> GeoTensor:
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


def _stack_frames(frames: Sequence[GeoTensor]) -> tuple[GeoTensor, np.ndarray]:
    base = _require_frames(frames)
    return base, np.stack([np.asarray(frame) for frame in frames], axis=0)


def _as_geotensor_like(base: GeoTensor, values: np.ndarray) -> GeoTensor:
    return base.array_as_geotensor(values)


def _take_by_spatial_index(stack: np.ndarray, index: np.ndarray) -> np.ndarray:
    indexer = np.broadcast_to(index, stack.shape[1:]).reshape((1, *stack.shape[1:]))
    return np.take_along_axis(stack, indexer, axis=0)[0]


def _mask_array(mask: Any, target_shape: tuple[int, ...]) -> np.ndarray:
    mask_arr = np.asarray(mask, dtype=bool)
    spatial_shape = target_shape[-2:]
    if mask_arr.shape == spatial_shape:
        return np.broadcast_to(mask_arr, target_shape)
    if mask_arr.shape == (1, *spatial_shape):
        return np.broadcast_to(mask_arr, target_shape)
    if mask_arr.shape == target_shape:
        return mask_arr
    raise ValueError(
        "Cloud masks must have spatial shape (H, W), (1, H, W), or match the "
        "GeoTensor shape; got "
        f"{mask_arr.shape}, expected compatible with {target_shape}."
    )


def _require_pairs(
    pairs: Sequence[tuple[GeoTensor, Any]],
) -> tuple[GeoTensor, np.ndarray, np.ndarray]:
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


def _score_array(value: Any, spatial_shape: tuple[int, int]) -> np.ndarray:
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


def _normalise_positive(stack: np.ndarray) -> np.ndarray:
    max_value = np.nanmax(stack)
    if not np.isfinite(max_value) or max_value <= 0:
        return np.zeros_like(stack, dtype=np.float32)
    return np.asarray(stack / max_value, dtype=np.float32)


class MedianComposite(Operator):
    """Per-pixel median across a stack of co-registered GeoTensors.

    Args:
        nan_policy: ``"ignore"`` skips NaNs with ``np.nanmedian``;
            ``"propagate"`` uses ``np.median``.
        return_count: When true, also return a GeoTensor with the number
            of non-NaN contributors per output pixel.
    """

    def __init__(
        self, *, nan_policy: NanPolicy = "ignore", return_count: bool = False
    ) -> None:
        self.nan_policy = _validate_nan_policy(nan_policy)
        self.return_count = return_count

    def _apply(
        self, frames: Sequence[GeoTensor]
    ) -> GeoTensor | tuple[GeoTensor, GeoTensor]:
        base, stack = _stack_frames(frames)
        values = (
            np.nanmedian(stack, axis=0)
            if self.nan_policy == "ignore"
            else np.median(stack, axis=0)
        )
        out = _as_geotensor_like(base, values)
        if not self.return_count:
            return out
        count = np.sum(~np.isnan(stack), axis=0).astype(np.int64)
        return out, _as_geotensor_like(base, count)

    def partial_composite(
        self, state: CompositeState | None, new_frame: GeoTensor
    ) -> CompositeState:
        frames = [] if state is None else list(state["frames"])
        frames.append(new_frame)
        return {"frames": frames, "composite": self(frames)}

    def get_config(self) -> dict[str, Any]:
        return {"nan_policy": self.nan_policy, "return_count": self.return_count}


class MaxNDVIComposite(Operator):
    """Pick the frame with maximum NDVI per pixel and return its band values."""

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
        self, frames: Sequence[GeoTensor]
    ) -> GeoTensor | tuple[GeoTensor, GeoTensor]:
        base, stack = _stack_frames(frames)
        red_idx = resolve_band(base, self.red)
        nir_idx = resolve_band(base, self.nir)
        red = stack[:, red_idx, ...].astype(np.float32, copy=False)
        nir = stack[:, nir_idx, ...].astype(np.float32, copy=False)
        ndvi = (nir - red) / (nir + red + self.eps)
        scores = np.where(np.isnan(ndvi), -np.inf, ndvi)
        index = np.argmax(scores, axis=0)
        values = _take_by_spatial_index(stack, index)
        values = values.astype(np.result_type(values.dtype, np.float32), copy=False)
        values[..., np.all(~np.isfinite(scores), axis=0)] = np.nan
        out = _as_geotensor_like(base, values)
        if not self.return_index:
            return out
        return out, _as_geotensor_like(base, index.astype(np.int64))

    def partial_composite(
        self, state: CompositeState | None, new_frame: GeoTensor
    ) -> CompositeState:
        frames = [] if state is None else list(state["frames"])
        frames.append(new_frame)
        return {"frames": frames, "composite": self(frames)}

    def get_config(self) -> dict[str, Any]:
        return {
            "red": self.red,
            "nir": self.nir,
            "return_index": self.return_index,
            "eps": self.eps,
        }


class CloudFreeComposite(Operator):
    """Per-pixel mean over frames where the cloud mask is false."""

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
        self, pairs: Sequence[tuple[GeoTensor, Any]]
    ) -> GeoTensor | tuple[GeoTensor, GeoTensor]:
        base, stack, cloudy = _require_pairs(pairs)
        clear = ~cloudy
        valid = clear & ~np.isnan(stack) if self.nan_policy == "ignore" else clear
        count = np.sum(valid, axis=0)
        total = np.sum(np.where(valid, stack, 0), axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            values = total / count
        values = np.where(count >= self.min_valid, values, np.nan)
        out = _as_geotensor_like(
            base, values.astype(np.result_type(stack.dtype, np.float32), copy=False)
        )
        if not self.return_count:
            return out
        return out, _as_geotensor_like(base, count.astype(np.int64))

    def partial_composite(
        self, state: CompositeState | None, new_pair: tuple[GeoTensor, Any]
    ) -> CompositeState:
        pairs = [] if state is None else list(state["frames"])
        pairs.append(new_pair)
        return {"frames": pairs, "composite": self(pairs)}

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
    scores. Each value may be a scalar or a per-pixel array.
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
        self, pairs: Sequence[tuple[GeoTensor, Mapping[str, Any]]]
    ) -> GeoTensor | tuple[GeoTensor, GeoTensor]:
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

        cloud_distance_stack = np.stack(cloud_distance_scores, axis=0)
        if all(raw_cloud_distance):
            cloud_distance_stack = _normalise_positive(cloud_distance_stack)
        score_stack = (
            self.w_view_angle * np.stack(view_scores, axis=0)
            + self.w_recency * np.stack(recency_scores, axis=0)
            + self.w_cloud_distance * cloud_distance_stack
            + self.w_opacity * np.stack(opacity_scores, axis=0)
        ).astype(np.float32, copy=False)
        index = np.argmax(score_stack, axis=0)
        out = _as_geotensor_like(base, _take_by_spatial_index(stack, index))
        if not self.return_score:
            return out
        best_score = np.take_along_axis(score_stack, index[None, ...], axis=0)[0]
        return out, _as_geotensor_like(base, best_score)

    def partial_composite(
        self,
        state: CompositeState | None,
        new_pair: tuple[GeoTensor, Mapping[str, Any]],
    ) -> CompositeState:
        pairs = [] if state is None else list(state["frames"])
        pairs.append(new_pair)
        return {"frames": pairs, "composite": self(pairs)}

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
    """Pick pixels from the least-cloudy clear frame available."""

    def __init__(self, *, return_count: bool = False) -> None:
        self.return_count = return_count

    def _apply(
        self, pairs: Sequence[tuple[GeoTensor, Any]]
    ) -> GeoTensor | tuple[GeoTensor, GeoTensor]:
        base, stack, cloudy = _require_pairs(pairs)
        clear = ~cloudy
        spatial_clear = clear.reshape((clear.shape[0], -1))
        coverage = 1.0 - spatial_clear.mean(axis=1)
        costs = np.where(
            clear, coverage.reshape((-1, *([1] * (clear.ndim - 1)))), np.inf
        )
        fallback = int(np.argmin(coverage))
        index = np.argmin(costs, axis=0)
        all_cloudy = ~np.any(clear, axis=0)
        index = np.where(all_cloudy, fallback, index)
        out = _as_geotensor_like(base, _take_by_spatial_index(stack, index))
        if not self.return_count:
            return out
        count = np.sum(clear, axis=0).astype(np.int64)
        return out, _as_geotensor_like(base, count)

    def partial_composite(
        self, state: CompositeState | None, new_pair: tuple[GeoTensor, Any]
    ) -> CompositeState:
        pairs = [] if state is None else list(state["frames"])
        pairs.append(new_pair)
        return {"frames": pairs, "composite": self(pairs)}

    def get_config(self) -> dict[str, Any]:
        return {"return_count": self.return_count}


__all__ = [
    "BAPComposite",
    "CloudFreeComposite",
    "MaxNDVIComposite",
    "MedianComposite",
    "MinCloudComposite",
]
