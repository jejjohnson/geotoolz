"""Matched-stack operators — multi-source fusion of co-registered tensors.

Sibling to the existing composites (``MaxNDVIComposite``,
``CloudFreeComposite``, etc.) but for a *matched tuple* of tensors
from different sources rather than a temporal stack of one sensor.

* `StackMatched` — concatenate N aligned tensors along the band axis.
* `BlendMatched` — weighted mean across N aligned tensors, with
  optional inverse-variance weighting when uncertainty maps are
  present.

Inputs are assumed already co-registered (typically by a
``geotoolz.geom.coregister`` operator before this op).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pipekit import Operator


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor


def _grid_matches(a: GeoTensor, b: GeoTensor) -> bool:
    """Exact grid equality — affine drift on a fused stack is a real bug source."""
    h_w_a = a.shape[-2:]
    h_w_b = b.shape[-2:]
    return h_w_a == h_w_b and a.transform == b.transform and a.crs == b.crs


def _normalize_to_sequence(
    tensors: Sequence[GeoTensor] | Mapping[str, GeoTensor],
    order: list[str] | None,
) -> tuple[list[GeoTensor], list[str] | None]:
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


def _as_band_first(values: np.ndarray) -> np.ndarray:
    """Promote ``(H, W)`` to ``(1, H, W)``; leave ``(C, H, W)`` alone."""
    if values.ndim == 2:
        return values[None, :, :]
    if values.ndim == 3:
        return values
    raise ValueError(
        "StackMatched expects 2-D (H, W) or 3-D (C, H, W) tensors; "
        f"got ndim={values.ndim}."
    )


class StackMatched(Operator):
    """Concatenate aligned tensors along the band axis.

    Inputs are either a `Sequence[GeoTensor]` or a
    ``Mapping[str, GeoTensor]`` — typical when called on
    `MatchedPatch.members`. All inputs must share spatial shape,
    transform, and CRS; the per-tensor band count may differ.

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
        Band-name metadata propagation and NaN-fill padding on grid
        mismatch are tracked for a future revision; today the
        operator requires strict grid equality and emits an unnamed
        band stack. Pre-coregister with
        ``geotoolz.geom.coregister.RasterToRasterLike`` if the
        inputs aren't already on the same grid.
    """

    def __init__(
        self,
        *,
        order: list[str] | None = None,
    ) -> None:
        self.order = list(order) if order is not None else None

    def get_config(self) -> dict[str, Any]:
        return {"order": self.order}

    def __call__(
        self,
        tensors: Sequence[GeoTensor] | Mapping[str, GeoTensor],
    ) -> GeoTensor:
        seq, _names = _normalize_to_sequence(tensors, self.order)
        if not seq:
            raise ValueError("StackMatched requires at least one input tensor.")

        # Validate grids exactly — silent affine drift on a per-pixel
        # fused stack is a real bug source; we'd rather fail loudly
        # than emit subtly misregistered output.
        base = seq[0]
        for idx, frame in enumerate(seq[1:], start=1):
            if not _grid_matches(base, frame):
                raise ValueError(
                    "StackMatched inputs must share spatial shape, "
                    "transform, and CRS; "
                    f"input 0 has shape {base.shape[-2:]}, "
                    f"transform {base.transform!r}; "
                    f"input {idx} has shape {frame.shape[-2:]}, "
                    f"transform {frame.transform!r}."
                )

        arrays = [_as_band_first(np.asarray(t)) for t in seq]
        stacked = np.concatenate(arrays, axis=0)
        return base.array_as_geotensor(stacked)


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

    All inputs must share spatial shape, transform, and CRS. The
    band axis must also be uniform (use `StackMatched` if you want
    cross-source band concatenation; `BlendMatched` is the per-pixel
    averaging counterpart).

    Args:
        method: One of ``"mean"`` / ``"weighted_mean"`` / ``"ivw"``.
        weights: Per-source scalar weights for ``"weighted_mean"``;
            ignored otherwise. Length must equal the number of input
            tensors at call time.
        nan_policy: ``"ignore"`` (default) or ``"propagate"``.
    """

    def __init__(
        self,
        *,
        method: Literal["mean", "weighted_mean", "ivw"] = "mean",
        weights: list[float] | None = None,
        nan_policy: Literal["ignore", "propagate"] = "ignore",
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

    def get_config(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "weights": self.weights,
            "nan_policy": self.nan_policy,
        }

    def __call__(
        self,
        tensors: Sequence[GeoTensor] | Mapping[str, GeoTensor],
        variances: Sequence[np.ndarray] | None = None,
    ) -> GeoTensor:
        seq, _names = _normalize_to_sequence(tensors, None)
        if not seq:
            raise ValueError("BlendMatched requires at least one input tensor.")

        # Strict grid + band-shape validation. BlendMatched is a
        # per-pixel reduction across inputs, so any shape difference
        # would mean we're averaging different physical quantities.
        base = seq[0]
        for idx, frame in enumerate(seq[1:], start=1):
            if not _grid_matches(base, frame):
                raise ValueError(
                    "BlendMatched inputs must share spatial shape, "
                    "transform, and CRS; "
                    f"input 0 has shape {base.shape[-2:]}, "
                    f"transform {base.transform!r}; "
                    f"input {idx} has shape {frame.shape[-2:]}, "
                    f"transform {frame.transform!r}."
                )
            if frame.shape != base.shape:
                raise ValueError(
                    "BlendMatched inputs must share full shape (including "
                    f"band axis); input 0 has shape {base.shape}, "
                    f"input {idx} has shape {frame.shape}."
                )

        # Stack along a new "source" axis at position 0. Shape is now
        # (N, ...spatial...) for 2-D or (N, C, H, W) for 3-D.
        stack = np.stack([np.asarray(t).astype(np.float64) for t in seq], axis=0)

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
            # Each variance can be either a full-shape array or a
            # spatial-only array (H, W) that broadcasts to (C, H, W).
            # Clamp to a small epsilon to avoid /0.
            var_stack = np.stack(
                [np.broadcast_to(np.asarray(v), base.shape) for v in var_list],
                axis=0,
            ).astype(np.float64)
            w = 1.0 / np.maximum(var_stack, 1e-12)
        elif self.method == "weighted_mean":
            assert self.weights is not None  # guarded in __init__
            if len(self.weights) != len(seq):
                raise ValueError(
                    f"BlendMatched(weights=...): got {len(self.weights)} "
                    f"weights for {len(seq)} input tensors."
                )
            w_arr = np.asarray(self.weights, dtype=np.float64)
            # Broadcast (N,) → (N, 1, ..., 1) so the per-source scalar
            # weight multiplies through every spatial / band pixel.
            w_shape = (len(seq),) + (1,) * (stack.ndim - 1)
            w = (
                np.broadcast_to(w_arr.reshape(w_shape), stack.shape)
                .astype(np.float64)
                .copy()
            )
        else:  # "mean"
            w = np.ones_like(stack, dtype=np.float64)

        if self.nan_policy == "propagate":
            # Any NaN in any source ↦ NaN output at that pixel.
            nan_mask = np.isnan(stack).any(axis=0)
            num = (stack * w).sum(axis=0)
            den = w.sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                result = np.where(den > 0, num / den, np.nan)
            result = np.where(nan_mask, np.nan, result)
        else:  # "ignore"
            # Zero the weight wherever the value is NaN, then drop
            # NaNs from the numerator. The denominator is the sum of
            # surviving weights — a pixel with all-NaN inputs ends
            # up with den == 0 and the np.where guard puts a NaN
            # in the output.
            valid = ~np.isnan(stack)
            safe_values = np.where(valid, stack, 0.0)
            safe_weights = np.where(valid, w, 0.0)
            num = (safe_values * safe_weights).sum(axis=0)
            den = safe_weights.sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                result = np.where(den > 0, num / den, np.nan)

        return base.array_as_geotensor(result)
