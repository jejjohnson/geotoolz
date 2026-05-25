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

    Args:
        method: ``"mean"`` (equal weights), ``"weighted_mean"`` (use
            ``weights``), or ``"ivw"`` (inverse-variance weighting,
            requires per-source variance maps via ``variances``).
        weights: Per-source scalar weights. Ignored unless
            ``method == "weighted_mean"``.
        nan_policy: ``"ignore"`` (treat NaN as missing) or
            ``"propagate"`` (any NaN poisons the output pixel).
    """

    def __init__(
        self,
        *,
        method: Literal["mean", "weighted_mean", "ivw"] = "mean",
        weights: list[float] | None = None,
        nan_policy: Literal["ignore", "propagate"] = "ignore",
    ) -> None:
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
        tensors: Sequence[GeoTensor],
        variances: Sequence[np.ndarray] | None = None,
    ) -> GeoTensor:
        raise NotImplementedError("Phase 3 PR — see design §5.")
