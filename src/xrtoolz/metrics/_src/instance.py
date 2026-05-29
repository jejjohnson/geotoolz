"""Instance-segmentation evaluation metrics.

Greedy score-sorted matching of predicted instance masks to reference
instance masks via mask-IoU, then VOC-style 11-point Average Precision
plus instance-level precision / recall / F1. Adapted from
Pérez Carrasco et al. (2026) — ``metrics_instance_segmentation``
(``compute_overlaps_masks``, ``compute_matches``,
``compute_ap_with_df_safe``).

Inputs are mask stacks shaped ``(N, H, W)``. Predicted and reference
stacks may carry different ``N`` (no constraint that a prediction exist
for every reference, or vice versa). The matcher accepts an optional
``scores`` array of length ``N_pred``; when omitted, predictions are
ranked by mask area (largest first).

These operators are **additive** — they do not overlap the
``ProbabilityOfDetection`` / ``IntersectionOverUnion`` / etc. name
reservations in :mod:`xrtoolz.metrics._src.object`, which remain
``NotImplementedError`` stubs pending the V5 epic and a future detector /
matcher framework.
"""

from __future__ import annotations

from typing import Any

import einx
import numpy as np
import xarray as xr

from xrtoolz._operator import Operator


def _as_3d_bool(masks: xr.DataArray | np.ndarray, *, name: str) -> np.ndarray:
    """Coerce a carrier to a 3-D boolean ``(N, H, W)`` ndarray."""
    arr = np.asarray(masks)
    if arr.ndim != 3:
        raise ValueError(
            f"{name} must be a (N, H, W) mask stack, got shape {arr.shape}"
        )
    return arr.astype(bool, copy=False)


def mask_iou_matrix(
    pred_masks: xr.DataArray | np.ndarray,
    ref_masks: xr.DataArray | np.ndarray,
) -> np.ndarray:
    """Pairwise mask-IoU matrix between two ``(N, H, W)`` stacks.

    Returns a dense ``(N_pred, N_ref)`` float matrix of mask
    intersection-over-union values in ``[0, 1]``. Empty masks contribute
    zero IoU.
    """
    pred = _as_3d_bool(pred_masks, name="pred_masks")
    ref = _as_3d_bool(ref_masks, name="ref_masks")
    if pred.shape[1:] != ref.shape[1:]:
        raise ValueError(
            f"pred/ref spatial shapes do not match: {pred.shape[1:]} vs {ref.shape[1:]}"
        )
    n_p, n_r = pred.shape[0], ref.shape[0]
    if n_p == 0 or n_r == 0:
        return np.zeros((n_p, n_r), dtype=np.float64)
    flat_pred = einx.id("n h w -> n (h w)", pred).astype(np.int64)
    flat_ref = einx.id("n h w -> n (h w)", ref).astype(np.int64)
    # ``inter[i, j] = sum(pred_i AND ref_j)`` computed via boolean matmul
    # over int64 (named: contract the flattened-pixel axis). For typical
    # N ~ 1e2-1e3 and HW ~ 1e6 this is the cheap path.
    inter = einx.dot("pred pix, ref pix -> pred ref", flat_pred, flat_ref)
    area_p = flat_pred.sum(axis=1, keepdims=True)
    area_r = flat_ref.sum(axis=1, keepdims=True).T
    union = area_p + area_r - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou.astype(np.float64, copy=False)


def match_instances(
    pred_masks: xr.DataArray | np.ndarray,
    ref_masks: xr.DataArray | np.ndarray,
    *,
    scores: np.ndarray | None = None,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Greedy score-sorted matcher.

    Predictions are sorted by score (descending; ties broken by input
    order). Each prediction takes the best unmatched reference whose
    mask-IoU clears ``iou_threshold``.

    Returns a dict with:

    - ``pred_match``: int array of length ``N_pred``; the matched
      reference index or ``-1``.
    - ``ref_match``: int array of length ``N_ref``; the matched
      prediction index or ``-1``.
    - ``ious``: full IoU matrix from :func:`mask_iou_matrix`.
    - ``tp`` / ``fp`` / ``fn``: scalar counts.
    """
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    ious = mask_iou_matrix(pred_masks, ref_masks)
    n_p, n_r = ious.shape

    if scores is None:
        # Rank by area so the matcher is well-defined without external
        # scores; the paper uses detector confidence here.
        pred = _as_3d_bool(pred_masks, name="pred_masks")
        rank_key = -pred.reshape(n_p, -1).sum(axis=1).astype(np.float64)
    else:
        scores_arr = np.asarray(scores, dtype=float)
        if scores_arr.shape != (n_p,):
            raise ValueError(
                f"scores length {scores_arr.shape} does not match pred count ({n_p})"
            )
        rank_key = -scores_arr
    order = np.argsort(rank_key, kind="stable")

    pred_match = -np.ones(n_p, dtype=np.int64)
    ref_match = -np.ones(n_r, dtype=np.int64)
    for idx in order:
        i = int(idx)
        # Best reference for this prediction; the paper sorts references
        # by IoU desc and walks until it hits one that clears the gate
        # and is unmatched.
        cand = np.argsort(-ious[i], kind="stable")
        for jdx in cand:
            j = int(jdx)
            if ious[i, j] < iou_threshold:
                break
            if ref_match[j] != -1:
                continue
            pred_match[i] = j
            ref_match[j] = i
            break

    tp = int((pred_match >= 0).sum())
    fp = int((pred_match < 0).sum())
    fn = int((ref_match < 0).sum())
    return {
        "pred_match": pred_match,
        "ref_match": ref_match,
        "ious": ious,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def instance_f1_at_iou(
    pred_masks: xr.DataArray | np.ndarray,
    ref_masks: xr.DataArray | np.ndarray,
    *,
    scores: np.ndarray | None = None,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Precision / recall / F1 from instance-level greedy matching at a
    single IoU threshold."""
    match = match_instances(
        pred_masks, ref_masks, scores=scores, iou_threshold=iou_threshold
    )
    tp, fp, fn = match["tp"], match["fp"], match["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def average_precision_matched(
    pred_masks: xr.DataArray | np.ndarray,
    ref_masks: xr.DataArray | np.ndarray,
    *,
    scores: np.ndarray | None = None,
    iou_threshold: float = 0.5,
) -> float:
    """VOC-style Average Precision at a single IoU threshold.

    Mirrors Pérez Carrasco et al. (2026), ``compute_ap_with_df_safe``:
    walk predictions in score-descending order, accumulate
    ``precision[k]`` and ``recall[k]``, enforce monotone-decreasing
    precision, and integrate the precision/recall curve. Returns ``0.0``
    when there are no references; ``0.0`` when there are no predictions.
    """
    pred = _as_3d_bool(pred_masks, name="pred_masks")
    ref = _as_3d_bool(ref_masks, name="ref_masks")
    n_p, n_r = pred.shape[0], ref.shape[0]
    if n_r == 0 or n_p == 0:
        return 0.0
    match = match_instances(pred, ref, scores=scores, iou_threshold=iou_threshold)
    pred_match = match["pred_match"]

    if scores is None:
        rank_key = -pred.reshape(n_p, -1).sum(axis=1).astype(np.float64)
    else:
        rank_key = -np.asarray(scores, dtype=float)
    order = np.argsort(rank_key, kind="stable")
    ordered_hits = (pred_match[order] >= 0).astype(np.float64)
    cum_tp = np.cumsum(ordered_hits)
    precisions = cum_tp / np.arange(1, n_p + 1, dtype=np.float64)
    recalls = cum_tp / float(n_r)

    # Pad endpoints and enforce monotonicity, per VOC.
    precisions = np.concatenate([[0.0], precisions, [0.0]])
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    changes = np.where(recalls[1:] != recalls[:-1])[0] + 1
    return float(
        np.sum((recalls[changes] - recalls[changes - 1]) * precisions[changes])
    )


# ---------- Operators ----------------------------------------------------


class MaskIoU(Operator):
    """Pairwise mask-IoU matrix between two ``(N, H, W)`` stacks."""

    def _apply(
        self,
        pred_masks: xr.DataArray | np.ndarray,
        ref_masks: xr.DataArray | np.ndarray,
    ) -> np.ndarray:
        return mask_iou_matrix(pred_masks, ref_masks)

    def get_config(self) -> dict[str, Any]:
        return {}


class InstanceMatcher(Operator):
    """Greedy IoU matching between predicted and reference mask stacks.

    Args:
        iou_threshold: Minimum IoU for a (pred, ref) pair to count as a
            match. Defaults to ``0.5`` (Pascal-VOC convention; the paper
            uses ``0.1`` for the much sparser MethaneSAT plume regime).
        scores: Optional length-``N_pred`` array of per-prediction
            scores. When ``None``, predictions are ranked by area.
    """

    def __init__(
        self,
        *,
        iou_threshold: float = 0.5,
        scores: np.ndarray | None = None,
    ) -> None:
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in (0, 1]")
        self.iou_threshold = float(iou_threshold)
        self.scores = None if scores is None else np.asarray(scores, dtype=float)

    def _apply(
        self,
        pred_masks: xr.DataArray | np.ndarray,
        ref_masks: xr.DataArray | np.ndarray,
    ) -> dict[str, Any]:
        return match_instances(
            pred_masks,
            ref_masks,
            scores=self.scores,
            iou_threshold=self.iou_threshold,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "iou_threshold": self.iou_threshold,
            "scores": None if self.scores is None else self.scores.tolist(),
        }


class InstanceF1AtIoU(Operator):
    """Instance precision / recall / F1 at a single IoU threshold."""

    def __init__(
        self,
        *,
        iou_threshold: float = 0.5,
        scores: np.ndarray | None = None,
    ) -> None:
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in (0, 1]")
        self.iou_threshold = float(iou_threshold)
        self.scores = None if scores is None else np.asarray(scores, dtype=float)

    def _apply(
        self,
        pred_masks: xr.DataArray | np.ndarray,
        ref_masks: xr.DataArray | np.ndarray,
    ) -> dict[str, float]:
        return instance_f1_at_iou(
            pred_masks,
            ref_masks,
            scores=self.scores,
            iou_threshold=self.iou_threshold,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "iou_threshold": self.iou_threshold,
            "scores": None if self.scores is None else self.scores.tolist(),
        }


class AveragePrecisionMatched(Operator):
    """VOC-style Average Precision at a single IoU threshold."""

    def __init__(
        self,
        *,
        iou_threshold: float = 0.5,
        scores: np.ndarray | None = None,
    ) -> None:
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in (0, 1]")
        self.iou_threshold = float(iou_threshold)
        self.scores = None if scores is None else np.asarray(scores, dtype=float)

    def _apply(
        self,
        pred_masks: xr.DataArray | np.ndarray,
        ref_masks: xr.DataArray | np.ndarray,
    ) -> float:
        return average_precision_matched(
            pred_masks,
            ref_masks,
            scores=self.scores,
            iou_threshold=self.iou_threshold,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "iou_threshold": self.iou_threshold,
            "scores": None if self.scores is None else self.scores.tolist(),
        }


__all__ = [
    "AveragePrecisionMatched",
    "InstanceF1AtIoU",
    "InstanceMatcher",
    "MaskIoU",
    "average_precision_matched",
    "instance_f1_at_iou",
    "mask_iou_matrix",
    "match_instances",
]
