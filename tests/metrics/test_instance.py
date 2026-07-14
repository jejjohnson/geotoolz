"""Tests for instance-segmentation evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from xrtoolz.metrics import (
    AveragePrecisionMatched,
    InstanceF1AtIoU,
    InstanceMatcher,
    MaskIoU,
    average_precision_matched,
    instance_f1_at_iou,
    mask_iou_matrix,
    match_instances,
)


def _stack_box(boxes, shape=(20, 20)) -> np.ndarray:
    """Build a (N, H, W) boolean mask stack from ``(y0, y1, x0, x1)`` boxes."""
    out = np.zeros((len(boxes), *shape), dtype=bool)
    for i, (y0, y1, x0, x1) in enumerate(boxes):
        out[i, y0:y1, x0:x1] = True
    return out


def test_mask_iou_matrix_matches_hand_computed_overlap() -> None:
    pred = _stack_box([(0, 5, 0, 5), (0, 5, 5, 10)])
    ref = _stack_box([(0, 5, 0, 5), (10, 15, 10, 15)])

    ious = mask_iou_matrix(pred, ref)
    assert ious.shape == (2, 2)
    assert ious[0, 0] == pytest.approx(1.0)  # exact match
    assert ious[1, 1] == pytest.approx(0.0)
    assert ious[0, 1] == pytest.approx(0.0)
    assert ious[1, 0] == pytest.approx(0.0)


def test_mask_iou_matrix_rejects_non_3d_input() -> None:
    with pytest.raises(ValueError, match="N, H, W"):
        mask_iou_matrix(np.ones((4, 4), dtype=bool), np.ones((1, 4, 4), dtype=bool))


def test_mask_iou_matrix_rejects_shape_mismatch() -> None:
    pred = _stack_box([(0, 5, 0, 5)], shape=(10, 10))
    ref = _stack_box([(0, 5, 0, 5)], shape=(20, 20))
    with pytest.raises(ValueError, match="spatial shapes"):
        mask_iou_matrix(pred, ref)


def test_mask_iou_matrix_empty_stacks_returns_zero_shape() -> None:
    empty = np.zeros((0, 5, 5), dtype=bool)
    ref = _stack_box([(0, 2, 0, 2)], shape=(5, 5))
    assert mask_iou_matrix(empty, ref).shape == (0, 1)
    assert mask_iou_matrix(ref, empty).shape == (1, 0)


def test_match_instances_greedy_matches_exact_overlaps() -> None:
    pred = _stack_box([(0, 5, 0, 5), (10, 15, 10, 15)])
    ref = _stack_box([(10, 15, 10, 15), (0, 5, 0, 5)])

    result = match_instances(pred, ref, iou_threshold=0.5)
    assert result["tp"] == 2
    assert result["fp"] == 0
    assert result["fn"] == 0
    # Each prediction should be matched to the spatially identical reference.
    assert int(result["pred_match"][0]) == 1
    assert int(result["pred_match"][1]) == 0


def test_match_instances_score_order_breaks_ties() -> None:
    pred = _stack_box([(0, 10, 0, 5), (0, 10, 0, 5)])  # two identical preds
    ref = _stack_box([(0, 10, 0, 5)])  # one reference
    # Higher-scored prediction (index 1) wins; the other becomes a false positive.
    result = match_instances(pred, ref, scores=np.array([0.1, 0.9]), iou_threshold=0.5)
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert int(result["pred_match"][1]) == 0
    assert int(result["pred_match"][0]) == -1


def test_match_instances_falls_below_threshold() -> None:
    pred = _stack_box([(0, 6, 0, 5)])  # 30 px
    ref = _stack_box([(0, 5, 0, 5)])  # 25 px, IoU = 25/30 ~= 0.83
    high = match_instances(pred, ref, iou_threshold=0.9)
    assert high["tp"] == 0 and high["fn"] == 1
    low = match_instances(pred, ref, iou_threshold=0.5)
    assert low["tp"] == 1 and low["fn"] == 0


def test_instance_f1_at_iou_perfect_match() -> None:
    pred = _stack_box([(0, 5, 0, 5), (10, 15, 10, 15)])
    ref = _stack_box([(0, 5, 0, 5), (10, 15, 10, 15)])
    stats = instance_f1_at_iou(pred, ref, iou_threshold=0.5)
    assert stats["precision"] == pytest.approx(1.0)
    assert stats["recall"] == pytest.approx(1.0)
    assert stats["f1"] == pytest.approx(1.0)


def test_instance_f1_at_iou_partial_credit() -> None:
    pred = _stack_box([(0, 5, 0, 5), (0, 5, 10, 15)])
    ref = _stack_box([(0, 5, 0, 5), (10, 15, 0, 5), (10, 15, 10, 15)])
    stats = instance_f1_at_iou(pred, ref, iou_threshold=0.5)
    # 1 TP (first pred matches first ref), 1 FP (second pred has no match),
    # 2 FN (two refs unmatched).
    assert stats["tp"] == 1.0
    assert stats["fp"] == 1.0
    assert stats["fn"] == 2.0
    assert stats["precision"] == pytest.approx(0.5)
    assert stats["recall"] == pytest.approx(1.0 / 3.0)


def test_average_precision_matched_is_one_for_perfect_ordering() -> None:
    pred = _stack_box([(0, 5, 0, 5), (10, 15, 10, 15)])
    ref = _stack_box([(0, 5, 0, 5), (10, 15, 10, 15)])
    ap = average_precision_matched(
        pred, ref, scores=np.array([0.9, 0.8]), iou_threshold=0.5
    )
    assert ap == pytest.approx(1.0)


def test_average_precision_matched_drops_when_fp_outscores_tp() -> None:
    pred = _stack_box(
        [(0, 5, 0, 5), (15, 20, 15, 20)]
    )  # pred 0 matches a ref, pred 1 is a FP
    ref = _stack_box([(0, 5, 0, 5), (10, 15, 10, 15)])
    # Order matters: low scores last -> hits in order. Place FP first to
    # depress precision on the leading recall step.
    ap = average_precision_matched(
        pred, ref, scores=np.array([0.1, 0.9]), iou_threshold=0.5
    )
    # With FP first then TP, cumulative precision goes 0 -> 0.5; recall
    # goes 0 -> 0.5. Integrated AP equals 0.25.
    assert ap == pytest.approx(0.25)


def test_average_precision_matched_zero_for_empty_ref_or_pred() -> None:
    pred = _stack_box([(0, 5, 0, 5)])
    empty = np.zeros((0, 20, 20), dtype=bool)
    assert average_precision_matched(pred, empty, iou_threshold=0.5) == 0.0
    assert average_precision_matched(empty, pred, iou_threshold=0.5) == 0.0


def test_operators_wrap_pure_functions() -> None:
    pred = _stack_box([(0, 5, 0, 5)])
    ref = _stack_box([(0, 5, 0, 5)])
    scores = np.array([0.9])

    assert np.array_equal(MaskIoU()(pred, ref), mask_iou_matrix(pred, ref))

    matcher = InstanceMatcher(iou_threshold=0.5, scores=scores)
    direct = match_instances(pred, ref, scores=scores, iou_threshold=0.5)
    out = matcher(pred, ref)
    assert out["tp"] == direct["tp"]
    assert np.array_equal(out["pred_match"], direct["pred_match"])

    f1_op = InstanceF1AtIoU(iou_threshold=0.5, scores=scores)
    assert f1_op(pred, ref) == instance_f1_at_iou(
        pred, ref, scores=scores, iou_threshold=0.5
    )

    ap_op = AveragePrecisionMatched(iou_threshold=0.5, scores=scores)
    assert ap_op(pred, ref) == average_precision_matched(
        pred, ref, scores=scores, iou_threshold=0.5
    )


def test_operators_reject_invalid_iou_threshold() -> None:
    for op_cls in (InstanceMatcher, InstanceF1AtIoU, AveragePrecisionMatched):
        with pytest.raises(ValueError):
            op_cls(iou_threshold=0.0)
        with pytest.raises(ValueError):
            op_cls(iou_threshold=1.5)


def test_instance_metrics_do_not_collide_with_v5_object_stubs() -> None:
    """The V5 object-metric stubs (ProbabilityOfDetection,
    IntersectionOverUnion, etc.) still raise on construction — the new
    instance metrics are additive and live under their own names.
    """
    from xrtoolz.metrics._src.object import (
        CentroidDistance,
        IntersectionOverUnion,
        ProbabilityOfDetection,
    )

    for stub in (ProbabilityOfDetection, IntersectionOverUnion, CentroidDistance):
        with pytest.raises(NotImplementedError):
            stub()


def test_operators_get_config_round_trip() -> None:
    scores = np.array([0.7, 0.3])
    cfg = InstanceMatcher(iou_threshold=0.25, scores=scores).get_config()
    assert cfg == {"iou_threshold": 0.25, "scores": [0.7, 0.3]}
    cfg_default = AveragePrecisionMatched(iou_threshold=0.5).get_config()
    assert cfg_default == {"iou_threshold": 0.5, "scores": None}
