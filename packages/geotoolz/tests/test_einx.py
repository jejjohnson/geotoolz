"""Tests for `geotoolz.einx`.

The pattern-analysis tier (`spatial_survives` / `output_axes`) is pure
string processing; the operator tier calls einx (a core dependency).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from _helpers import fill_pixel_mask, toy_geotensor

import geotoolz as gz
from geotoolz.einx import (
    CHWtoHWC,
    Einx,
    HWCtoCHW,
    PerBandReduce,
    SpatialPool,
)
from geotoolz.einx._src.array import output_axes, spatial_survives


# ---------------------------------------------------------------------------
# Tier-A: pattern analysis (no einx needed, but grouped here for locality)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("c y x -> y x", True),
        ("c y x -> c y x", True),
        ("band y x, sig band -> sig y x", True),
        ("c y x -> x y", False),  # transposed
        ("c y x -> y x c", False),  # channels-last
        ("c y x -> c", False),  # spatial consumed
        ("c (y py) (x px) -> c y x", False),  # pooled: sizes change
        ("(y x) c -> y x c", False),  # composed input
        ("b [y x] -> b [y x]", False),  # bracketed vmap axes
        ("c y x", False),  # no explicit output
        ("t c y x -> t c y x", True),
        ("c x y -> c y x", False),  # input-side transpose (#123)
        ("y x c -> c y x", False),  # channels-last input (#123)
        ("c a b -> c y x", False),  # renamed spatial axes (#123)
        ("sig band, band y x -> sig y x", False),  # carrier is the first input
    ],
)
def test_spatial_survives(pattern: str, expected: bool) -> None:
    assert spatial_survives(pattern) is expected


def test_spatial_survives_custom_axes() -> None:
    assert spatial_survives("c h w -> h w", ("h", "w"))
    assert not spatial_survives("c h w -> h w")


def test_output_axes_and_errors() -> None:
    assert output_axes("a (b c) -> (a b) c") == ["(a b)", "c"]
    assert output_axes("c y x") is None
    with pytest.raises(ValueError, match="unbalanced"):
        spatial_survives("c (y x -> y x")


# ---------------------------------------------------------------------------
# Tier-B: the Einx operator
# ---------------------------------------------------------------------------


def _gt(values: np.ndarray):
    # -9999 fill: these fixtures contain real zeros, and a 0 fill would
    # mark them as nodata (georeader's convention).
    return toy_geotensor(values, fill_value_default=-9999)


def test_einx_surviving_pattern_returns_geotensor() -> None:
    gt = _gt(np.arange(24, dtype=float).reshape(2, 3, 4))
    out = Einx(op="mean", pattern="c y x -> y x")(gt)
    assert type(out).__name__ == "GeoTensor"
    assert out.shape == (3, 4)
    assert out.transform == gt.transform
    assert out.crs == gt.crs
    np.testing.assert_allclose(np.asarray(out), np.asarray(gt).mean(axis=0))


def test_einx_destructive_pattern_returns_plain_array() -> None:
    gt = _gt(np.arange(24, dtype=float).reshape(2, 3, 4))
    out = Einx(op="sum", pattern="c y x -> c")(gt)
    assert type(out) is np.ndarray
    np.testing.assert_allclose(out, np.asarray(gt).sum(axis=(1, 2)))


def test_einx_plain_array_in_plain_array_out() -> None:
    arr = np.arange(24, dtype=float).reshape(2, 3, 4)
    out = Einx(op="mean", pattern="c y x -> y x")(arr)
    assert type(out) is np.ndarray
    np.testing.assert_allclose(out, arr.mean(axis=0))


def test_einx_multi_input_dot_keeps_georeferencing() -> None:
    gt = _gt(np.arange(24, dtype=float).reshape(2, 3, 4))
    signatures = np.asarray([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])  # (sig, band)
    op = Einx(op="dot", pattern="band y x, sig band -> sig y x")
    out = op(gt, signatures)
    # The band axis changes (2 -> 3 signatures) but the spatial grid is
    # untouched, so matched-filter-style scores stay georeferenced.
    assert type(out).__name__ == "GeoTensor"
    assert out.shape == (3, 3, 4)
    assert out.transform == gt.transform
    expected = np.einsum("byx,sb->syx", np.asarray(gt), signatures)
    np.testing.assert_allclose(np.asarray(out), expected)


def test_input_side_transpose_does_not_survive() -> None:
    # Square tile: the transposed output has the carrier's shape, so the
    # shape check in `array_as_geotensor` can't catch a stale transform.
    gt = _gt(np.arange(48, dtype=float).reshape(3, 4, 4))
    out = Einx(op="id", pattern="c x y -> c y x")(gt)
    assert type(out) is np.ndarray
    np.testing.assert_allclose(out, np.swapaxes(np.asarray(gt), 1, 2))


def test_hwc_to_chw_survival_comes_from_pattern_analysis() -> None:
    # No hand-set override needed: the pattern itself is non-surviving.
    assert not spatial_survives("y x c -> c y x")
    assert HWCtoCHW()._survives is False


def test_einx_rejects_bad_ops() -> None:
    with pytest.raises(ValueError, match="not supported"):
        Einx(op="vmap", pattern="c y x -> y x")
    with pytest.raises(ValueError, match="not an einx operation"):
        Einx(op="definitely_not_real", pattern="c y x -> y x")


def test_einx_get_config_roundtrip() -> None:
    op = Einx(op="mean", pattern="c (y py) (x px) -> c y x", py=2, px=2)
    cfg = op.get_config()
    # Extra einx kwargs are nested, never flattened next to op / pattern.
    assert cfg == {
        "op": "mean",
        "pattern": "c (y py) (x px) -> c y x",
        "spatial_axes": ["y", "x"],
        "op_kwargs": [["py", 2], ["px", 2]],
    }
    rebuilt = Einx(**cfg)
    arr = np.arange(16, dtype=float).reshape(1, 4, 4)
    np.testing.assert_allclose(rebuilt(arr), op(arr))
    reloaded = gz.Einx.from_state(json.loads(json.dumps(op.state)))
    assert reloaded.op_kwargs == {"py": 2, "px": 2}
    np.testing.assert_allclose(reloaded(arr), op(arr))


def test_einx_legacy_flat_config_reloads() -> None:
    # Configs written before op_kwargs was nested still rebuild.
    legacy = {"op": "mean", "pattern": "c (y py) (x px) -> c y x", "py": 2, "px": 2}
    assert Einx(**legacy).op_kwargs == {"py": 2, "px": 2}


def test_unknown_kwarg_rejected() -> None:
    with pytest.raises(ValueError, match="spatial_axis"):
        Einx(op="mean", pattern="c y x -> y x", spatial_axis=("h", "w"))
    with pytest.raises(ValueError, match="pz"):
        Einx(op="mean", pattern="c (y py) x -> c y x", op_kwargs={"pz": 2})
    with pytest.raises(ValueError, match="both"):
        Einx(op="mean", pattern="c (y py) x -> c y x", op_kwargs={"py": 2}, py=2)
    # einx call options and pattern axis sizes are accepted.
    Einx(op="mean", pattern="c y x -> y x", op_kwargs={"backend": "numpy"})
    Einx(op="mean", pattern="c (y py) x -> c y x", op_kwargs={"py": 2})


def test_pattern_rank_mismatch_names_operator_and_shape() -> None:
    op = Einx(op="rearrange", pattern="c h w -> c w h")
    with pytest.raises(ValueError, match=r"Einx: pattern .*\(2, 3, 4, 4\)"):
        op(np.zeros((2, 3, 4, 4)))


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


def test_chw_hwc_roundtrip() -> None:
    gt = _gt(np.arange(24, dtype=float).reshape(2, 3, 4))
    hwc = CHWtoHWC()(gt)
    assert type(hwc) is np.ndarray
    assert hwc.shape == (3, 4, 2)
    chw = HWCtoCHW()(hwc)
    assert type(chw) is np.ndarray
    np.testing.assert_allclose(chw, np.asarray(gt))
    assert CHWtoHWC().get_config() == {}


def test_per_band_reduce() -> None:
    gt = toy_geotensor(np.arange(1, 25, dtype=float).reshape(2, 3, 4))
    out = PerBandReduce(reduce="max")(gt)
    assert type(out) is np.ndarray
    np.testing.assert_allclose(out, np.asarray(gt).max(axis=(1, 2)))
    assert PerBandReduce(reduce="max").get_config() == {"reduce": "max"}


def test_spatial_pool_scales_transform() -> None:
    gt = _gt(np.arange(32, dtype=float).reshape(2, 4, 4))
    out = SpatialPool(reduce="mean", factor=2)(gt)
    assert type(out).__name__ == "GeoTensor"
    assert out.shape == (2, 2, 2)
    assert out.transform.a == gt.transform.a * 2  # pixel width doubled
    assert out.transform.e == gt.transform.e * 2  # pixel height doubled
    assert out.transform.c == gt.transform.c  # same origin
    assert out.transform.f == gt.transform.f
    block = np.asarray(gt)[:, :2, :2]
    np.testing.assert_allclose(np.asarray(out)[:, 0, 0], block.mean(axis=(1, 2)))


def test_spatial_pool_plain_and_2d() -> None:
    arr = np.arange(16, dtype=float).reshape(4, 4)
    out = SpatialPool(reduce="max", factor=(2, 2))(arr)
    assert type(out) is np.ndarray
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out, [[5.0, 7.0], [13.0, 15.0]])


def test_spatial_pool_divisibility_error() -> None:
    gt = _gt(np.zeros((1, 5, 4)))
    with pytest.raises(ValueError, match="not divisible"):
        SpatialPool(factor=2)(gt)
    with pytest.raises(ValueError, match=">= 1"):
        SpatialPool(factor=0)
    assert SpatialPool(factor=(2, 3)).get_config() == {
        "reduce": "mean",
        "factor": [2, 3],
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_factor_validation() -> None:
    assert SpatialPool(factor=np.int64(2)).factor == (2, 2)
    assert SpatialPool(factor=(np.int32(2), 3)).factor == (2, 3)
    with pytest.raises(TypeError, match="factor"):
        SpatialPool(factor=True)
    with pytest.raises(TypeError, match="factor"):
        SpatialPool(factor=(True, 2))
    with pytest.raises(TypeError, match="factor"):
        SpatialPool(factor=2.0)
    with pytest.raises(ValueError, match="pair"):
        SpatialPool(factor=(2,))
    with pytest.raises(ValueError, match="pair"):
        SpatialPool(factor=(2, 2, 2))
    with pytest.raises(ValueError, match=">= 1"):
        SpatialPool(factor=(2, 0))


@pytest.mark.parametrize("reduce", ["rearrange", "dot", "id", "any", "nope"])
def test_reduce_whitelist(reduce: str) -> None:
    with pytest.raises(ValueError, match="reduce must be one of"):
        SpatialPool(reduce=reduce)
    with pytest.raises(ValueError, match="reduce must be one of"):
        PerBandReduce(reduce=reduce)


# ---------------------------------------------------------------------------
# Nodata
# ---------------------------------------------------------------------------


def test_spatial_pool_ignores_fill() -> None:
    values = np.ones((2, 4, 4))
    values[:, 0, 0] = -9999.0  # one fill pixel in the top-left block
    values[:, 2:, 2:] = -9999.0  # a fully-invalid bottom-right block
    values[1, 0, 3] = np.nan  # a NaN in band 1 invalidates the whole pixel
    values[0, 0, 3] = 5.0
    gt = toy_geotensor(values, attrs={"band_names": ["a", "b"]})
    out = SpatialPool(reduce="mean", factor=2)(gt)
    arr = np.asarray(out)
    np.testing.assert_allclose(arr[:, 0, 0], [1.0, 1.0])  # not -2499.0
    np.testing.assert_allclose(arr[:, 0, 1], [1.0, 1.0])  # 5.0 excluded with its NaN
    np.testing.assert_allclose(arr[:, 1, 0], [1.0, 1.0])
    np.testing.assert_array_equal(arr[:, 1, 1], [-9999.0, -9999.0])
    assert out.fill_value_default == -9999
    assert out.attrs is not gt.attrs
    assert out.attrs["band_names"] == ["a", "b"]
    # Sum must not turn an all-invalid block into 0.
    summed = np.asarray(SpatialPool(reduce="sum", factor=2)(gt))
    np.testing.assert_array_equal(summed[:, 1, 1], [-9999.0, -9999.0])
    np.testing.assert_allclose(summed[:, 0, 0], [3.0, 3.0])


def test_spatial_pool_nan_block_without_fill() -> None:
    values = np.ones((1, 4, 4))
    values[0, 0, 0] = np.nan
    values[0, 2:, :2] = np.nan
    out = SpatialPool(reduce="mean", factor=2)(values)
    assert type(out) is np.ndarray
    np.testing.assert_allclose(out[0], [[1.0, 1.0], [np.nan, 1.0]])
    gt = toy_geotensor(values, fill_value_default=None)
    pooled = SpatialPool(factor=2)(gt)
    assert np.isnan(pooled.fill_value_default)


def test_spatial_pool_integer_extrema_keep_dtype() -> None:
    values = np.arange(1, 17, dtype=np.uint16).reshape(1, 4, 4)
    values[0, :2, :2] = 0
    gt = toy_geotensor(values, fill_value_default=0)
    out = SpatialPool(reduce="max", factor=2)(gt)
    assert out.dtype == np.uint16
    np.testing.assert_array_equal(np.asarray(out)[0], [[0, 8], [14, 16]])
    assert out.fill_value_default == 0
    mean = SpatialPool(reduce="mean", factor=2)(gt)
    assert mean.dtype == np.float64


def test_per_band_reduce_ignores_fill() -> None:
    values = np.stack([np.full((4, 4), 2.0), np.full((4, 4), 3.0)])
    gt = toy_geotensor(values, with_fill_pixels=True)
    np.testing.assert_allclose(PerBandReduce(reduce="mean")(gt), [2.0, 3.0])
    np.testing.assert_allclose(PerBandReduce(reduce="min")(gt), [2.0, 3.0])
    nan_gt = toy_geotensor(values.copy(), fill_value_default=None)
    np.asarray(nan_gt)[1, 0, 0] = np.nan
    np.testing.assert_allclose(PerBandReduce(reduce="std")(nan_gt), [0.0, 0.0])
    empty = toy_geotensor(np.full((2, 2, 2), -9999.0))
    np.testing.assert_array_equal(PerBandReduce(reduce="sum")(empty), [np.nan] * 2)


def test_einx_value_changing_output_gets_nan_fill() -> None:
    values = np.arange(1, 49, dtype=float).reshape(3, 4, 4)
    gt = toy_geotensor(values, with_fill_pixels=True)
    out = Einx(op="mean", pattern="c y x -> y x")(gt)
    assert out.dtype == np.float64
    assert np.isnan(out.fill_value_default)
    arr = np.asarray(out)
    mask = fill_pixel_mask(gt.shape)
    assert np.isnan(arr[mask]).all()
    np.testing.assert_allclose(arr[~mask], values.mean(axis=0)[~mask])


def test_einx_boolean_output_gets_false_fill() -> None:
    gt = toy_geotensor(np.full((2, 4, 4), 0.9), with_fill_pixels=True)
    out = Einx(op="greater", pattern="c y x, c -> c y x")(gt, np.full(2, 0.5))
    assert out.dtype == np.bool_
    assert out.fill_value_default is False
    assert not np.asarray(out)[:, fill_pixel_mask(gt.shape)].any()
    assert np.asarray(out)[:, ~fill_pixel_mask(gt.shape)].all()


def test_einx_integer_output_keeps_representable_fill() -> None:
    values = np.arange(1, 33, dtype=np.int32).reshape(2, 4, 4)
    gt = toy_geotensor(values, fill_value_default=-1, with_fill_pixels=True)
    out = Einx(op="sum", pattern="c y x -> y x")(gt)
    assert out.fill_value_default == -1
    np.testing.assert_array_equal(np.asarray(out)[fill_pixel_mask(gt.shape)], -1)


def test_einx_id_inherits_fill() -> None:
    gt = toy_geotensor(np.ones((2, 3, 4, 4)), with_fill_pixels=True)
    out = Einx(op="id", pattern="t c y x -> c t y x")(gt)
    assert out.fill_value_default == -9999
    np.testing.assert_array_equal(np.asarray(out), np.swapaxes(np.asarray(gt), 0, 1))


# ---------------------------------------------------------------------------
# 4-D (time, band, y, x) carriers
# ---------------------------------------------------------------------------


def test_4d_time_stack() -> None:
    rng = np.random.default_rng(0)
    values = rng.uniform(1.0, 2.0, (2, 3, 4, 4))
    values[1, :, 0, 0] = -9999.0  # fill in frame 1 only
    gt = toy_geotensor(values, attrs={"band_names": ["a", "b", "c"]})

    pooled = SpatialPool(reduce="mean", factor=2)(gt)
    assert pooled.shape == (2, 3, 2, 2)
    assert pooled.transform.a == gt.transform.a * 2
    assert pooled.attrs["band_names"] == ["a", "b", "c"]
    np.testing.assert_allclose(
        np.asarray(pooled)[0, :, 0, 0], values[0, :, :2, :2].mean(axis=(1, 2))
    )
    frame1 = values[1, :, :2, :2].reshape(3, -1)[:, 1:]  # fill pixel excluded
    np.testing.assert_allclose(np.asarray(pooled)[1, :, 0, 0], frame1.mean(axis=1))

    stats = PerBandReduce(reduce="max")(gt)
    assert stats.shape == (2, 3)
    np.testing.assert_allclose(stats[0], values[0].max(axis=(1, 2)))

    # Per-frame validity: frame 0 keeps the pixel frame 1 lost.
    band_mean = Einx(op="mean", pattern="t c y x -> t y x")(gt)
    assert band_mean.shape == (2, 4, 4)
    assert not np.isnan(np.asarray(band_mean)[0, 0, 0])
    assert np.isnan(np.asarray(band_mean)[1, 0, 0])

    hwc = CHWtoHWC()(gt)
    assert hwc.shape == (2, 4, 4, 3)
    np.testing.assert_array_equal(HWCtoCHW()(hwc), values)


def test_per_band_reduce_rejects_2d() -> None:
    with pytest.raises(ValueError, match=r"\(C, H, W\)"):
        PerBandReduce()(np.ones((4, 4)))


# ---------------------------------------------------------------------------
# Top-level lazy exports
# ---------------------------------------------------------------------------


def test_top_level_lazy_exports() -> None:
    assert gz.Einx is Einx
    assert gz.SpatialPool is SpatialPool
    assert "Einx" in gz.__all__
    with pytest.raises(AttributeError):
        gz.NotARealOperator  # noqa: B018
