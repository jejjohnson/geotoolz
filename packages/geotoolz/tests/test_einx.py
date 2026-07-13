"""Tests for `geotoolz.einx`.

The pattern-analysis tier (`spatial_survives` / `output_axes`) is pure
string processing and runs without the extra; everything touching the
`Einx` operators requires einx and skips otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest
from _helpers import toy_geotensor

from geotoolz.einx._src.array import output_axes, spatial_survives


einx = pytest.importorskip("einx", reason="requires the [einx] extra")

import geotoolz as gz
from geotoolz.einx import (
    CHWtoHWC,
    Einx,
    HWCtoCHW,
    PerBandReduce,
    SpatialPool,
)


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
    return toy_geotensor(values, fill_value_default=0)


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


def test_einx_rejects_bad_ops() -> None:
    with pytest.raises(ValueError, match="not supported"):
        Einx(op="vmap", pattern="c y x -> y x")
    with pytest.raises(ValueError, match="not an einx operation"):
        Einx(op="definitely_not_real", pattern="c y x -> y x")


def test_einx_get_config_roundtrip() -> None:
    op = Einx(op="mean", pattern="c (y py) (x px) -> c y x", py=2, px=2)
    cfg = op.get_config()
    assert cfg == {
        "op": "mean",
        "pattern": "c (y py) (x px) -> c y x",
        "spatial_axes": ["y", "x"],
        "py": 2,
        "px": 2,
    }
    rebuilt = Einx(**cfg)
    arr = np.arange(16, dtype=float).reshape(1, 4, 4)
    np.testing.assert_allclose(rebuilt(arr), op(arr))


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
    gt = _gt(np.arange(24, dtype=float).reshape(2, 3, 4))
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
# Top-level lazy exports
# ---------------------------------------------------------------------------


def test_top_level_lazy_exports() -> None:
    assert gz.Einx is Einx
    assert gz.SpatialPool is SpatialPool
    assert "Einx" in gz.__all__
    with pytest.raises(AttributeError):
        gz.NotARealOperator  # noqa: B018
