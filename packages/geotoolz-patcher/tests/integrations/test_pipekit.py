"""Tests for the pipekit-Operator wrappers around `geopatcher`.

Skipped cleanly when the optional ``[pipekit]`` extra isn't installed.
"""

from __future__ import annotations

import pytest


pytest.importorskip(
    "pipekit",
    reason="geopatcher.integrations.pipekit requires the [pipekit] extra",
)

import json
import pickle

import numpy as np
import rasterio
from geoarray import GeoArray
from pipekit import Graph, Input, Lambda, Operator, Sequential, check_pickleable

from geopatcher import Patch, RasterField, SpatialPatcher, spatial
from geopatcher.config import axis_envelope, from_config
from geopatcher.integrations.pipekit import (
    ApplyToChips,
    GridSampler,
    MergePatches,
)


@pytest.fixture
def field() -> RasterField:
    # 2-D field so OverlapAdd's row/col slicer matches the domain shape.
    arr = np.ones((16, 16), dtype=np.float32)
    gt = GeoArray(
        data=arr,
        transform=rasterio.Affine.identity(),
        crs="EPSG:32630",
        dims="y x",
    )
    return RasterField(gt)


@pytest.fixture
def patcher() -> SpatialPatcher:
    return SpatialPatcher(
        geometry=spatial.geometry.Rectangular(size=(8, 8)),
        sampler=spatial.sampler.RegularStride(step=8),
        window=spatial.window.Boxcar(),
        aggregation=spatial.aggregation.OverlapAdd(),
    )


class TestGridSampler:
    def test_returns_list_of_patches(
        self, field: RasterField, patcher: SpatialPatcher
    ) -> None:
        op = GridSampler(patcher=patcher)
        patches = op(field)
        assert isinstance(patches, list)
        assert all(isinstance(p, Patch) for p in patches)
        assert len(patches) == 4  # 2x2 tiles


class TestApplyToChips:
    def test_each_chip_runs_through_operator(
        self, field: RasterField, patcher: SpatialPatcher
    ) -> None:
        patches = list(patcher.split(field))
        double = Lambda(lambda gt: np.asarray(gt) * 2.0, name="double")
        out = ApplyToChips(operator=double)(patches)
        assert len(out) == len(patches)
        for src, dst in zip(patches, out, strict=True):
            assert dst.anchor == src.anchor
            np.testing.assert_allclose(dst.data, 2.0)


class TestStitchInSequential:
    def test_chip_predict_stitch_roundtrip(
        self, field: RasterField, patcher: SpatialPatcher
    ) -> None:
        double = Lambda(lambda gt: np.asarray(gt) * 2.0, name="double")
        pipe = Sequential(
            [
                GridSampler(patcher=patcher),
                ApplyToChips(operator=double),
                MergePatches(
                    aggregation=spatial.aggregation.OverlapAdd(), domain=field.domain
                ),
            ]
        )
        result = pipe(field)
        np.testing.assert_allclose(result, 2.0)


class TestOperatorContract:
    def test_grid_sampler_forbid_in_yaml(self, patcher: SpatialPatcher) -> None:
        op = GridSampler(patcher=patcher)
        assert GridSampler.forbid_in_yaml is True
        with pytest.raises(RuntimeError, match="forbid_in_yaml"):
            Operator.from_state(op.state)

    def test_grid_sampler_config_envelopes_the_patcher(
        self, field: RasterField, patcher: SpatialPatcher
    ) -> None:
        cfg = json.loads(json.dumps(GridSampler(patcher=patcher).get_config()))
        # String compare: the NaN fill values never compare equal as floats.
        assert json.dumps(cfg["patcher"]) == json.dumps(axis_envelope(patcher))
        rebuilt = from_config(cfg["patcher"])
        assert [p.anchor for p in rebuilt.split(field)] == [
            p.anchor for p in patcher.split(field)
        ]

    def test_stitch_config_envelopes_the_aggregation(self, field: RasterField) -> None:
        op = MergePatches(
            aggregation=spatial.aggregation.OverlapAdd(), domain=field.domain
        )
        assert op.get_config()["aggregation"] == axis_envelope(
            spatial.aggregation.OverlapAdd()
        )
        with pytest.raises(RuntimeError, match="forbid_in_yaml"):
            Operator.from_state(op.state)

    def test_apply_to_chips_config_nests_the_operator(self) -> None:
        cfg = ApplyToChips(operator=Lambda(lambda gt: gt, name="id")).get_config()
        assert cfg["operator"]["class"] == "Lambda"

    def test_apply_to_chips_keeps_carrier_fields(
        self, field: RasterField, patcher: SpatialPatcher
    ) -> None:
        patches = list(patcher.split(field, max_in_flight=8))
        out = ApplyToChips(operator=Lambda(lambda gt: gt, name="id"))(patches)
        for src, dst in zip(patches, out, strict=True):
            assert (dst.anchor, dst.indices) == (src.anchor, src.indices)
            np.testing.assert_array_equal(dst.weights, src.weights)
            # The copy never owns the source patch's backpressure slot.
            assert dst._release is None
            src.close()

    def test_operators_graph_mode(
        self, field: RasterField, patcher: SpatialPatcher
    ) -> None:
        src = Input("field")
        patches = GridSampler(patcher=patcher)(src)
        doubled = ApplyToChips(
            operator=Lambda(lambda gt: np.asarray(gt) * 2.0, name="double")
        )(patches)
        merged = MergePatches(
            aggregation=spatial.aggregation.OverlapAdd(), domain=field.domain
        )(doubled)
        graph = Graph(inputs={"field": src}, outputs={"merged": merged})
        np.testing.assert_allclose(graph(field=field)["merged"], 2.0)

    def test_check_pickleable_surfaces_custom_window(self) -> None:
        closure_patcher = SpatialPatcher(
            geometry=spatial.geometry.Rectangular(size=(8, 8)),
            sampler=spatial.sampler.RegularStride(step=8),
            window=spatial.window.Custom(fn=lambda g: np.ones(g.size)),
            aggregation=spatial.aggregation.OverlapAdd(),
        )
        sampler = GridSampler(patcher=closure_patcher)
        pipe = Sequential(
            [sampler, ApplyToChips(operator=Lambda(lambda gt: gt, name="id"))]
        )
        assert sampler in check_pickleable(pipe)
        with pytest.raises((pickle.PicklingError, AttributeError, TypeError)):
            pickle.dumps(sampler)


# ---------------------------------------------------------------------------
# PatchReachWarning: patching that changes the answer (#620)
# ---------------------------------------------------------------------------

import types
import warnings
from typing import Any

from scipy import ndimage

from geopatcher.integrations.pipekit import PatchReachWarning


def _coupling(couples: set[str], reach: int | None) -> types.SimpleNamespace:
    """A duck-typed stand-in for geoops.carrier.Coupling."""
    return types.SimpleNamespace(couples=frozenset(couples), reach=reach)


class _Box(Operator):
    """A (2r + 1) x (2r + 1) mean filter: spatial reach r."""

    def __init__(self, *, r: int = 1) -> None:
        self.r = r

    @property
    def coupling(self) -> types.SimpleNamespace:
        return _coupling({"space"}, self.r)

    def _apply(self, x: GeoArray) -> GeoArray:
        size = 2 * self.r + 1
        out = ndimage.uniform_filter(np.asarray(x), size=size, mode="nearest")
        return x.replace(data=out)


class _MinusMean(Operator):
    coupling = _coupling({"space"}, None)

    def _apply(self, x: GeoArray) -> GeoArray:
        values = np.asarray(x)
        return x.replace(data=values - values.mean())


class _Undeclared(Operator):
    def _apply(self, x: GeoArray) -> GeoArray:
        return x


def _scene(n: int = 32) -> GeoArray:
    values = np.random.default_rng(0).uniform(size=(n, n))
    return GeoArray(
        values,
        transform=rasterio.Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0),
        crs="EPSG:32630",
        dims=("y", "x"),
    )


def _chips(size: int, step: int, window: Any = None) -> tuple[RasterField, list]:
    field = RasterField(_scene())
    patcher = SpatialPatcher(
        geometry=spatial.geometry.Rectangular(size=(size, size)),
        sampler=spatial.sampler.RegularStride(step=(step, step)),
        window=window if window is not None else spatial.window.Hann(),
        aggregation=spatial.aggregation.OverlapAdd(),
    )
    return field, GridSampler(patcher=patcher)(field)


def test_warns_when_the_overlap_is_smaller_than_the_reach() -> None:
    _, chips = _chips(size=8, step=6)  # overlap 2 px
    with pytest.warns(PatchReachWarning, match="reach of 4 px.*overlap by 2 px"):
        ApplyToChips(operator=_Box(r=4))(chips)


def test_silent_when_the_overlap_covers_the_reach() -> None:
    _, chips = _chips(size=16, step=8)  # overlap 8 px
    with warnings.catch_warnings():
        warnings.simplefilter("error", PatchReachWarning)
        ApplyToChips(operator=_Box(r=4))(chips)


def test_warns_for_a_global_operator_whatever_the_overlap() -> None:
    _, chips = _chips(size=16, step=4)  # overlap 12 px
    with pytest.warns(PatchReachWarning, match="reads the whole scene"):
        ApplyToChips(operator=_MinusMean())(chips)


def test_silent_for_an_operator_without_coupling_or_when_off() -> None:
    _, chips = _chips(size=8, step=6)
    with warnings.catch_warnings():
        warnings.simplefilter("error", PatchReachWarning)
        ApplyToChips(operator=_Undeclared())(chips)
        ApplyToChips(operator=_MinusMean(), check_reach=False)(chips)


def test_check_reach_round_trips_in_config() -> None:
    op = ApplyToChips(operator=_Undeclared(), check_reach=False)
    assert op.get_config()["check_reach"] is False


def _edge_zero(r: int) -> Any:
    """A boxcar window with zero weight on the outer ``r`` pixels."""

    def weights(geometry: Any) -> np.ndarray:
        h, w = geometry.size
        out = np.zeros((h, w))
        out[r : h - r, r : w - r] = 1.0
        return out

    return spatial.window.Custom(weights)


def test_enough_overlap_reproduces_the_whole_scene() -> None:
    """Gate OC-G4: overlap >= 2r, and the r edge pixels of a patch unweighted.

    A 3 x 3 mean (r = 1) on 16 px patches every 12 px (overlap 4), stitched
    with OverlapAdd and a window that gives each patch's outer pixel zero
    weight, equals the filter on the whole scene.
    """
    field = RasterField(_scene(40))
    patcher = SpatialPatcher(
        geometry=spatial.geometry.Rectangular(size=(16, 16)),
        sampler=spatial.sampler.RegularStride(step=(12, 12)),
        window=_edge_zero(1),
        aggregation=spatial.aggregation.OverlapAdd(),
    )
    chips = GridSampler(patcher=patcher)(field)
    with warnings.catch_warnings():
        warnings.simplefilter("error", PatchReachWarning)
        processed = ApplyToChips(operator=_Box(r=1))(chips)
    merged = MergePatches(
        aggregation=spatial.aggregation.OverlapAdd(), domain=field.domain
    )(processed)
    whole = ndimage.uniform_filter(np.asarray(_scene(40)), size=3, mode="nearest")
    # the scene's own outer pixel is a patch edge with no neighbour to cover it
    interior = (slice(1, -1), slice(1, -1))
    got = np.asarray(merged)[interior]
    np.testing.assert_allclose(got, whole[interior], rtol=0, atol=1e-12)
