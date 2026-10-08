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
from georeader.geotensor import GeoTensor
from pipekit import Graph, Input, Lambda, Operator, Sequential, check_pickleable

from geopatcher import Patch, RasterField, SpatialPatcher, spatial
from geopatcher.config import axis_envelope, from_config
from geopatcher.integrations.pipekit import (
    ApplyToChips,
    GridSampler,
    Stitch,
)


@pytest.fixture
def field() -> RasterField:
    # 2-D field so OverlapAdd's row/col slicer matches the domain shape.
    arr = np.ones((16, 16), dtype=np.float32)
    gt = GeoTensor(
        values=arr,
        transform=rasterio.Affine.identity(),
        crs="EPSG:32630",
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
                Stitch(
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
        op = Stitch(aggregation=spatial.aggregation.OverlapAdd(), domain=field.domain)
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
        merged = Stitch(
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
