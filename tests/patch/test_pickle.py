"""Pickle round-trip for the YAML-safe axes.

`SpatialCustom`, `TemporalFold`, and `SpatialLearned` carry closures and
are intentionally excluded.
"""

from __future__ import annotations

import pickle

import pytest

from geotoolz.patch import (
    SpatialBoxcar,
    SpatialHann,
    SpatialMean,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRectangular,
    SpatialRegularStride,
    SpatialSum,
    SpatialTukey,
    TemporalCausalBoxcar,
    TemporalExponentialDecay,
    TemporalFixedLookback,
    TemporalLookbackHorizon,
    TemporalMean,
    TemporalPatcher,
    TemporalRegularStride,
)


@pytest.mark.parametrize(
    "op",
    [
        SpatialRectangular(size=(8, 8)),
        SpatialRegularStride(step=8),
        SpatialBoxcar(),
        SpatialHann(),
        SpatialTukey(alpha=0.5),
        SpatialSum(),
        SpatialMean(),
        SpatialOverlapAdd(),
        TemporalFixedLookback(length=5),
        TemporalLookbackHorizon(lookback=3, horizon=2),
        TemporalRegularStride(step=2),
        TemporalCausalBoxcar(),
        TemporalExponentialDecay(tau=2.0),
        TemporalMean(),
    ],
)
def test_axis_pickle_roundtrip(op) -> None:
    blob = pickle.dumps(op)
    clone = pickle.loads(blob)
    assert type(clone) is type(op)
    assert clone.get_config() == op.get_config()


class TestSpatialPatcherPickle:
    def test_roundtrip(self) -> None:
        sp = SpatialPatcher(
            geometry=SpatialRectangular(size=(8, 8)),
            sampler=SpatialRegularStride(step=8),
            window=SpatialHann(),
            aggregation=SpatialOverlapAdd(),
        )
        clone = pickle.loads(pickle.dumps(sp))
        assert clone.get_config() == sp.get_config()


class TestTemporalPatcherPickle:
    def test_roundtrip(self) -> None:
        tp = TemporalPatcher(
            geometry=TemporalFixedLookback(length=4),
            sampler=TemporalRegularStride(step=2),
            window=TemporalCausalBoxcar(),
            aggregation=TemporalMean(),
        )
        clone = pickle.loads(pickle.dumps(tp))
        assert clone.get_config() == tp.get_config()
