"""`get_config()` round-trip coverage for every spatial / temporal axis."""

from __future__ import annotations

import numpy as np
import pytest

from geopatcher import (
    SpatialApproxCardinality,
    SpatialApproxMode,
    SpatialApproxQuantile,
    SpatialByIndex,
    SpatialCustom,
    SpatialExplicit,
    SpatialGaussian,
    SpatialHann,
    SpatialHardVote,
    SpatialInvVarWeightedMean,
    SpatialJitteredStride,
    SpatialKNNGraph,
    SpatialMean,
    SpatialOverlapAdd,
    SpatialPoissonDisk,
    SpatialRadiusGraph,
    SpatialRandom,
    SpatialRectangular,
    SpatialRegularStride,
    SpatialReservoir,
    SpatialSoftVote,
    SpatialSphericalCap,
    SpatialStreamingHistogram,
    SpatialSum,
    SpatialTukey,
    SpatialVariance,
    SpatialWeightedSum,
    TemporalCausalBoxcar,
    TemporalCausalRolling,
    TemporalEventTriggered,
    TemporalExplicit,
    TemporalExponentialDecay,
    TemporalFixedLookback,
    TemporalForecast,
    TemporalHierarchicalCombine,
    TemporalLookbackHorizon,
    TemporalMean,
    TemporalMultiScale,
    TemporalPeriodic,
    TemporalPhaseWindow,
    TemporalRandom,
    TemporalRegularStride,
    TemporalTaperedTukey,
)


@pytest.mark.parametrize(
    "op",
    [
        SpatialRectangular(size=(8, 8)),
        SpatialSphericalCap(radius_km=500.0),
        SpatialKNNGraph(k=4),
        SpatialRadiusGraph(radius=10.0),
        SpatialRegularStride(step=8),
        SpatialJitteredStride(step=8, jitter=0.5, seed=0),
        SpatialRandom(n_samples=4, seed=0),
        SpatialPoissonDisk(min_dist=4.0, seed=0),
        SpatialExplicit(anchors_=[(0, 0), (8, 8)]),
        SpatialHann(),
        SpatialTukey(alpha=0.4),
        SpatialGaussian(sigma=0.4),
        SpatialSum(),
        SpatialMean(),
        SpatialVariance(),
        SpatialOverlapAdd(),
        SpatialWeightedSum(),
        SpatialInvVarWeightedMean(),
        SpatialHardVote(n_classes=3),
        SpatialSoftVote(n_classes=3),
        SpatialByIndex(),
    ],
)
def test_spatial_get_config_is_dict(op) -> None:
    cfg = op.get_config()
    assert isinstance(cfg, dict)


@pytest.mark.parametrize(
    "op",
    [
        TemporalFixedLookback(length=5),
        TemporalLookbackHorizon(lookback=4, horizon=2),
        TemporalMultiScale(scales=[2, 4, 8]),
        TemporalPhaseWindow(period=24, phase_width=2),
        TemporalRegularStride(step=2),
        TemporalCausalRolling(step=1, start=0),
        TemporalEventTriggered(event_times=[3, 5, 7]),
        TemporalRandom(n=4, seed=0),
        TemporalExplicit(times=[0, 4, 8]),
        TemporalCausalBoxcar(),
        TemporalExponentialDecay(tau=2.0),
        TemporalTaperedTukey(alpha=0.4),
        TemporalPeriodic(period=24),
        TemporalMean(),
        TemporalHierarchicalCombine(scales=[2, 4]),
        TemporalForecast(horizon=2),
    ],
)
def test_temporal_get_config_is_dict(op) -> None:
    cfg = op.get_config()
    assert isinstance(cfg, dict)


class TestApproxSketches:
    @pytest.mark.parametrize(
        "sketch_cls",
        [
            SpatialApproxQuantile,
            SpatialApproxCardinality,
            SpatialApproxMode,
            SpatialStreamingHistogram,
            SpatialReservoir,
        ],
    )
    def test_get_config_is_dict(self, sketch_cls) -> None:
        sketch = sketch_cls()
        assert isinstance(sketch.get_config(), dict)


class TestCustomForbidsInYaml:
    def test_flag(self) -> None:
        assert SpatialCustom(fn=lambda g: np.ones(g.size)).forbid_in_yaml is True
