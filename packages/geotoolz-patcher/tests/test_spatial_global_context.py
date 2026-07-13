"""Tests for spatial two-pass / global-context helpers."""

from __future__ import annotations

import numpy as np
import pytest

from geopatcher import (
    RasterField,
    SpatialBoxcar,
    SpatialMeanStd,
    SpatialMinMax,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRectangular,
    SpatialRegularStride,
)


@pytest.fixture
def field(raster_field_factory) -> RasterField:
    return raster_field_factory(16, dtype=np.float64)


@pytest.fixture
def patcher() -> SpatialPatcher:
    return SpatialPatcher(
        geometry=SpatialRectangular(size=(4, 4)),
        sampler=SpatialRegularStride(step=4),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )


def test_reduce_mean_std_matches_numpy(
    patcher: SpatialPatcher, field: RasterField
) -> None:
    stats = patcher.reduce(field, agg=SpatialMeanStd())
    data = np.asarray(field.reader)

    assert stats["mean"] == pytest.approx(float(np.mean(data)))
    assert stats["std"] == pytest.approx(float(np.std(data, ddof=1)))


def test_reduce_min_max_matches_numpy(
    patcher: SpatialPatcher, field: RasterField
) -> None:
    stats = patcher.reduce(field, agg=SpatialMinMax())
    data = np.asarray(field.reader)

    assert stats == {"min": float(np.min(data)), "max": float(np.max(data))}


def test_two_pass_applies_global_stats(
    patcher: SpatialPatcher, field: RasterField
) -> None:
    out = patcher.two_pass(
        field,
        reduce_with=SpatialMeanStd(),
        apply=lambda data, stats: (np.asarray(data) - stats["mean"]) / stats["std"],
    )

    np.testing.assert_allclose(np.mean(out), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.std(out, ddof=1), 1.0, atol=1e-12)
