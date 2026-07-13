"""Tests for reference runner helpers."""

from __future__ import annotations

import numpy as np
import pytest

from geopatcher import (
    RasterField,
    SpatialBoxcar,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRectangular,
    SpatialRegularStride,
)
from geopatcher.runners import parallel_map


def _double(data) -> np.ndarray:
    return np.asarray(data) * 2


@pytest.fixture
def field(raster_field_factory) -> RasterField:
    return raster_field_factory(32)


@pytest.fixture
def patcher() -> SpatialPatcher:
    return SpatialPatcher(
        geometry=SpatialRectangular(size=(8, 8)),
        sampler=SpatialRegularStride(step=8),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )


def test_parallel_map_preserves_sequential_output(
    patcher: SpatialPatcher, field: RasterField
) -> None:
    expected = [
        type(p)(
            data=_double(p.data), anchor=p.anchor, indices=p.indices, weights=p.weights
        )
        for p in patcher.split(field)
    ]

    actual = parallel_map(patcher, field, _double, n_workers=4)

    assert [p.anchor for p in actual] == [p.anchor for p in expected]
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(got.data, want.data)


def test_parallel_map_supports_process_backend(
    patcher: SpatialPatcher, field: RasterField
) -> None:
    patches = parallel_map(patcher, field, _double, n_workers=2, backend="process")

    assert [p.anchor for p in patches] == patcher.anchors(field)


def test_parallel_map_process_backend_rejects_unpicklable_operator(
    patcher: SpatialPatcher, field: RasterField
) -> None:
    scale = 2

    def local_op(data):
        return np.asarray(data) * scale

    with pytest.raises(TypeError, match="requires a picklable operator"):
        parallel_map(patcher, field, local_op, backend="process")


def test_parallel_map_skip_policy_omits_failed_patches(
    patcher: SpatialPatcher, field: RasterField
) -> None:
    anchors = patcher.anchors(field)

    def fail_first_patch(data):
        arr = np.asarray(data)
        if arr[0, 0] == 0:
            raise ValueError("boom")
        return arr

    with pytest.warns(RuntimeWarning, match="skipped patch"):
        patches = parallel_map(patcher, field, fail_first_patch, on_error="skip")

    assert len(patches) == len(anchors) - 1
    assert [p.anchor for p in patches] == anchors[1:]
