"""Lifecycle tests for the patch carriers' backpressure-slot release.

Covers the `weakref.finalize`-based safety net: a patch dropped without
``close()`` — including one caught in a reference cycle — must still
hand its ``max_in_flight`` slot back once collected, while ``close()``
and the context-manager path stay the primary, exactly-once mechanism.
"""

from __future__ import annotations

import gc
import threading

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor

from geopatcher import (
    RasterField,
    SpatialBoxcar,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRectangular,
    SpatialRegularStride,
)
from geopatcher._src.patch import Patch, SpatioTemporalPatch, TemporalPatch


def _patch(release=None) -> Patch:
    return Patch(
        data=np.zeros((2, 2), dtype=np.float32),
        anchor=(0, 0),
        indices=None,
        weights=None,
        _release=release,
    )


class TestCloseIsPrimaryPath:
    def test_close_releases_exactly_once(self) -> None:
        released: list[int] = []
        patch = _patch(lambda: released.append(1))
        patch.close()
        assert released == [1]

    def test_double_close_is_a_noop(self) -> None:
        released: list[int] = []
        patch = _patch(lambda: released.append(1))
        patch.close()
        patch.close()
        assert released == [1]

    def test_context_manager_releases_exactly_once(self) -> None:
        released: list[int] = []
        with _patch(lambda: released.append(1)) as patch:
            assert released == []
        assert released == [1]
        # Exiting the block closed it; a later explicit close is a no-op.
        patch.close()
        assert released == [1]

    def test_closed_patch_is_not_released_again_on_collection(self) -> None:
        released: list[int] = []
        patch = _patch(lambda: released.append(1))
        patch.close()
        del patch
        gc.collect()
        assert released == [1]


class TestFinalizerSafetyNet:
    def test_dropped_patch_releases_slot(self) -> None:
        released: list[int] = []
        patch = _patch(lambda: released.append(1))
        del patch
        gc.collect()
        assert released == [1]

    def test_ref_cycled_patch_releases_after_collection(self) -> None:
        released: list[int] = []
        patch = _patch(lambda: released.append(1))
        cycle = {"patch": patch}
        patch.weights = cycle  # patch -> dict -> patch reference cycle
        del patch, cycle
        gc.collect()
        assert released == [1]

    def test_release_assigned_after_construction_is_finalized(self) -> None:
        # Patchers attach ownership in-place after building the patch.
        released: list[int] = []
        patch = _patch()
        patch._release = lambda: released.append(1)
        del patch
        gc.collect()
        assert released == [1]

    def test_reassigning_release_detaches_previous_finalizer(self) -> None:
        first: list[int] = []
        second: list[int] = []
        patch = _patch(lambda: first.append(1))
        patch._release = lambda: second.append(1)
        del patch
        gc.collect()
        assert first == []
        assert second == [1]


class TestWithData:
    def test_with_data_copy_does_not_own_the_slot(self) -> None:
        released: list[int] = []
        patch = _patch(lambda: released.append(1))
        copy = patch.with_data(np.ones((2, 2)))
        del copy
        gc.collect()
        assert released == []
        patch.close()
        assert released == [1]


@pytest.mark.parametrize(
    "make",
    [
        lambda cb: Patch(data=1, anchor=0, indices=None, _release=cb),
        lambda cb: TemporalPatch(data=1, anchor=0, indices=None, _release=cb),
        lambda cb: SpatioTemporalPatch(data=1, space=0, time=0, _release=cb),
    ],
    ids=["Patch", "TemporalPatch", "SpatioTemporalPatch"],
)
class TestAllCarriersShareTheLifecycle:
    def test_close_idempotent_and_finalizer_detached(self, make) -> None:
        released: list[int] = []
        patch = make(lambda: released.append(1))
        with patch:
            pass
        patch.close()
        del patch
        gc.collect()
        assert released == [1]

    def test_dropped_carrier_releases_via_finalizer(self, make) -> None:
        released: list[int] = []
        patch = make(lambda: released.append(1))
        del patch
        gc.collect()
        assert released == [1]


class TestSplitBackpressureIntegration:
    """End-to-end: `max_in_flight` slots free even without close()."""

    def _field(self) -> RasterField:
        gt = GeoTensor(
            values=np.arange(64, dtype=np.float32).reshape(8, 8),
            transform=rasterio.Affine.identity(),
            crs="EPSG:32630",
        )
        return RasterField(gt)

    def test_iteration_without_close_completes(self) -> None:
        """Dropped patches must return slots via the finalizer.

        With ``max_in_flight=2`` and a consumer that never calls
        ``close()``, the iterator can only make progress if rebinding
        the loop variable (dropping the previous patch) frees its slot.
        Run in a worker thread so a regression fails the test instead
        of hanging the suite.
        """
        patcher = SpatialPatcher(
            geometry=SpatialRectangular(size=(2, 2)),
            sampler=SpatialRegularStride(step=2),
            window=SpatialBoxcar(),
            aggregation=SpatialOverlapAdd(),
        )
        field = self._field()
        seen: list[int] = []

        def _consume() -> None:
            for i, _patch_obj in enumerate(patcher.split(field, max_in_flight=2)):
                seen.append(i)

        worker = threading.Thread(target=_consume, daemon=True)
        worker.start()
        worker.join(timeout=10.0)
        assert not worker.is_alive(), "split(max_in_flight=2) stalled"
        assert len(seen) == 16  # 8x8 field, 2x2 tiles, stride 2

    def test_explicit_close_releases_slot_immediately(self) -> None:
        patcher = SpatialPatcher(
            geometry=SpatialRectangular(size=(2, 2)),
            sampler=SpatialRegularStride(step=2),
            window=SpatialBoxcar(),
            aggregation=SpatialOverlapAdd(),
        )
        field = self._field()
        count = 0
        for patch in patcher.split(field, max_in_flight=1):
            with patch:
                count += 1
        assert count == 16
