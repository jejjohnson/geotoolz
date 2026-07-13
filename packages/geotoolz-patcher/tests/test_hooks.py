"""Tests for patcher observability hooks."""

from __future__ import annotations

import numpy as np
import pytest
from _helpers import ArrField as _ArrField

from geopatcher import (
    PatcherHook,
    RasterField,
    SpatialBoxcar,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRectangular,
    SpatialRegularStride,
)


@pytest.fixture
def patcher() -> SpatialPatcher:
    return SpatialPatcher(
        geometry=SpatialRectangular(size=(16, 16)),
        sampler=SpatialRegularStride(step=16),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )


class RecordingHook:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []
        self.bytes_: list[int] = []
        self.runtimes: list[float] = []

    def on_split_start(self, n_anchors: int) -> None:
        self.events.append(("split_start", n_anchors))

    def on_patch_start(self, anchor: object) -> None:
        self.events.append(("patch_start", anchor))

    def on_patch_done(self, anchor: object, runtime_s: float, bytes_: int) -> None:
        self.events.append(("patch_done", anchor))
        self.bytes_.append(bytes_)
        self.runtimes.append(runtime_s)

    def on_split_end(self) -> None:
        self.events.append(("split_end", None))


def test_spatial_split_dispatches_hooks_in_order(
    field: RasterField, patcher: SpatialPatcher
) -> None:
    hook = RecordingHook()

    patches = list(patcher.split(field, hooks=[hook]))

    assert len(patches) == 16
    assert hook.events[0] == ("split_start", 16)
    assert hook.events[-1] == ("split_end", None)
    assert [name for name, _ in hook.events].count("patch_start") == 16
    assert [name for name, _ in hook.events].count("patch_done") == 16
    assert all(runtime_s >= 0 for runtime_s in hook.runtimes)
    assert all(bytes_ > 0 for bytes_ in hook.bytes_)


class MergeHook:
    def __init__(self) -> None:
        self.events: list[tuple[str, int]] = []

    def on_merge_start(self, n_patches: int) -> None:
        self.events.append(("merge_start", n_patches))

    def on_merge_end(self, output_bytes: int) -> None:
        self.events.append(("merge_end", output_bytes))


def test_spatial_merge_dispatches_hooks(
    field: RasterField, patcher: SpatialPatcher
) -> None:
    hook = MergeHook()
    patches = list(patcher.split(field))

    patcher.merge(patches, field.reader, hooks=[hook])

    assert hook.events[0] == ("merge_start", 16)
    assert hook.events[1][0] == "merge_end"
    assert hook.events[1][1] > 0


class FailingHook:
    def on_patch_start(self, anchor: object) -> None:
        raise RuntimeError(f"bad hook for {anchor!r}")


def test_hook_errors_warn_without_aborting_split(
    field: RasterField, patcher: SpatialPatcher
) -> None:
    with pytest.warns(RuntimeWarning, match="PatcherHook.on_patch_start"):
        patches = list(patcher.split(field, hooks=[FailingHook()]))

    assert len(patches) == 16


class ErrorRecordingHook:
    def __init__(self) -> None:
        self.errors: list[tuple[object, Exception]] = []

    def on_error(self, anchor: object, exc: Exception) -> None:
        self.errors.append((anchor, exc))


class FailingField:
    def __init__(self, domain: object) -> None:
        self.domain = domain

    def select(self, indexer: object) -> object:
        raise ValueError("boom")

    def with_data(self, array: object) -> object:
        return array


def test_patch_errors_dispatch_on_error(
    field: RasterField, patcher: SpatialPatcher
) -> None:
    hook = ErrorRecordingHook()

    with pytest.raises(ValueError, match="boom"):
        list(patcher.split(FailingField(field.domain), hooks=[hook]))

    assert len(hook.errors) == 1
    assert hook.errors[0][0] is not None
    assert isinstance(hook.errors[0][1], ValueError)


def test_protocol_is_public() -> None:
    assert PatcherHook.__name__ == "PatcherHook"


# ---------------------------------------------------------------------------
# Matched patchers — verify `hooks=` is plumbed through Phase 4 surfaces.
# ---------------------------------------------------------------------------


def test_matched_temporal_split_forwards_hooks() -> None:
    """`MatchedTemporalPatcher.split` should forward hooks to the primary."""
    from geopatcher._src.matched import MatchedField, MatchedTemporalPatcher
    from geopatcher._src.time.aggregation import TemporalMean
    from geopatcher._src.time.geometry import TemporalFixedLookback
    from geopatcher._src.time.patcher import TemporalPatcher
    from geopatcher._src.time.sampler import TemporalRegularStride
    from geopatcher._src.time.window import TemporalCausalBoxcar

    mf = MatchedField(
        primary=_ArrField(np.arange(100, dtype=np.float64)),
        secondaries={"s2": _ArrField(np.arange(100, dtype=np.float64) * 2)},
        coreg={"s2": lambda raw, prim: raw},
    )
    primary = TemporalPatcher(
        geometry=TemporalFixedLookback(length=5),
        sampler=TemporalRegularStride(step=10),
        window=TemporalCausalBoxcar(),
        aggregation=TemporalMean(),
    )
    mtp = MatchedTemporalPatcher(primary=primary)
    hook = RecordingHook()

    patches = list(mtp.split(mf, hooks=[hook]))

    assert len(patches) == 10
    assert hook.events[0] == ("split_start", 10)
    assert hook.events[-1] == ("split_end", None)
    assert [name for name, _ in hook.events].count("patch_done") == 10


def test_matched_spatial_split_forwards_hooks(
    field: RasterField, patcher: SpatialPatcher
) -> None:
    """`MatchedSpatialPatcher.split` should forward hooks to the primary."""
    from geopatcher._src.matched import MatchedField, MatchedSpatialPatcher

    mf = MatchedField(primary=field)
    msp = MatchedSpatialPatcher(primary=patcher)
    hook = RecordingHook()

    patches = list(msp.split(mf, hooks=[hook]))

    assert len(patches) == 16
    assert hook.events[0] == ("split_start", 16)
    assert hook.events[-1] == ("split_end", None)


def test_matched_spatiotemporal_split_dispatches_hooks(
    field: RasterField, patcher: SpatialPatcher
) -> None:
    """`MatchedSpatioTemporalPatcher.split` should emit per-pair hook events."""
    from geopatcher._src.matched import (
        MatchedField,
        MatchedSpatioTemporalPatcher,
    )
    from geopatcher._src.spatial_time import SpatioTemporalPatcher
    from geopatcher._src.time.aggregation import TemporalMean
    from geopatcher._src.time.geometry import TemporalFixedLookback
    from geopatcher._src.time.patcher import TemporalPatcher
    from geopatcher._src.time.sampler import TemporalRegularStride
    from geopatcher._src.time.window import TemporalCausalBoxcar

    mf = MatchedField(primary=field)
    temporal = TemporalPatcher(
        geometry=TemporalFixedLookback(length=2),
        sampler=TemporalRegularStride(step=4),
        window=TemporalCausalBoxcar(),
        aggregation=TemporalMean(),
    )
    stp = SpatioTemporalPatcher(spatial=patcher, temporal=temporal, time_axis=0)
    mstp = MatchedSpatioTemporalPatcher(primary=stp)
    hook = RecordingHook()

    patches = list(mstp.split(mf, hooks=[hook]))

    assert len(patches) >= 1
    assert hook.events[0][0] == "split_start"
    assert hook.events[-1] == ("split_end", None)
    assert [name for name, _ in hook.events].count("patch_done") == len(patches)


def test_hook_error_is_exception_not_baseexception(
    field: RasterField, patcher: SpatialPatcher
) -> None:
    """KeyboardInterrupt from a hook must still abort — only Exception is swallowed."""

    class InterruptHook:
        def on_patch_start(self, anchor: object) -> None:
            raise KeyboardInterrupt("user pressed ctrl+c")

    with pytest.raises(KeyboardInterrupt):
        list(patcher.split(field, hooks=[InterruptHook()]))
