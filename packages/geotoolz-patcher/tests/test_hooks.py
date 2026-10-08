"""Tests for patcher observability hooks."""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest
from _helpers import ArrField as _ArrField

from geopatcher import RasterField, SpatialPatcher, spatial, temporal
from geopatcher._src.hooks import _positional_arity
from geopatcher.observe import PatcherHook


@pytest.fixture
def patcher() -> SpatialPatcher:
    return SpatialPatcher(
        geometry=spatial.geometry.Rectangular(size=(16, 16)),
        sampler=spatial.sampler.RegularStride(step=16),
        window=spatial.window.Boxcar(),
        aggregation=spatial.aggregation.OverlapAdd(),
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
    from geopatcher._src.temporal.patcher import TemporalPatcher

    mf = MatchedField(
        primary=_ArrField(np.arange(100, dtype=np.float64)),
        secondaries={"s2": _ArrField(np.arange(100, dtype=np.float64) * 2)},
        coreg={"s2": lambda raw, prim: raw},
    )
    primary = TemporalPatcher(
        geometry=temporal.geometry.FixedLookback(length=5),
        sampler=temporal.sampler.RegularStride(step=10),
        window=temporal.window.CausalBoxcar(),
        aggregation=temporal.aggregation.Mean(),
    )
    mtp = MatchedTemporalPatcher(primary=primary)
    hook = RecordingHook()

    patches = list(mtp.split(mf, hooks=[hook]))

    # Anchors 10..90: anchor 0's 5-step lookback overflows and is dropped.
    assert len(patches) == 9
    assert hook.events[0] == ("split_start", 9)
    assert hook.events[-1] == ("split_end", None)
    assert [name for name, _ in hook.events].count("patch_done") == 9


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
    from geopatcher._src.temporal.patcher import TemporalPatcher

    mf = MatchedField(primary=field)
    temporal_patcher = TemporalPatcher(
        geometry=temporal.geometry.FixedLookback(length=2),
        sampler=temporal.sampler.RegularStride(step=4),
        window=temporal.window.CausalBoxcar(),
        aggregation=temporal.aggregation.Mean(),
    )
    stp = SpatioTemporalPatcher(spatial=patcher, temporal=temporal_patcher, time_axis=0)
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


def test_hook_objects_are_collectable_after_split(
    field: RasterField, patcher: SpatialPatcher
) -> None:
    """The arity cache must not pin hook objects (spans, progress bars, …).

    Regression for #195: ``_positional_arity`` was an ``lru_cache`` keyed on
    the *bound* method, so up to 256 hook instances stayed reachable via
    ``callback.__self__`` long after their split finished.
    """
    hook = RecordingHook()
    ref = weakref.ref(hook)
    list(patcher.split(field, hooks=[hook]))
    assert hook.events[0] == ("split_start", 16)
    del hook
    gc.collect()
    assert ref() is None, "hook object retained after split"


def test_positional_arity_trims_bound_methods_and_callables() -> None:
    class Hook:
        def on_patch_start(self, anchor: object) -> None: ...

        def on_patch_done(self, *args: object) -> None: ...

    class CallableHook:
        def __call__(self, anchor: object, coord_value: object = None) -> None: ...

    assert _positional_arity(Hook().on_patch_start) == 1
    assert _positional_arity(Hook().on_patch_done) >= 4
    assert _positional_arity(CallableHook()) == 2
    assert _positional_arity(lambda a, b: None) == 2


@pytest.mark.parametrize("policy", ["skip", "mask", "retry"])
def test_on_error_receives_original_exception(field: RasterField, policy: str) -> None:
    """A swallowed read failure reaches on_error as the raised exception (#188)."""
    boom = OSError("tile 0 unreadable")

    class Flaky:
        domain = field.domain

        def select(self, window: object) -> object:
            raise boom

    class ErrorHook:
        def __init__(self) -> None:
            self.calls: list[tuple[object, Exception]] = []

        def on_error(self, anchor: object, exc: Exception) -> None:
            self.calls.append((anchor, exc))

    patcher = SpatialPatcher(
        geometry=spatial.geometry.Rectangular(size=(32, 32)),
        sampler=spatial.sampler.Explicit(anchors_=[(0, 0)]),
        window=spatial.window.Boxcar(),
        aggregation=spatial.aggregation.OverlapAdd(),
        on_error=policy,
        max_retries=1,
    )
    hook = ErrorHook()
    list(patcher.split(Flaky(), hooks=[hook]))
    assert hook.calls
    assert all(anchor == (0, 0) and exc is boom for anchor, exc in hook.calls)
    assert len(hook.calls) == len(patcher.errors)
