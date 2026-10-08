"""Tests for spatial two-pass / global-context helpers."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import pytest

from geopatcher import Patch, RasterField, SpatialPatcher, spatial
from geopatcher.observe import get_strict, set_strict


@pytest.fixture
def field(raster_field_factory) -> RasterField:
    return raster_field_factory(16, dtype=np.float64)


@pytest.fixture
def patcher() -> SpatialPatcher:
    return SpatialPatcher(
        geometry=spatial.geometry.Rectangular(size=(4, 4)),
        sampler=spatial.sampler.RegularStride(step=4),
        window=spatial.window.Boxcar(),
        aggregation=spatial.aggregation.OverlapAdd(),
    )


def test_reduce_mean_std_matches_numpy(
    patcher: SpatialPatcher, field: RasterField
) -> None:
    stats = patcher.reduce(field, agg=spatial.aggregation.MeanStd())
    data = np.asarray(field.reader)

    assert stats["mean"] == pytest.approx(float(np.mean(data)))
    assert stats["std"] == pytest.approx(float(np.std(data, ddof=1)))


def test_reduce_min_max_matches_numpy(
    patcher: SpatialPatcher, field: RasterField
) -> None:
    stats = patcher.reduce(field, agg=spatial.aggregation.MinMax())
    data = np.asarray(field.reader)

    assert stats == {"min": float(np.min(data)), "max": float(np.max(data))}


def test_two_pass_applies_global_stats(
    patcher: SpatialPatcher, field: RasterField
) -> None:
    out = patcher.two_pass(
        field,
        reduce_with=spatial.aggregation.MeanStd(),
        apply=lambda data, stats: (np.asarray(data) - stats["mean"]) / stats["std"],
    )

    np.testing.assert_allclose(np.mean(out), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.std(out, ddof=1), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# #194: reduce / two_pass go through `split`
# ---------------------------------------------------------------------------


class _Recorder:
    """`PatcherHook` that logs every event it sees."""

    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    def on_split_start(self, n_anchors: int) -> None:
        self.events.append(("split_start", n_anchors))

    def on_patch_done(self, anchor: Any, seconds: float, nbytes: int) -> None:
        self.events.append(("patch_done", anchor))

    def on_merge_start(self, n_patches: int) -> None:
        self.events.append(("merge_start", n_patches))

    def on_merge_end(self, output_bytes: int) -> None:
        self.events.append(("merge_end", output_bytes))


class _FailingField:
    """Delegating `Field` whose read of one window raises `OSError`."""

    def __init__(self, inner: RasterField, bad: tuple[int, int]) -> None:
        self.inner = inner
        self.bad = bad

    @property
    def domain(self) -> Any:
        return self.inner.domain

    def select(self, window: Any) -> Any:
        if (int(window.row_off), int(window.col_off)) == self.bad:
            raise OSError("tile unavailable")
        return self.inner.select(window)

    def with_data(self, array: Any) -> Any:
        return self.inner.with_data(array)


def test_reduce_dispatches_hooks_for_every_patch(
    patcher: SpatialPatcher, field: RasterField
) -> None:
    hook = _Recorder()

    patcher.reduce(field, spatial.aggregation.MinMax(), hooks=[hook])

    done = [anchor for kind, anchor in hook.events if kind == "patch_done"]
    assert done == patcher.anchors(field)
    kinds = [kind for kind, _ in hook.events]
    assert kinds[0] == "merge_start"
    assert "split_start" in kinds
    assert kinds[-1] == "merge_end"


def test_reduce_applies_on_error_skip(field: RasterField) -> None:
    patcher = SpatialPatcher(
        geometry=spatial.geometry.Rectangular(size=(4, 4)),
        sampler=spatial.sampler.RegularStride(step=4),
        window=spatial.window.Boxcar(),
        aggregation=spatial.aggregation.OverlapAdd(),
        on_error="skip",
    )
    flaky = _FailingField(field, bad=(12, 12))  # holds the global max

    stats = patcher.reduce(flaky, spatial.aggregation.MinMax())

    data = np.asarray(field.reader).copy()
    data[12:, 12:] = np.nan  # the skipped tile
    assert stats == {"min": float(np.nanmin(data)), "max": float(np.nanmax(data))}
    assert [e.anchor for e in patcher.errors] == [(12, 12)]


def test_reduce_honours_journal(patcher: SpatialPatcher, field: RasterField) -> None:
    done = {(0, 0), (0, 4)}

    class _Journal:
        def has(self, anchor: Any) -> bool:
            return anchor in done

    hook = _Recorder()
    patcher.reduce(
        field, spatial.aggregation.MinMax(), hooks=[hook], journal=_Journal()
    )

    seen = {anchor for kind, anchor in hook.events if kind == "patch_done"}
    assert seen == set(patcher.anchors(field)) - done


def test_reduce_and_two_pass_release_backpressure(
    patcher: SpatialPatcher, field: RasterField
) -> None:
    # Aggregations never close patches; without close-on-advance the split
    # producer blocks forever on the single in-flight slot.
    results: dict[str, Any] = {}

    def run() -> None:
        results["reduce"] = patcher.reduce(
            field, spatial.aggregation.MinMax(), max_in_flight=1
        )
        results["two_pass"] = patcher.two_pass(
            field,
            reduce_with=spatial.aggregation.MeanStd(),
            apply=lambda data, stats: np.asarray(data) - stats["mean"],
            max_in_flight=1,
            prefetch=2,
        )

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=30)
    assert not worker.is_alive(), "reduce/two_pass deadlocked on backpressure"
    data = np.asarray(field.reader)
    assert results["reduce"] == {"min": float(data.min()), "max": float(data.max())}
    np.testing.assert_allclose(results["two_pass"], data - data.mean())


def test_two_pass_maps_patches_with_patch_with_data(
    patcher: SpatialPatcher, field: RasterField, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Any] = []
    original = Patch.with_data

    def spy(self: Patch, data: Any) -> Patch:
        calls.append(self.anchor)
        return original(self, data)

    monkeypatch.setattr(Patch, "with_data", spy)

    patcher.two_pass(
        field,
        reduce_with=spatial.aggregation.MeanStd(),
        apply=lambda data, stats: np.asarray(data) - stats["mean"],
    )

    assert calls == patcher.anchors(field)


def test_two_pass_places_both_passes_on_one_anchor_draw(field: RasterField) -> None:
    # An unseeded sampler re-draws on every `anchors()` call; both passes
    # must still see the same anchors (drawn once).
    patcher = SpatialPatcher(
        geometry=spatial.geometry.Rectangular(size=(4, 4)),
        sampler=spatial.sampler.Random(n_samples=6),
        window=spatial.window.Boxcar(),
        aggregation=spatial.aggregation.OverlapAdd(),
    )
    hook = _Recorder()

    patcher.two_pass(
        field,
        reduce_with=spatial.aggregation.MinMax(),
        apply=lambda data, stats: data,
        hooks=[hook],
    )

    anchors = [anchor for kind, anchor in hook.events if kind == "patch_done"]
    assert len(anchors) == 12
    assert anchors[:6] == anchors[6:]


def test_reduce_runs_the_strict_streaming_check(
    patcher: SpatialPatcher, field: RasterField
) -> None:
    original = get_strict()
    set_strict(True)
    try:
        with pytest.raises(RuntimeError, match="Median has streaming_safe"):
            patcher.reduce(field, spatial.aggregation.Median())
    finally:
        set_strict(original)
