"""Tests for reference runner helpers."""

from __future__ import annotations

import json
import multiprocessing
import threading
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from georeader.geotensor import GeoTensor

from geopatcher import (
    PatchJournal,
    RasterField,
    SpatialBoxcar,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRectangular,
    SpatialRegularStride,
)
from geopatcher.runners import _resolve_mp_context, parallel_map


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


@dataclass
class _FailingTileField:
    """Raster field whose read of one window raises ``OSError``."""

    inner: RasterField
    bad_offset: tuple[int, int] = (8, 8)

    @property
    def domain(self) -> Any:
        return self.inner.domain

    def select(self, indexer: Any) -> Any:
        if (indexer.row_off, indexer.col_off) == self.bad_offset:
            raise OSError("tile read failed")
        return self.inner.select(indexer)

    def with_data(self, array: Any) -> Any:
        return self.inner.with_data(array)


@dataclass
class _FailingTileBatchedField(_FailingTileField):
    """Same, behind the ``select_many`` fast path."""

    def select_many(self, indexers: list[Any]) -> list[Any]:
        return [self.select(i) for i in indexers]


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("policy", ["skip", "mask", "raise"])
def test_parallel_map_respects_patcher_on_error(
    patcher: SpatialPatcher, field: RasterField, policy: str, batched: bool
) -> None:
    """#196: read failures go through the patcher's policy on every path."""
    patcher = replace(patcher, on_error=policy)
    cls = _FailingTileBatchedField if batched else _FailingTileField
    failing = cls(inner=field)
    anchors = patcher.anchors(field)

    if policy == "raise":
        with pytest.raises(OSError, match="tile read failed"):
            parallel_map(patcher, failing, _double, n_workers=2)
        return

    out = parallel_map(patcher, failing, _double, n_workers=2)

    assert [(e.anchor, e.kind) for e in patcher.errors] == [((8, 8), "OSError")]
    if policy == "skip":
        assert [p.anchor for p in out] == [a for a in anchors if a != (8, 8)]
    else:
        assert [p.anchor for p in out] == anchors
        masked = {p.anchor: p for p in out}[(8, 8)]
        assert np.isnan(masked.data).all()


def test_parallel_map_journal_resumes(
    patcher: SpatialPatcher, field: RasterField, tmp_path: Path
) -> None:
    """#196: committed anchors are skipped; every finished patch is committed."""
    anchors = patcher.anchors(field)
    journal = PatchJournal(str(tmp_path / "run.jsonl"))
    for anchor in anchors[:5]:  # a previous, interrupted run
        journal.commit(anchor, status="ok", runtime_s=0.0)
    seen: list[float] = []

    def record(data: Any) -> np.ndarray:
        seen.append(float(np.asarray(data)[0, 0]))
        return _double(data)

    out = parallel_map(patcher, field, record, n_workers=2, journal=journal)

    assert [p.anchor for p in out] == anchors[5:]
    assert len(seen) == len(anchors) - 5
    reopened = PatchJournal(str(tmp_path / "run.jsonl"))
    assert all(reopened.has(anchor) for anchor in anchors)
    lines = (tmp_path / "run.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    assert all(row["status"] == "ok" and row["runtime_s"] >= 0 for row in rows)

    seen.clear()
    assert parallel_map(patcher, field, record, journal=reopened) == []
    assert seen == []


def test_parallel_map_journal_records_operator_errors(
    patcher: SpatialPatcher, field: RasterField, tmp_path: Path
) -> None:
    journal = PatchJournal(str(tmp_path / "run.jsonl"))

    def fail_first_patch(data: Any) -> np.ndarray:
        if np.asarray(data)[0, 0] == 0:
            raise ValueError("boom")
        return np.asarray(data)

    with pytest.warns(RuntimeWarning, match="skipped patch"):
        parallel_map(patcher, field, fail_first_patch, journal=journal, on_error="skip")

    first = patcher.anchors(field)[0]
    assert not journal.has(first)
    lines = (tmp_path / "run.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    errors = [row for row in rows if row["status"] == "error"]
    assert [row["error"] for row in errors] == ["ValueError: boom"]


def test_parallel_map_fail_fast_cancels_queued_patches(
    patcher: SpatialPatcher, field: RasterField
) -> None:
    """#196: an operator error cancels queued work instead of running it all."""
    calls: list[float] = []

    def fail_first(data: Any) -> np.ndarray:
        value = float(np.asarray(data)[0, 0])
        calls.append(value)
        if value == 0:
            raise ValueError("boom")
        time.sleep(0.01)
        return np.asarray(data)

    with pytest.raises(ValueError, match="boom"):
        parallel_map(patcher, field, fail_first, n_workers=1)

    # 16 patches; the default window is 2 * n_workers submitted patches.
    assert calls[0] == 0
    assert len(calls) <= 2


def test_parallel_map_max_in_flight_validation(
    patcher: SpatialPatcher, field: RasterField
) -> None:
    with pytest.raises(ValueError, match="max_in_flight must be >= 1"):
        parallel_map(patcher, field, _double, max_in_flight=0)


def test_default_mp_context_never_forks() -> None:
    """#196: the process pool uses forkserver (or spawn), never fork."""
    method = _resolve_mp_context(None).get_start_method()
    methods = multiprocessing.get_all_start_methods()
    expected = "forkserver" if "forkserver" in methods else "spawn"
    assert method == expected
    assert _resolve_mp_context("spawn").get_start_method() == "spawn"
    ctx = multiprocessing.get_context("spawn")
    assert _resolve_mp_context(ctx) is ctx


def test_process_backend_does_not_fork_a_threaded_process(
    patcher: SpatialPatcher, field: RasterField
) -> None:
    """No ``fork()`` DeprecationWarning even while other threads are alive."""
    stop = threading.Event()
    busy = threading.Thread(target=stop.wait, daemon=True)
    busy.start()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            patches = parallel_map(
                patcher, field, _double, n_workers=2, backend="process"
            )
    finally:
        stop.set()
        busy.join()

    # `os.fork()` reports the warning without raising it, so record it.
    assert not [w for w in caught if "fork" in str(w.message)]
    assert [p.anchor for p in patches] == patcher.anchors(field)
    for patch in patches:
        r, c = patch.anchor
        np.testing.assert_array_equal(
            patch.data, _double(field.reader.values)[r : r + 8, c : c + 8]
        )


def _georeferenced_double(data: Any) -> Any:
    # Runs in a worker process: the chip must arrive georeferenced.
    if not isinstance(data, GeoTensor) or data.transform is None:
        raise TypeError(f"chip lost its georeferencing: {type(data)}")
    return data * 2


@pytest.mark.parametrize("backend", ["thread", "process"])
def test_process_backend_keeps_geotensor_georeferencing(
    patcher: SpatialPatcher, field: RasterField, backend: str
) -> None:
    """GeoTensor chips and outputs keep transform / CRS through the pool.

    georeader's GeoTensor pickles as a bare-metadata ndarray subclass, so
    the runner ships it in a wire form and rebuilds it on both sides.
    """
    expected = {p.anchor: p.data for p in patcher.split(field)}

    out = parallel_map(
        patcher, field, _georeferenced_double, n_workers=2, backend=backend
    )

    assert len(out) == len(expected)
    for patch in out:
        want = expected[patch.anchor]
        assert isinstance(patch.data, GeoTensor)
        assert patch.data.transform == want.transform
        assert patch.data.crs == want.crs
        assert patch.data.fill_value_default == want.fill_value_default
        np.testing.assert_array_equal(np.asarray(patch.data), np.asarray(want) * 2)
    merged = patcher.merge(out, field.domain)
    np.testing.assert_array_equal(merged, np.asarray(field.reader.values) * 2)


def test_process_backend_accepts_explicit_spawn(
    patcher: SpatialPatcher, field: RasterField
) -> None:
    patches = parallel_map(
        patcher, field, _double, n_workers=1, backend="process", mp_context="spawn"
    )
    assert [p.anchor for p in patches] == patcher.anchors(field)
