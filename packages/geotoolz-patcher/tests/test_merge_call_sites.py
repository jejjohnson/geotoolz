"""Merge entry points: warning attribution and async streaming (#194).

* The ``streaming_safe = False`` warning names the user's line (not a
  line inside geopatcher) from every merge entry point.
* `amerge` feeds an async patch stream into the aggregation one patch at
  a time instead of materialising it first.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterable
from typing import Any, ClassVar

import numpy as np
import pytest

from geopatcher import AsyncSpatialPatcher, RasterField, SpatialPatcher, spatial
from geopatcher._src.matched import MatchedField, MatchedSpatialPatcher


def _patcher(
    aggregation: spatial.aggregation.Aggregation, cls: type = SpatialPatcher
) -> Any:
    return cls(
        geometry=spatial.geometry.Rectangular(size=(4, 4)),
        sampler=spatial.sampler.RegularStride(step=4),
        window=spatial.window.Boxcar(),
        aggregation=aggregation,
    )


async def _astream(patches: Iterable[Any]) -> AsyncIterator[Any]:
    for patch in patches:
        yield patch


async def _amerge(field: RasterField) -> Any:
    # The user's coroutine is the frame the warning must name.
    patcher = _patcher(spatial.aggregation.Median())
    return await patcher.amerge(_astream(patcher.split(field)), field.domain)


def _matched_merge(field: RasterField) -> Any:
    mfield = MatchedField(
        primary=field,
        secondaries={"sec": field},
        coreg={"sec": lambda raw, prim: raw},
    )
    mpatcher = MatchedSpatialPatcher(
        primary=_patcher(spatial.aggregation.OverlapAdd()),
        secondary_aggregators={"sec": spatial.aggregation.Median()},
    )
    return mpatcher.merge(mpatcher.split(mfield), mfield)


_CALL_SITES: dict[str, Callable[[RasterField], Any]] = {
    "merge": lambda f: _patcher(spatial.aggregation.Median()).merge(
        _patcher(spatial.aggregation.Median()).split(f), f.domain
    ),
    "merge_to_field": lambda f: _patcher(spatial.aggregation.Median()).merge_to_field(
        _patcher(spatial.aggregation.Median()).split(f), f
    ),
    "reduce": lambda f: _patcher(spatial.aggregation.OverlapAdd()).reduce(
        f, spatial.aggregation.Median()
    ),
    "two_pass": lambda f: _patcher(spatial.aggregation.OverlapAdd()).two_pass(
        f,
        reduce_with=spatial.aggregation.MinMax(),
        apply=lambda data, stats: data,
        aggregation=spatial.aggregation.Median(),
    ),
    "amerge": lambda f: asyncio.run(_amerge(f)),
    "async_patcher.merge": lambda f: _patcher(
        spatial.aggregation.Median(), AsyncSpatialPatcher
    ).merge(_patcher(spatial.aggregation.Median()).split(f), f.domain),
    "matched.merge": _matched_merge,
}


@pytest.mark.parametrize("call", list(_CALL_SITES.values()), ids=list(_CALL_SITES))
def test_streaming_warning_points_at_the_caller(
    call: Callable[[RasterField], Any], raster_field_factory: Any
) -> None:
    field = raster_field_factory(8)

    with pytest.warns(RuntimeWarning, match="streaming_safe") as record:
        call(field)

    assert len(record) == 1
    assert record[0].filename == __file__


# ---------------------------------------------------------------------------
# amerge streams
# ---------------------------------------------------------------------------


class _Counting(spatial.aggregation.Aggregation):
    """Streaming aggregation that logs how many patches had been produced
    each time it consumed one."""

    streaming_safe: ClassVar[bool] = True

    def __init__(self, produced: list[int]) -> None:
        self.produced = produced
        self.seen: list[int] = []

    def merge(self, patches: Iterable[Any], domain: Any) -> int:
        for _ in patches:
            self.seen.append(self.produced[0])
        return len(self.seen)


@pytest.mark.parametrize("cls", [SpatialPatcher, AsyncSpatialPatcher])
def test_amerge_consumes_async_stream_incrementally(
    cls: type, raster_field_factory: Any
) -> None:
    field = raster_field_factory(8)
    patches = list(_patcher(spatial.aggregation.OverlapAdd()).split(field))
    produced = [0]

    async def counting() -> AsyncIterator[Any]:
        for patch in patches:
            produced[0] += 1
            yield patch
            await asyncio.sleep(0)

    agg = _Counting(produced)
    out = asyncio.run(_patcher(agg, cls).amerge(counting(), field.domain))

    assert out == len(patches)
    # Each patch was folded before the next one was produced — the first
    # long before the stream ended.
    assert agg.seen == list(range(1, len(patches) + 1))


def test_amerge_streamed_result_matches_merge(raster_field_factory: Any) -> None:
    field = raster_field_factory(8)
    patcher = _patcher(spatial.aggregation.OverlapAdd())

    streamed = asyncio.run(patcher.amerge(_astream(patcher.split(field)), field.domain))

    np.testing.assert_array_equal(
        streamed, patcher.merge(patcher.split(field), field.domain)
    )


def test_amerge_propagates_stream_errors(raster_field_factory: Any) -> None:
    field = raster_field_factory(8)
    patcher = _patcher(spatial.aggregation.OverlapAdd())

    async def broken() -> AsyncIterator[Any]:
        for i, patch in enumerate(patcher.split(field)):
            if i == 2:
                raise OSError("tile 2 unavailable")
            yield patch

    with pytest.raises(OSError, match="tile 2"):
        asyncio.run(patcher.amerge(broken(), field.domain))


def test_amerge_closes_stream_when_aggregation_fails(
    raster_field_factory: Any,
) -> None:
    field = raster_field_factory(8)
    closed: list[bool] = []

    class _Boom(spatial.aggregation.Aggregation):
        streaming_safe: ClassVar[bool] = True

        def merge(self, patches: Iterable[Any], domain: Any) -> Any:
            next(iter(patches))
            raise ValueError("boom")

    async def stream() -> AsyncIterator[Any]:
        try:
            for patch in _patcher(spatial.aggregation.OverlapAdd()).split(field):
                yield patch
        finally:
            closed.append(True)

    with pytest.raises(ValueError, match="boom"):
        asyncio.run(_patcher(_Boom()).amerge(stream(), field.domain))
    assert closed == [True]
