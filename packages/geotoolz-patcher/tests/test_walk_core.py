"""The one anchor-walk core (#188): every split path runs the same read pipeline.

`SpatialPatcher.split` was the reference; `asplit` (both spatial patchers)
and the coupled / product `SpatioTemporalPatcher` splits used to bypass
parts of it (``on_error``, journal, hooks, backpressure, the boundary /
weights pipeline). These tests pin the parity.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor

from geopatcher import (
    AsyncSpatialPatcher,
    RasterField,
    SpatialPatcher,
    SpatioTemporalPatcher,
    TemporalPatcher,
    spatial,
    temporal,
)
from geopatcher._src.hooks import UNKNOWN_TOTAL


# -- fixtures ----------------------------------------------------------------


class _Flaky:
    """A raster field whose reads fail at the given window origins."""

    def __init__(self, inner: RasterField, bad: set[tuple[int, int]]) -> None:
        self.inner = inner
        self.bad = bad
        self.reads: list[tuple[int, int]] = []

    @property
    def domain(self) -> Any:
        return self.inner.domain

    def select(self, window: Any) -> Any:
        origin = (int(window.row_off), int(window.col_off))
        self.reads.append(origin)
        if origin in self.bad:
            raise OSError(f"tile {origin} unreadable")
        return self.inner.select(window)

    async def aselect(self, window: Any) -> Any:
        await asyncio.sleep(0)
        return self.select(window)


class _Journal:
    def __init__(self, keys: list[Any]) -> None:
        self.keys = list(keys)

    def has(self, key: Any) -> bool:
        return key in self.keys


class _Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []

    def on_split_start(self, n: int) -> None:
        self.events.append(("split_start", n))

    def on_patch_start(self, anchor: Any) -> None:
        self.events.append(("start", anchor))

    def on_patch_done(self, anchor: Any, runtime_s: float, bytes_: int) -> None:
        self.events.append(("done", anchor))

    def on_patch_skipped(self, anchor: Any) -> None:
        self.events.append(("skipped", anchor))

    def on_error(self, anchor: Any, exc: Exception) -> None:
        self.events.append(("error", anchor, exc))

    def on_split_end(self) -> None:
        self.events.append(("split_end",))

    def kinds(self, kind: str) -> list[Any]:
        return [e[1] for e in self.events if e[0] == kind]


def _raster(shape: tuple[int, ...]) -> RasterField:
    arr = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    return RasterField(
        GeoTensor(values=arr, transform=rasterio.Affine.identity(), crs="EPSG:32630")
    )


def _spatial(cls: type = SpatialPatcher, **kwargs: Any) -> Any:
    kwargs.setdefault("sampler", spatial.sampler.RegularStride(step=8))
    return cls(
        geometry=spatial.geometry.Rectangular(size=(8, 8)),
        window=spatial.window.Boxcar(),
        aggregation=spatial.aggregation.OverlapAdd(),
        **kwargs,
    )


def _temporal(geometry: Any = None) -> TemporalPatcher:
    return TemporalPatcher(
        geometry=geometry or temporal.geometry.FixedLookback(length=2),
        sampler=temporal.sampler.RegularStride(step=1),
        window=temporal.window.CausalBoxcar(),
        aggregation=temporal.aggregation.Mean(),
    )


def _coupled(pairs: list[Any], **spatial_kwargs: Any) -> SpatioTemporalPatcher:
    return SpatioTemporalPatcher(
        spatial=_spatial(
            sampler=spatial.sampler.Explicit(anchors_=pairs), **spatial_kwargs
        ),
        temporal=_temporal(),
        coupling="coupled",
    )


async def _collect(agen: Any) -> list[Any]:
    return [p async for p in agen]


# -- asplit ------------------------------------------------------------------


@pytest.mark.parametrize("cls", [SpatialPatcher, AsyncSpatialPatcher])
def test_asplit_honours_journal_and_hooks_like_split(cls: type) -> None:
    field = _raster((16, 16))
    flaky = _Flaky(field, bad={(8, 0)})
    patcher = _spatial(cls, on_error="skip")
    sync = _spatial(SpatialPatcher, on_error="skip")
    journal = _Journal([(0, 8)])

    hook_a, hook_s = _Recorder(), _Recorder()
    out_a = asyncio.run(_collect(patcher.asplit(flaky, [hook_a], journal=journal)))
    out_s = list(sync.split(_Flaky(field, bad={(8, 0)}), [hook_s], journal=journal))

    assert [p.anchor for p in out_a] == [p.anchor for p in out_s] == [(0, 0), (8, 8)]
    assert hook_a.kinds("skipped") == [(0, 8)]
    assert [e[:2] for e in hook_a.events] == [e[:2] for e in hook_s.events]
    assert (0, 8) not in flaky.reads  # journaled: never read


def test_asplit_backpressure_bounds_bytes() -> None:
    patcher = _spatial(on_error="skip")
    with pytest.raises(ValueError, match="max_in_flight_bytes"):
        asyncio.run(
            _collect(
                patcher.asplit(_Flaky(_raster((16, 16)), set()), max_in_flight_bytes=1)
            )
        )


# -- coupled spatio-temporal split -------------------------------------------


def test_coupled_split_honours_on_error_and_hooks() -> None:
    field = _Flaky(_raster((8, 16, 16)), bad={(0, 8)})
    stp = _coupled([((0, 0), 3), ((0, 8), 4), ((8, 8), 5)], on_error="skip")
    hook = _Recorder()

    patches = list(stp.split(field, [hook]))

    assert [(p.space, p.time) for p in patches] == [((0, 0), 3), ((8, 8), 5)]
    assert [(e.anchor, e.kind) for e in stp.spatial.errors] == [
        (((0, 8), 4), "OSError")
    ]
    ((anchor, exc),) = [e[1:] for e in hook.events if e[0] == "error"]
    assert anchor == ((0, 8), 4)
    assert isinstance(exc, OSError)  # the original exception, not a summary
    assert hook.events[0] == ("split_start", 3)
    assert hook.kinds("done") == [((0, 0), 3), ((8, 8), 5)]


def test_coupled_split_masks_failed_chip() -> None:
    field = _Flaky(_raster((8, 16, 16)), bad={(0, 8)})
    stp = _coupled([((0, 8), 4)], on_error="mask")

    (patch,) = list(stp.split(field))

    assert patch.data.shape == (2, 8, 8)
    assert np.isnan(np.asarray(patch.data)).all()


def test_coupled_split_honours_journal() -> None:
    field = _Flaky(_raster((8, 16, 16)), bad=set())
    stp = _coupled([((0, 0), 3), ((0, 8), 4), ((8, 8), 5)])
    hook = _Recorder()

    patches = list(stp.split(field, [hook], journal=_Journal([((0, 8), 4)])))

    assert [(p.space, p.time) for p in patches] == [((0, 0), 3), ((8, 8), 5)]
    assert hook.kinds("skipped") == [((0, 8), 4)]
    # The time length is known after the first chip: the journaled one is
    # never read.
    assert (0, 8) not in field.reads


def test_coupled_split_backpressure() -> None:
    stp = _coupled([((0, 0), 3), ((8, 8), 5)])
    field = _raster((8, 16, 16))
    with pytest.raises(ValueError, match="max_in_flight_bytes"):
        list(stp.split(field, max_in_flight_bytes=1))
    patches = list(stp.split(field, max_in_flight=2))
    assert all(p._release is not None for p in patches)
    for p in patches:
        p.close()


def test_coupled_asplit_honours_on_error_and_journal() -> None:
    field = _Flaky(_raster((8, 16, 16)), bad={(0, 8)})
    stp = _coupled([((0, 0), 3), ((0, 8), 4), ((8, 8), 5)], on_error="skip")
    hook = _Recorder()

    patches = asyncio.run(
        _collect(stp.asplit(field, [hook], journal=_Journal([((8, 8), 5)])))
    )

    assert [(p.space, p.time) for p in patches] == [((0, 0), 3)]
    assert [e.anchor for e in stp.spatial.errors] == [((0, 8), 4)]
    assert hook.kinds("skipped") == [((8, 8), 5)]
    assert [e[1] for e in hook.events if e[0] == "error"] == [((0, 8), 4)]


def test_coupled_split_keeps_the_chip_carrier() -> None:
    field = _raster((8, 16, 16))
    stp = _coupled([((8, 8), 5)])
    (patch,) = list(stp.split(field))
    chip = stp.spatial.patch_at(field, (8, 8))
    assert isinstance(patch.data, GeoTensor)
    assert patch.data.transform == chip.data.transform
    np.testing.assert_array_equal(np.asarray(patch.data), np.asarray(chip.data)[4:6])


# -- product spatio-temporal split -------------------------------------------


def test_product_split_reports_unknown_total_and_policy() -> None:
    field = _Flaky(_raster((4, 16, 16)), bad={(8, 8)})
    stp = SpatioTemporalPatcher(
        spatial=_spatial(on_error="skip"), temporal=_temporal(), coupling="product"
    )
    hook = _Recorder()

    patches = list(stp.split(field, [hook]))

    assert hook.events[0] == ("split_start", UNKNOWN_TOTAL)
    assert {p.space for p in patches} == {(0, 0), (0, 8), (8, 0)}
    assert [e.anchor for e in stp.spatial.errors] == [(8, 8)]
    assert len(patches) == 3 * 3  # windows ending at t = 1, 2, 3


def test_product_split_skips_fully_journaled_chips() -> None:
    field = _Flaky(_raster((4, 16, 16)), bad=set())
    stp = SpatioTemporalPatcher(spatial=_spatial(), temporal=_temporal())
    journal = _Journal([((0, 8), t) for t in (1, 2, 3)] + [((8, 8), 2)])

    patches = list(stp.split(field, journal=journal))

    assert sorted((p.space, p.time) for p in patches if p.space[1] == 8) == [
        ((8, 8), 1),
        ((8, 8), 3),
    ]
    assert (0, 8) not in field.reads  # every window journaled: chip not read


def test_product_split_keys_multi_window_patches() -> None:
    stp = SpatioTemporalPatcher(
        spatial=_spatial(sampler=spatial.sampler.Explicit(anchors_=[(0, 0)])),
        temporal=_temporal(temporal.geometry.MultiScale(scales=[1, 2])),
    )
    field = _raster((4, 16, 16))
    keys = [((0, 0), (3, 0)), ((0, 0), (3, 1))]
    hook = _Recorder()
    patches = list(stp.split(field, [hook], journal=_Journal(keys)))
    assert {p.time for p in patches} == {1, 2}
    assert hook.kinds("skipped") == [((0, 0), 3), ((0, 0), 3)]


# -- errors and masks ---------------------------------------------------------


def test_errors_are_scoped_to_the_latest_split() -> None:
    patcher = _spatial(on_error="skip")
    field = _raster((16, 16))
    list(patcher.split(_Flaky(field, bad={(0, 0)})))
    first = patcher.errors
    list(patcher.split(_Flaky(field, bad={(8, 8)})))
    assert [e.anchor for e in patcher.errors] == [(8, 8)]
    assert [e.anchor for e in first] == [(0, 0)]


def test_temporal_errors_are_scoped_to_the_latest_split() -> None:
    class _Series:
        shape = (6,)

        def __init__(self, bad: int) -> None:
            self.bad = bad

        def __getitem__(self, s: slice) -> np.ndarray:
            if s.start <= self.bad < s.stop:
                raise OSError("gap")
            return np.arange(6.0)[s]

    tp = TemporalPatcher(
        geometry=temporal.geometry.FixedLookback(length=1),
        sampler=temporal.sampler.RegularStride(step=1),
        window=temporal.window.CausalBoxcar(),
        aggregation=temporal.aggregation.Mean(),
        on_error="skip",
    )
    list(tp.split(_Series(bad=1)))
    list(tp.split(_Series(bad=4)))
    assert [e.anchor for e in tp.errors] == [4]


def test_mask_patch_for_graph_indices() -> None:
    import geopandas as gpd
    import shapely

    from geopatcher._src.fields.geopandas import GeoPandasField

    gdf = gpd.GeoDataFrame(
        {"v": np.arange(5.0)},
        geometry=[shapely.Point(float(i), float(i % 3)) for i in range(5)],
        crs="EPSG:4326",
    )

    class _Broken(GeoPandasField):
        def select(self, indexer: Any) -> Any:
            raise OSError("offline")

    patcher = SpatialPatcher(
        geometry=spatial.geometry.KNNGraph(k=2),
        sampler=spatial.sampler.Random(n_samples=2, seed=0),
        window=spatial.window.Boxcar(),
        aggregation=spatial.aggregation.OverlapAdd(),
        on_error="mask",
    )
    patches = list(patcher.split(_Broken(gdf, as_points=True)))
    assert [p.data.shape for p in patches] == [(2,), (2,)]
    assert all(np.isnan(p.data).all() for p in patches)
    assert len(patcher.errors) == 2
