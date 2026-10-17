"""`on_error="mask"` placeholders and async `reduce` / `two_pass` (#405).

A failed read under ``on_error="mask"`` yields an all-NaN patch flagged
``Patch.masked``. Where the field allows, the placeholder keeps the carrier
and georeferencing of a successful read: a `GeoArray` on the window's grid
for a raster field, a `DataArray` from ``Field.placeholder`` for the xarray
adapters. `AsyncSpatialPatcher.reduce` / `two_pass` mirror the sync ones.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest
from _helpers import ArrayField, make_raster_field
from geoarray import GeoArray
from rasterio.windows import Window, transform as window_transform

from geopatcher import AsyncSpatialPatcher, RasterField, SpatialPatcher, spatial


BAD = (0, 8)  # the anchor whose read fails


def _patcher(cls: type = SpatialPatcher, **kwargs: Any) -> Any:
    return cls(
        geometry=spatial.geometry.Rectangular(size=(8, 8)),
        sampler=spatial.sampler.RegularStride(step=8),
        window=spatial.window.Boxcar(),
        aggregation=spatial.aggregation.OverlapAdd(),
        **kwargs,
    )


def _fails(window: Any) -> bool:
    if isinstance(window, dict):
        return (window["y"].start, window["x"].start) == BAD
    return (int(window.row_off), int(window.col_off)) == BAD


class FlakyRaster(RasterField):
    def select(self, window: Any) -> Any:
        if _fails(window):
            raise OSError("tile unreadable")
        return super().select(window)


def test_raster_placeholder_is_a_georeferenced_geoarray() -> None:
    src = make_raster_field(16, dtype=np.float32)
    patcher = _patcher(on_error="mask")
    patches = list(patcher.split(FlakyRaster(src.reader)))

    masked = [p for p in patches if p.masked]
    assert [p.anchor for p in masked] == [BAD]
    chip = masked[0].data
    assert isinstance(chip, GeoArray)
    assert np.isnan(np.asarray(chip)).all()
    assert chip.grid.affine == window_transform(
        Window(8, 0, 8, 8), src.reader.grid.affine
    )
    assert chip.crs == src.reader.crs
    assert not any(p.masked for p in patches if p.anchor != BAD)


def test_raster_placeholder_merges_to_field() -> None:
    src = make_raster_field(16, dtype=np.float64)
    flaky = FlakyRaster(src.reader)
    patcher = _patcher(on_error="mask")
    out = patcher.merge_to_field(list(patcher.split(flaky)), flaky)

    # The hole takes the source's nodata; every other cell is the data.
    assert isinstance(out, GeoArray)
    assert out.grid.affine == src.reader.grid.affine
    expected = np.asarray(src.reader).copy()
    expected[0:8, 8:16] = out.fill_value
    np.testing.assert_array_equal(np.asarray(out), expected)


def test_plain_array_field_keeps_a_bare_placeholder() -> None:
    class Flaky(ArrayField):
        def __init__(self, array: np.ndarray) -> None:
            super().__init__(array)
            self.crs = None  # no georeferencing: nothing to wrap

        def select(self, window: Any) -> np.ndarray:
            if _fails(window):
                raise OSError("tile unreadable")
            return super().select(window)

    field = Flaky(np.zeros((16, 16), dtype=np.float32))
    masked = next(p for p in _patcher(on_error="mask").split(field) if p.masked)
    assert type(masked.data) is np.ndarray
    assert np.isnan(masked.data).all()


def _dataarray() -> Any:
    xr = pytest.importorskip("xarray")
    pytest.importorskip("rioxarray")
    from rasterio.transform import from_origin

    transform = from_origin(500_000.0, 4_200_000.0, 10.0, 10.0)
    da = xr.DataArray(
        np.arange(16 * 16, dtype=np.float64).reshape(16, 16),
        dims=("y", "x"),
        coords={
            "y": transform.f + (np.arange(16) + 0.5) * transform.e,
            "x": transform.c + (np.arange(16) + 0.5) * transform.a,
        },
        name="v",
    )
    return da.rio.write_crs("EPSG:32630").rio.write_transform(transform)


def test_xarray_placeholder_is_a_dataarray_with_the_chip_coords() -> None:
    from geopatcher.fields import XarrayField

    class Flaky(XarrayField):
        def select(self, indexer: Any) -> Any:
            if _fails(indexer):
                raise OSError("tile unreadable")
            return super().select(indexer)

    da = _dataarray()
    patches = list(_patcher(on_error="mask").split(Flaky(da)))
    masked = next(p for p in patches if p.masked)
    good = XarrayField(da).select(masked.indices)

    assert type(masked.data) is type(good)
    assert masked.data.dims == good.dims
    np.testing.assert_array_equal(masked.data["x"], good["x"])
    np.testing.assert_array_equal(masked.data["y"], good["y"])
    assert np.isnan(np.asarray(masked.data)).all()


def test_rioxarray_placeholder_carries_the_window_transform() -> None:
    from geopatcher.fields import RioXarrayField

    class Flaky(RioXarrayField):
        def select(self, window: Any) -> Any:
            if _fails(window):
                raise OSError("tile unreadable")
            return super().select(window)

    da = _dataarray()
    field = Flaky(da)
    patcher = _patcher(on_error="mask")
    patches = list(patcher.split(field))
    masked = next(p for p in patches if p.masked)
    good = RioXarrayField(da).select(masked.indices)

    assert masked.data.rio.transform() == good.rio.transform()
    assert masked.data.rio.crs == good.rio.crs
    np.testing.assert_allclose(masked.data["x"], good["x"])
    np.testing.assert_allclose(masked.data["y"], good["y"])
    assert np.isnan(np.asarray(masked.data)).all()

    merged = patcher.merge_to_field(patches, field)
    values = np.asarray(getattr(merged, "da", merged))
    assert np.isnan(values[0:8, 8:16]).all()
    assert not np.isnan(values[8:, :]).any()


# ---------------------------------------------------------------------------
# AsyncSpatialPatcher.reduce / two_pass


class AsyncRaster(RasterField):
    async def aselect(self, window: Any) -> Any:
        await asyncio.sleep(0)
        return self.select(window)


class FlakyAsyncRaster(AsyncRaster):
    def select(self, window: Any) -> Any:
        if _fails(window):
            raise OSError("tile unreadable")
        return super().select(window)


@pytest.mark.parametrize(
    "agg", [spatial.aggregation.MeanStd, spatial.aggregation.MinMax]
)
def test_async_reduce_matches_sync(agg: type) -> None:
    field = AsyncRaster(make_raster_field(16, dtype=np.float64).reader)
    expected = _patcher().reduce(field, agg())
    got = asyncio.run(_patcher(AsyncSpatialPatcher).reduce(field, agg()))
    assert got == pytest.approx(expected)


def test_async_two_pass_matches_sync() -> None:
    field = AsyncRaster(make_raster_field(16, dtype=np.float64).reader)

    def apply(data: Any, stats: dict[str, float]) -> np.ndarray:
        return (np.asarray(data) - stats["mean"]) / stats["std"]

    kwargs = {"reduce_with": spatial.aggregation.MeanStd(), "apply": apply}
    expected = _patcher().two_pass(field, **kwargs)
    got = asyncio.run(_patcher(AsyncSpatialPatcher).two_pass(field, **kwargs))
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected))
    np.testing.assert_allclose(np.mean(got), 0.0, atol=1e-12)


def test_async_reduce_applies_the_on_error_policy() -> None:
    field = FlakyAsyncRaster(make_raster_field(16, dtype=np.float64).reader)
    patcher = _patcher(AsyncSpatialPatcher, on_error="skip")
    stats = asyncio.run(patcher.reduce(field, spatial.aggregation.MinMax()))

    data = np.asarray(field.reader).copy()
    data[0:8, 8:16] = np.nan
    assert stats == {"min": float(np.nanmin(data)), "max": float(np.nanmax(data))}
    assert [e.kind for e in patcher.errors] == ["OSError"]


def test_async_two_pass_records_both_passes_into_one_error_list() -> None:
    field = FlakyAsyncRaster(make_raster_field(16, dtype=np.float64).reader)
    patcher = _patcher(AsyncSpatialPatcher, on_error="mask")
    out = asyncio.run(
        patcher.two_pass(
            field,
            reduce_with=spatial.aggregation.MinMax(),
            apply=lambda data, stats: np.asarray(data) - stats["min"],
        )
    )
    assert np.isnan(np.asarray(out)[0:8, 8:16]).all()
    assert len(patcher.errors) == 2  # the bad tile, once per pass


def test_async_reduce_releases_backpressure_slots() -> None:
    field = AsyncRaster(make_raster_field(16, dtype=np.float64).reader)
    patcher = _patcher(AsyncSpatialPatcher)
    stats = asyncio.run(
        asyncio.wait_for(
            patcher.reduce(field, spatial.aggregation.MinMax(), max_in_flight=1),
            timeout=10,
        )
    )
    assert stats == {"min": 0.0, "max": 255.0}
