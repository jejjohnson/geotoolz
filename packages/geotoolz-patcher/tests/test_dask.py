"""`DaskField` against the real ``dask`` package — #178.

``select`` returns the materialised `xarray.DataArray` chip (not another
`DaskField`), so the patches feed the dense aggregations and split →
merge round-trips.
"""

from __future__ import annotations

import numpy as np
import pytest


pytest.importorskip("dask.array")
xr = pytest.importorskip("xarray")

from geopatcher import (
    SpatialBoxcar,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRectangular,
    SpatialRegularStride,
)
from geopatcher._src.fields.dask import DaskField


def _cube(n: int = 12) -> xr.DataArray:
    cols = np.arange(n) + 0.5
    return xr.DataArray(
        np.arange(n * n, dtype=np.float32).reshape(n, n),
        dims=("y", "x"),
        coords={"y": 4600000.0 - 10.0 * cols, "x": 500000.0 + 10.0 * cols},
    ).chunk({"y": 5, "x": 5})


def _patcher(step: int = 4) -> SpatialPatcher:
    return SpatialPatcher(
        geometry=SpatialRectangular(size=(4, 4)),
        sampler=SpatialRegularStride(step=step),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )


def test_select_returns_materialised_dataarray() -> None:
    da = _cube()
    chip = DaskField(da).select({"y": slice(4, 8), "x": slice(0, 4)})
    assert isinstance(chip, xr.DataArray)
    assert isinstance(chip.data, np.ndarray)  # computed, not a dask graph
    np.testing.assert_array_equal(chip.values, da.values[4:8, 0:4])
    np.testing.assert_array_equal(chip["x"].values, da["x"].values[0:4])
    np.testing.assert_array_equal(chip["y"].values, da["y"].values[4:8])


def test_select_writes_window_transform_when_georeferenced() -> None:
    pytest.importorskip("rioxarray")
    from rasterio import Affine
    from rasterio.windows import Window, transform as window_transform

    t = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4600000.0)
    da = (
        xr.DataArray(np.zeros((12, 12), dtype=np.float32), dims=("y", "x"))
        .rio.write_crs("EPSG:32630")
        .rio.write_transform(t)
        .chunk()
    )
    chip = DaskField(da).select({"y": slice(4, 8), "x": slice(6, 10)})
    assert chip.rio.transform() == window_transform(Window(6, 4, 4, 4), t)


@pytest.mark.parametrize("step", [2, 4])
def test_dask_field_split_merge(step: int) -> None:
    da = _cube()
    field = DaskField(da)
    patcher = _patcher(step)
    merged = patcher.merge(patcher.split(field), field.domain)
    np.testing.assert_allclose(np.asarray(merged), da.values)
