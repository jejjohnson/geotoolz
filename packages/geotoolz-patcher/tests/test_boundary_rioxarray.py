"""`boundary="pad"` on a `RioXarrayField` — regression for issue #19.

Historically ``boundary="pad"`` silently degraded to ``"shrink"`` on a
`RioXarrayField` because its ``select`` clips via ``isel``. The patcher
now guarantees padding itself (clip-and-pad), so the edge chip is the
full geometry size with the requested fill on any `Field`.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio

from geopatcher import (
    SpatialBoxcar,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRectangular,
    SpatialRegularStride,
)


xr = pytest.importorskip("xarray")
pytest.importorskip("rioxarray")

from geopatcher._src.fields.rio_xarray import RioXarrayField


def _rio_field(n: int = 10) -> RioXarrayField:
    arr = np.arange(n * n, dtype=np.float32).reshape(n, n)
    da = xr.DataArray(arr, dims=("y", "x"))
    da = da.rio.write_crs("EPSG:32630")
    da = da.rio.write_transform(rasterio.Affine.identity())
    return RioXarrayField(da)


def _patcher(pad_value: float | None) -> SpatialPatcher:
    return SpatialPatcher(
        geometry=SpatialRectangular(size=(4, 4), boundary="pad", pad_value=pad_value),
        sampler=SpatialRegularStride(step=4),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )


def test_rioxarray_pad_is_full_size_with_fill() -> None:
    field = _rio_field(10)
    raw = np.asarray(field.da.values)
    patches = {p.anchor: p for p in _patcher(pad_value=-1.0).split(field)}
    # Anchor (8, 8) overflows by 2 on each axis on the 10x10 domain.
    corner = patches[(8, 8)]
    chip = np.asarray(corner.data.da.values)
    assert chip.shape == (4, 4)  # not silently shrunk to (2, 2)
    np.testing.assert_array_equal(chip[0:2, 0:2], raw[8:10, 8:10])
    assert np.all(chip[2:, :] == -1.0)
    assert np.all(chip[:, 2:] == -1.0)


def test_rioxarray_interior_chip_unpadded() -> None:
    field = _rio_field(10)
    raw = np.asarray(field.da.values)
    patches = {p.anchor: p for p in _patcher(pad_value=-1.0).split(field)}
    interior = patches[(0, 0)]
    np.testing.assert_array_equal(np.asarray(interior.data.da.values), raw[0:4, 0:4])
