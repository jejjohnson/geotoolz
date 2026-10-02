"""`RioXarrayField` chips and ``boundary="pad"`` — regressions for #19 / #178.

Historically ``boundary="pad"`` silently degraded to ``"shrink"`` on a
`RioXarrayField` because its ``select`` clips via ``isel``. The patcher
now guarantees padding itself (clip-and-pad), so the edge chip is the
full geometry size with the requested fill on any `Field`.

#178: ``select`` returns the sliced `DataArray` (not a field wrapper)
with an exact window transform — for both ``write_transform``-only and
coordinate-bearing arrays — and padded chips keep exact coords instead
of NaN (``pad``) or mirrored (``reflect``) ones. Every chip's bounds are
checked against `rasterio.windows.bounds`.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.windows import (
    Window,
    bounds as window_bounds,
    transform as window_transform,
)

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


# 10 m UTM grid with origin (500000, 4600000) — non-identity, so a chip
# that kept the full-image transform is caught.
_T = rasterio.Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4600000.0)


def _rio_field(n: int = 10) -> RioXarrayField:
    arr = np.arange(n * n, dtype=np.float32).reshape(n, n)
    da = xr.DataArray(arr, dims=("y", "x"))
    da = da.rio.write_crs("EPSG:32630")
    da = da.rio.write_transform(rasterio.Affine.identity())
    return RioXarrayField(da)


def _geo_da(n: int = 10, *, coords: bool, nodata: float | None = None) -> xr.DataArray:
    """``write_transform``-only (``coords=False``) or coordinate-bearing array."""
    arr = np.arange(n * n, dtype=np.float32).reshape(n, n)
    if coords:
        cols = np.arange(n) + 0.5
        da = xr.DataArray(
            arr,
            dims=("y", "x"),
            coords={"y": _T.f + cols * _T.e, "x": _T.c + cols * _T.a},
        )
    else:
        da = xr.DataArray(arr, dims=("y", "x"))
    da = da.rio.write_crs("EPSG:32630").rio.write_transform(_T)
    if nodata is not None:
        da = da.rio.write_nodata(nodata)
    return da


def _assert_georef(chip: xr.DataArray, window: Window) -> None:
    """Chip transform/bounds equal rasterio's for ``window`` on the source grid."""
    assert chip.shape[-2:] == (int(window.height), int(window.width))
    assert chip.rio.transform() == window_transform(window, _T)
    np.testing.assert_allclose(chip.rio.bounds(), window_bounds(window, _T))
    if "x" in chip.coords:
        assert np.all(np.isfinite(chip["x"].values))
        assert np.all(np.isfinite(chip["y"].values))
        cols = np.arange(int(window.width)) + 0.5
        rows = np.arange(int(window.height)) + 0.5
        wt = window_transform(window, _T)
        np.testing.assert_allclose(chip["x"].values, wt.c + cols * wt.a)
        np.testing.assert_allclose(chip["y"].values, wt.f + rows * wt.e)


def _patcher(pad_value: float | None, boundary: str = "pad") -> SpatialPatcher:
    return SpatialPatcher(
        geometry=SpatialRectangular(
            size=(4, 4),
            boundary=boundary,  # type: ignore[arg-type]
            pad_value=pad_value,
        ),
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
    chip = np.asarray(corner.data.values)
    assert chip.shape == (4, 4)  # not silently shrunk to (2, 2)
    np.testing.assert_array_equal(chip[0:2, 0:2], raw[8:10, 8:10])
    assert np.all(chip[2:, :] == -1.0)
    assert np.all(chip[:, 2:] == -1.0)


def test_rioxarray_interior_chip_unpadded() -> None:
    field = _rio_field(10)
    raw = np.asarray(field.da.values)
    patches = {p.anchor: p for p in _patcher(pad_value=-1.0).split(field)}
    interior = patches[(0, 0)]
    np.testing.assert_array_equal(np.asarray(interior.data.values), raw[0:4, 0:4])


@pytest.mark.parametrize("coords", [False, True], ids=["transform-only", "coords"])
def test_select_returns_dataarray(coords: bool) -> None:
    chip = RioXarrayField(_geo_da(coords=coords)).select(Window(4, 4, 4, 4))
    assert isinstance(chip, xr.DataArray)
    assert np.asarray(chip).dtype == np.float32


@pytest.mark.parametrize(
    "window",
    [
        Window(4, 4, 4, 4),  # interior
        Window(0, 0, 4, 4),  # top-left
        Window(6, 0, 4, 4),  # top-right
        Window(0, 6, 4, 4),  # bottom-left
        Window(6, 6, 4, 4),  # bottom-right
    ],
    ids=["interior", "tl", "tr", "bl", "br"],
)
def test_select_transform_without_coords(window: Window) -> None:
    field = RioXarrayField(_geo_da(coords=False))
    chip = field.select(window)
    _assert_georef(chip, window)
    np.testing.assert_array_equal(chip.values, field.da.values[window.toslices()])


@pytest.mark.parametrize(
    "window",
    [
        Window(4, 4, 4, 4),
        Window(0, 0, 4, 4),
        Window(6, 0, 4, 4),
        Window(0, 6, 4, 4),
        Window(6, 6, 4, 4),
    ],
    ids=["interior", "tl", "tr", "bl", "br"],
)
def test_select_transform_with_coords(window: Window) -> None:
    field = RioXarrayField(_geo_da(coords=True))
    _assert_georef(field.select(window), window)


@pytest.mark.parametrize("coords", [False, True], ids=["transform-only", "coords"])
@pytest.mark.parametrize(
    "window",
    [
        Window(-1, -1, 4, 4),
        Window(7, -1, 4, 4),
        Window(-1, 7, 4, 4),
        Window(7, 7, 4, 4),
    ],
    ids=["tl", "tr", "bl", "br"],
)
@pytest.mark.parametrize("boundary", ["pad", "reflect"])
def test_pad_chip_bounds_exact_with_coords(
    coords: bool, window: Window, boundary: str
) -> None:
    field = RioXarrayField(_geo_da(coords=coords))
    patcher = _patcher(pad_value=-1.0, boundary=boundary)
    anchor = (int(window.row_off), int(window.col_off))
    patch = patcher.patch_at(field, anchor)
    _assert_georef(patch.data, window)
    if boundary == "pad":
        chip = np.asarray(patch.data.values)
        assert np.sum(chip == -1.0) == 7  # 1 overflow row + 1 col of a 4x4 chip


@pytest.mark.parametrize("coords", [False, True], ids=["transform-only", "coords"])
def test_select_out_of_range_window_padded_with_nodata(coords: bool) -> None:
    field = RioXarrayField(_geo_da(coords=coords, nodata=-9999.0))
    window = Window(-2, -2, 4, 4)
    chip = field.select(window)
    _assert_georef(chip, window)
    np.testing.assert_array_equal(chip.values[2:, 2:], field.da.values[:2, :2])
    assert np.all(chip.values[:2, :] == -9999.0)
    assert np.all(chip.values[:, :2] == -9999.0)
    # Fully outside the raster: all nodata, still exactly georeferenced.
    outside = Window(20, 20, 3, 3)
    far = field.select(outside)
    _assert_georef(far, outside)
    assert np.all(far.values == -9999.0)


@pytest.mark.parametrize("coords", [False, True], ids=["transform-only", "coords"])
@pytest.mark.parametrize("boundary", ["drop", "shrink", "pad", "reflect"])
def test_rioxarray_merge(coords: bool, boundary: str) -> None:
    da = _geo_da(coords=coords)
    field = RioXarrayField(da)
    patcher = SpatialPatcher(
        geometry=SpatialRectangular(size=(4, 4), boundary=boundary),  # type: ignore[arg-type]
        sampler=SpatialRegularStride(step=3),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )
    merged = patcher.merge(patcher.split(field), field.domain)
    np.testing.assert_allclose(np.asarray(merged), da.values)
    out = patcher.merge_to_xarray(patcher.split(field), field)
    np.testing.assert_allclose(out.values, da.values)
    assert out.rio.transform() == _T
