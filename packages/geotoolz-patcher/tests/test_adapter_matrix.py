"""Every `Field` adapter splits and merges through one `SpatialPatcher` (#168).

One parametrised round trip over a real (file-backed or
coordinate-bearing) fixture per adapter, so a regression in any
adapter's ``select`` / ``domain`` / ``with_data`` contract shows up in
the same place:

- chips are the natural, materialised payload (``GeoTensor`` or
  ``xarray.DataArray``), never a Field wrapper or a lazy reader;
- raster chips carry the exact window transform;
- ``merge(split(field), field.domain)`` reproduces the source;
- ``with_data(merged)`` rebuilds a field on the same grid.

The 40 x 50 extent is not a multiple of the 16-px chip, so the edge
chips are exercised under every edge-covering boundary mode: ``shrink``
clips them, ``pad`` / ``reflect`` overhang the domain and the merge
crops them back (#185) — on the raster path and the `GridDomain` path
(`XarrayField` / `DaskField`) alike.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor
from rasterio import Affine
from rasterio.windows import Window, transform as window_transform

from geopatcher import RasterField, SpatialPatcher, spatial
from geopatcher._src.fields.reproject import ReprojectingRasterField


_H, _W, _BANDS = 40, 50, 2
_CRS = "EPSG:32630"
_T = Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_600_000.0)


def _values(bands: int = _BANDS) -> np.ndarray:
    b, r, c = np.meshgrid(np.arange(bands), np.arange(_H), np.arange(_W), indexing="ij")
    return (b * 10_000 + r * 100 + c).astype(np.float32)


def _write_tif(path: Path, *, tiled: bool = False) -> Path:
    data = _values()
    options: dict[str, Any] = {}
    if tiled:
        options = {"tiled": True, "blockxsize": 16, "blockysize": 16}
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=_H,
        width=_W,
        count=_BANDS,
        dtype="float32",
        crs=_CRS,
        transform=_T,
        nodata=-1.0,
        compress="deflate",
        **options,
    ) as dst:
        dst.write(data)
    return path


def _dataarray(*, coords: bool) -> Any:
    xr = pytest.importorskip("xarray")
    pytest.importorskip("rioxarray")
    centres_x = _T.c + (np.arange(_W) + 0.5) * _T.a
    centres_y = _T.f + (np.arange(_H) + 0.5) * _T.e
    da = xr.DataArray(
        _values(1)[0],
        dims=("y", "x"),
        coords={"y": centres_y, "x": centres_x} if coords else None,
    )
    return da.rio.write_crs(_CRS).rio.write_transform(_T)


@dataclass
class Case:
    """An adapter under test, the array its merge must reproduce, and its grid."""

    field: Any
    expected: np.ndarray
    transform: Affine | None  # source grid for chip georeferencing checks


def _raster_geotensor(tmp_path: Path) -> Case:
    gt = GeoTensor(values=_values(), transform=_T, crs=_CRS, fill_value_default=-1.0)
    return Case(RasterField(gt), _values(), _T)


def _raster_rasterio_reader(tmp_path: Path) -> Case:
    from georeader.rasterio_reader import RasterioReader

    path = _write_tif(tmp_path / "src.tif")
    return Case(RasterField(RasterioReader(str(path))), _values(), _T)


def _xarray(tmp_path: Path) -> Case:
    from geopatcher.fields import XarrayField

    da = _dataarray(coords=True)
    return Case(XarrayField(da), da.values, None)


def _rioxarray(tmp_path: Path) -> Case:
    from geopatcher.fields import RioXarrayField

    da = _dataarray(coords=True)
    return Case(RioXarrayField(da), da.values, _T)


def _dask(tmp_path: Path) -> Case:
    pytest.importorskip("dask.array")
    from geopatcher._src.fields.dask import DaskField

    da = _dataarray(coords=True).chunk({"y": 16, "x": 16})
    return Case(DaskField(da), da.values, _T)


def _cog_field(tmp_path: Path) -> Case:
    pytest.importorskip("obstore")
    pytest.importorskip("async_geotiff")
    from obstore.store import LocalStore

    from geopatcher.fields import CogField

    path = _write_tif(tmp_path / "cog.tif", tiled=True)
    field = CogField.open(
        url=f"file://{path}", store=LocalStore(prefix=str(tmp_path)), path=path.name
    )
    return Case(field, _values(), _T)


def _reprojecting_3d(tmp_path: Path) -> Case:
    # Same-CRS, same-resolution, nearest: the warp is the identity, so the
    # merge must reproduce the source exactly while every chip still goes
    # through the per-chip crop + warp path.
    gt = GeoTensor(values=_values(), transform=_T, crs=_CRS, fill_value_default=-1.0)
    field = ReprojectingRasterField(
        gt, dst_crs=_CRS, resolution=10.0, resampling="nearest"
    )
    assert field.domain.shape == (_BANDS, _H, _W)
    return Case(field, _values(), field.domain.transform)


ADAPTERS: dict[str, Callable[[Path], Case]] = {
    "RasterField-GeoTensor": _raster_geotensor,
    "RasterField-RasterioReader": _raster_rasterio_reader,
    "XarrayField": _xarray,
    "RioXarrayField-coords": _rioxarray,
    "DaskField": _dask,
    "CogField": _cog_field,
    "ReprojectingRasterField-3d": _reprojecting_3d,
}


def _patcher(boundary: str) -> SpatialPatcher:
    return SpatialPatcher(
        geometry=spatial.geometry.Rectangular(size=(16, 16), boundary=boundary),  # type: ignore[arg-type]
        sampler=spatial.sampler.RegularStride(step=16),
        window=spatial.window.Boxcar(),
        aggregation=spatial.aggregation.OverlapAdd(),
    )


def _chip_transform(data: Any) -> Affine:
    if isinstance(data, GeoTensor):
        return data.transform
    return data.rio.transform()


@pytest.mark.parametrize("boundary", ["shrink", "pad", "reflect"])
@pytest.mark.parametrize("adapter", list(ADAPTERS))
def test_adapter_split_merge_round_trip(
    adapter: str, boundary: str, tmp_path: Path
) -> None:
    case = ADAPTERS[adapter](tmp_path)
    field = case.field
    patcher = _patcher(boundary)

    patches = list(patcher.split(field))
    assert patches
    for p in patches:
        assert not hasattr(p.data, "domain"), f"{adapter} returned a Field wrapper"
        values = np.asarray(p.data)
        assert values.dtype != object and values.ndim >= 2
        if case.transform is not None:
            row, col = _row_col(p.anchor)
            window = Window(col, row, values.shape[-1], values.shape[-2])
            assert _chip_transform(p.data) == window_transform(window, case.transform)

    merged = np.asarray(patcher.merge(patches, field.domain))
    np.testing.assert_array_equal(merged, case.expected)

    rebuilt = field.with_data(merged)
    np.testing.assert_array_equal(np.asarray(_payload(rebuilt)), case.expected)


def _row_col(anchor: Any) -> tuple[int, int]:
    """Raster anchors are ``(row, col)``; `GridDomain` anchors are dim dicts."""
    if isinstance(anchor, dict):
        return int(anchor["y"]), int(anchor["x"])
    return int(anchor[-2]), int(anchor[-1])


def _payload(rebuilt: Any) -> Any:
    """The array behind a ``with_data`` result (a carrier or a new Field)."""
    for attr in ("da", "array"):
        inner = getattr(rebuilt, attr, None)
        if inner is not None and not callable(inner):
            return inner
    return rebuilt
