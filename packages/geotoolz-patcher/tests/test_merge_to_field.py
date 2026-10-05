"""`merge_to_field` / `merge_to_xarray` — rebuilding a georeferenced carrier (#194).

`SpatialPatcher.merge` returns the aggregation's raw output (ADR-007);
`merge_to_field` wraps it through ``field.with_data`` so the transform,
CRS, nodata and attrs of the source survive. Covered on every sync
raster/grid adapter: an in-memory `GeoTensor`, a file-backed
`RasterioReader`, `RioXarrayField` and `XarrayField`. Dict-returning
aggregations and off-grid shapes raise a clear `TypeError`, and the
merged values keep the source dtype when they fit it.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
import rasterio
from _helpers import make_rasterio_reader_field
from georeader.geotensor import GeoTensor

from geopatcher import (
    RasterField,
    SpatialAggregation,
    SpatialBoxcar,
    SpatialMeanStd,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRectangular,
    SpatialRegularStride,
)
from geopatcher._src.matched import MatchedField, MatchedSpatialPatcher
from geopatcher._src.matched.patch import PRIMARY_KEY


try:
    import rioxarray  # noqa: F401 - registers the `.rio` accessor
    import xarray as xr

    from geopatcher._src.fields.rio_xarray import RioXarrayField
    from geopatcher._src.fields.xarray import XarrayField
except ImportError:  # the GeoTensor / RasterioReader tests still run
    xr = None

needs_xarray = pytest.mark.skipif(
    xr is None, reason="needs the [xarray-raster] extra (xarray, rioxarray)"
)


_T = rasterio.Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_600_000.0)


def _patcher(size: int = 4, aggregation: SpatialAggregation | None = None) -> Any:
    return SpatialPatcher(
        geometry=SpatialRectangular(size=(size, size)),
        sampler=SpatialRegularStride(step=(size, size)),
        window=SpatialBoxcar(),
        aggregation=aggregation or SpatialOverlapAdd(),
    )


def _geotensor_field(dtype: Any = np.uint16) -> RasterField:
    values = np.arange(2 * 8 * 8, dtype=dtype).reshape(2, 8, 8)
    return RasterField(
        GeoTensor(
            values=values,
            transform=_T,
            crs="EPSG:32630",
            fill_value_default=999,
            attrs={"band_names": ["b1", "b2"]},
        )
    )


def _rio_da() -> Any:
    da = xr.DataArray(
        np.arange(8 * 8, dtype=np.float32).reshape(8, 8),
        dims=("y", "x"),
        attrs={"long_name": "radiance"},
    )
    da = da.rio.write_crs("EPSG:32630")
    da = da.rio.write_transform(_T)
    return da.rio.write_nodata(-9999.0)


class _FixedOutput(SpatialAggregation):
    """Aggregation returning a canned output — for the type / shape guards."""

    streaming_safe: ClassVar[bool] = True

    def __init__(self, output: Any) -> None:
        self.output = output

    def merge(self, patches: Iterable[Any], domain: Any) -> Any:
        for _ in patches:
            pass
        return self.output


# ---------------------------------------------------------------------------
# merge_to_field on every sync adapter
# ---------------------------------------------------------------------------


def test_merge_to_field_geotensor_carries_georeferencing() -> None:
    field = _geotensor_field()
    patcher = _patcher()

    out = patcher.merge_to_field(patcher.split(field), field)

    assert isinstance(out, GeoTensor)
    assert out.transform == _T
    assert str(out.crs) == "EPSG:32630"
    assert out.fill_value_default == 999
    assert out.attrs == {"band_names": ["b1", "b2"]}
    # Integral values fit the uint16 source, so its dtype is restored.
    assert out.values.dtype == np.uint16
    np.testing.assert_array_equal(out.values, field.reader.values)


def test_merge_to_field_rasterio_reader(tmp_path: Path) -> None:
    field = make_rasterio_reader_field(tmp_path / "scene.tif", nodata=-1.0)
    reader = field.reader
    patcher = _patcher(size=10)  # 70 x 70 tiles exactly

    out = patcher.merge_to_field(patcher.split(field), field)

    assert isinstance(out, GeoTensor)
    assert out.transform == reader.transform
    assert out.crs == reader.crs
    assert out.fill_value_default == -1.0
    assert out.attrs == dict(reader.attrs or {})
    assert out.values.dtype == np.float32
    np.testing.assert_array_equal(out.values, reader.load().values)


@needs_xarray
def test_merge_to_field_rioxarray() -> None:
    da = _rio_da()
    field = RioXarrayField(da)
    patcher = _patcher()

    out = patcher.merge_to_field(patcher.split(field), field)

    assert isinstance(out, RioXarrayField)
    assert out.da.rio.transform() == _T
    assert out.da.rio.crs == da.rio.crs
    assert out.da.rio.nodata == -9999.0
    assert out.da.attrs["long_name"] == "radiance"
    assert out.da.dtype == np.float32
    np.testing.assert_array_equal(out.da.values, da.values)


@needs_xarray
def test_merge_to_field_xarray() -> None:
    da = xr.DataArray(
        np.arange(8 * 12, dtype=np.float32).reshape(8, 12),
        dims=("latitude", "longitude"),
        coords={
            "latitude": np.linspace(-30, 30, 8),
            "longitude": np.linspace(0, 60, 12),
        },
        attrs={"units": "K"},
    ).rio.write_crs("EPSG:4326")
    field = XarrayField(da)
    patcher = _patcher()

    out = patcher.merge_to_field(patcher.split(field), field)

    assert isinstance(out, XarrayField)
    xr.testing.assert_identical(out.da, da)
    assert out.domain.crs == field.domain.crs


# ---------------------------------------------------------------------------
# dtype rule
# ---------------------------------------------------------------------------


def test_merge_to_field_keeps_float_when_values_do_not_fit_integer_source() -> None:
    field = _geotensor_field()
    patcher = _patcher()
    halves = (p.with_data(np.asarray(p.data) + 0.5) for p in patcher.split(field))

    out = patcher.merge_to_field(halves, field)

    assert out.values.dtype == np.float64
    np.testing.assert_array_equal(out.values, field.reader.values + 0.5)


def test_merge_to_field_keeps_float_for_nan_on_integer_source() -> None:
    field = _geotensor_field()
    nan_out = np.full((2, 8, 8), np.nan)
    patcher = _patcher(aggregation=_FixedOutput(nan_out))

    out = patcher.merge_to_field([], field)

    assert out.values.dtype == np.float64
    assert np.isnan(out.values).all()


def test_merge_to_field_out_of_range_keeps_aggregation_dtype() -> None:
    field = _geotensor_field(np.uint8)
    patcher = _patcher(aggregation=_FixedOutput(np.full((2, 8, 8), 300.0)))

    out = patcher.merge_to_field([], field)

    assert out.values.dtype == np.float64


# ---------------------------------------------------------------------------
# Type / shape guards
# ---------------------------------------------------------------------------


def test_merge_to_field_dict_output_raises() -> None:
    field = _geotensor_field(np.float32)
    patcher = _patcher(aggregation=SpatialMeanStd())

    with pytest.raises(TypeError, match=r"SpatialMeanStd returned a dict"):
        patcher.merge_to_field(patcher.split(field), field)


def test_merge_to_field_shape_mismatch_raises() -> None:
    field = _geotensor_field()
    patcher = _patcher(aggregation=_FixedOutput(np.zeros((8, 8))))

    with pytest.raises(TypeError, match=r"returned shape \(8, 8\).*\(2, 8, 8\)"):
        patcher.merge_to_field([], field)


def test_merge_to_field_non_array_output_raises() -> None:
    field = _geotensor_field()
    patcher = _patcher(aggregation=_FixedOutput("s3://bucket/out.tif"))

    with pytest.raises(TypeError, match="returned str"):
        patcher.merge_to_field([], field)


def test_merge_to_field_needs_with_data() -> None:
    class _NoWithData:
        domain = _geotensor_field().domain

    with pytest.raises(TypeError, match="needs a field with `with_data`"):
        _patcher().merge_to_field([], _NoWithData())


def test_merge_still_returns_raw_output() -> None:
    # ADR-007: `merge` is unchanged — the bare aggregation output.
    field = _geotensor_field()
    patcher = _patcher()
    out = patcher.merge(patcher.split(field), field.domain)
    assert type(out) is np.ndarray
    assert out.dtype == np.float64


# ---------------------------------------------------------------------------
# merge_to_xarray
# ---------------------------------------------------------------------------


@needs_xarray
def test_merge_to_xarray_dict_output_raises_clear_typeerror() -> None:
    field = RioXarrayField(_rio_da())
    patcher = _patcher(aggregation=SpatialMeanStd())

    with pytest.raises(TypeError, match=r"merge_to_xarray .*returned a dict"):
        patcher.merge_to_xarray(patcher.split(field), field)


@needs_xarray
@pytest.mark.parametrize("dtype", [np.float32, np.uint8, np.int16])
def test_merge_to_xarray_preserves_source_dtype(dtype: Any) -> None:
    da = xr.DataArray(np.arange(8 * 8).reshape(8, 8).astype(dtype), dims=("lat", "lon"))
    field = XarrayField(da)
    patcher = _patcher()

    out = patcher.merge_to_xarray(patcher.split(field), field)

    assert out.dtype == dtype
    np.testing.assert_array_equal(out.values, da.values)


@needs_xarray
def test_merge_to_xarray_fractional_values_stay_float() -> None:
    da = xr.DataArray(np.arange(8 * 8, dtype=np.uint8).reshape(8, 8), dims=("y", "x"))
    field = XarrayField(da)
    patcher = _patcher()
    halves = (p.with_data(np.asarray(p.data) / 2) for p in patcher.split(field))

    out = patcher.merge_to_xarray(halves, field)

    assert out.dtype == np.float64
    np.testing.assert_array_equal(out.values, da.values / 2)


# ---------------------------------------------------------------------------
# Matched patcher
# ---------------------------------------------------------------------------


def test_matched_merge_to_field_rebuilds_every_source_on_primary_grid() -> None:
    primary = _geotensor_field(np.float32)
    # The secondary sits on its own 5 m grid over the primary's extent;
    # the coreg decimates its footprint chip onto the 10 m primary grid.
    secondary = RasterField(
        GeoTensor(
            values=np.full((2, 16, 16), 7, dtype=np.int16),
            transform=rasterio.Affine(5.0, 0.0, 500_000.0, 0.0, -5.0, 4_600_000.0),
            crs="EPSG:32630",
        )
    )
    mfield = MatchedField(
        primary=primary,
        secondaries={"sec": secondary},
        coreg={"sec": lambda raw, prim: np.asarray(raw)[..., ::2, ::2]},
    )
    mpatcher = MatchedSpatialPatcher(
        primary=_patcher(), secondary_aggregators={"sec": SpatialOverlapAdd()}
    )

    out = mpatcher.merge_to_field(mpatcher.split(mfield), mfield)

    assert set(out) == {PRIMARY_KEY, "sec"}
    for value in out.values():
        assert isinstance(value, GeoTensor)
        assert value.transform == _T  # the primary grid, not the secondary's
        assert str(value.crs) == "EPSG:32630"
    assert out[PRIMARY_KEY].values.dtype == np.float32
    # Each source keeps its own dtype when its values fit it.
    assert out["sec"].values.dtype == np.int16
    np.testing.assert_array_equal(out["sec"].values, 7)


def test_matched_merge_to_field_dict_output_raises() -> None:
    primary = _geotensor_field(np.float32)
    mfield = MatchedField(
        primary=primary,
        secondaries={"sec": _geotensor_field(np.float32)},
        coreg={"sec": lambda raw, prim: raw},
    )
    mpatcher = MatchedSpatialPatcher(
        primary=_patcher(), secondary_aggregators={"sec": SpatialMeanStd()}
    )

    with pytest.raises(TypeError, match="SpatialMeanStd returned a dict"):
        mpatcher.merge_to_field(mpatcher.split(mfield), mfield)
