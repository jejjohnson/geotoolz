"""Field adapters: cached ``domain`` and positional vector indexers (#182)."""

from __future__ import annotations

import numpy as np
import pytest


gpd = pytest.importorskip("geopandas")
shapely = pytest.importorskip("shapely")
try:
    import xarray as xr
except ImportError:  # the geopandas-backed tests still run
    xr = None

needs_xarray = pytest.mark.skipif(xr is None, reason="needs the [grid] extra")

from geopatcher import (
    PointDomain,
    SpatialBoxcar,
    SpatialKNNGraph,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRandom,
    SpatialRectangular,
    SpatialRegularStride,
)
from geopatcher._src.fields.geopandas import GeoPandasField
from geopatcher._src.fields.xarray import XarrayField


def _points_gdf(index: list[int] | None = None) -> gpd.GeoDataFrame:
    pts = [shapely.Point(float(i), float(i % 3)) for i in range(5)]
    return gpd.GeoDataFrame(
        {"v": np.arange(5.0)}, geometry=pts, crs="EPSG:4326", index=index
    )


def _xvec_ds(geometry_dim: str = "geometry") -> xr.Dataset:
    pytest.importorskip("xvec")
    pts = np.array([shapely.Point(float(i), 0.0) for i in range(4)], dtype=object)
    ds = xr.Dataset(
        {"v": ((geometry_dim,), np.arange(4.0))}, coords={geometry_dim: pts}
    )
    return ds.xvec.set_geom_indexes(geometry_dim, crs="EPSG:4326")


def _grid_da() -> xr.DataArray:
    return xr.DataArray(
        np.arange(24.0).reshape(4, 6),
        dims=("lat", "lon"),
        coords={"lat": np.arange(4.0), "lon": np.arange(6.0)},
    )


def _fields() -> list:
    out = [
        GeoPandasField(_points_gdf()),
        GeoPandasField(_points_gdf(), as_points=True),
    ]
    if xr is None:
        return out
    out.append(XarrayField(_grid_da()))
    try:
        from geopatcher._src.fields.xvec import XvecField

        out.append(XvecField(_xvec_ds()))
    except pytest.skip.Exception:
        pass
    try:
        from geopatcher._src.fields.dask import DaskField

        pytest.importorskip("dask")
        out.append(DaskField(_grid_da()))
    except pytest.skip.Exception:
        pass
    return out


@pytest.mark.parametrize("field", _fields(), ids=lambda f: type(f).__name__)
def test_domain_is_cached(field) -> None:
    # `patch_at` evaluates `field.domain` per patch; it must not rebuild a
    # cKDTree / spatial index / coord dict every time.
    assert field.domain is field.domain


class TestGeoPandasField:
    def test_select_is_positional_with_integer_labels(self) -> None:
        # Integer labels that are also valid positions must not be read
        # by label: positions 0 and 1 are labels 40 and 30.
        gdf = _points_gdf(index=[40, 30, 20, 10, 0])
        sub = GeoPandasField(gdf).select([0, 1]).gdf
        assert list(sub.index) == [40, 30]
        np.testing.assert_array_equal(sub["v"].to_numpy(), [0.0, 1.0])

    def test_select_rejects_labels(self) -> None:
        gdf = _points_gdf(index=[40, 30, 20, 10, 0])
        with pytest.raises(IndexError):
            GeoPandasField(gdf).select([40])

    def test_select_boolean_mask(self) -> None:
        mask = np.array([True, False, True, False, False])
        sub = GeoPandasField(_points_gdf()).select(mask).gdf
        np.testing.assert_array_equal(sub["v"].to_numpy(), [0.0, 2.0])

    def test_with_data_one_value_per_row(self) -> None:
        out = GeoPandasField(_points_gdf()).with_data(np.arange(5) * 2.0)
        np.testing.assert_array_equal(out.gdf["_value"].to_numpy(), np.arange(5) * 2.0)

    @pytest.mark.parametrize("shape", [(5, 2), (4,), ()])
    def test_with_data_rejects_non_1d_or_wrong_length(self, shape) -> None:
        with pytest.raises(ValueError, match="1-D array with one value per row"):
            GeoPandasField(_points_gdf()).with_data(np.zeros(shape))

    def test_knn_split_on_point_domain(self) -> None:
        field = GeoPandasField(_points_gdf(), as_points=True)
        assert isinstance(field.domain, PointDomain)
        patcher = SpatialPatcher(
            geometry=SpatialKNNGraph(k=2),
            sampler=SpatialRandom(n_samples=3, seed=0),
            window=SpatialBoxcar(),
            aggregation=SpatialOverlapAdd(),
        )
        patches = list(patcher.split(field))
        assert patches


class TestXvecField:
    def test_geometry_dim_is_public_and_propagates(self) -> None:
        from geopatcher._src.fields.xvec import XvecField

        field = XvecField(_xvec_ds("stations"), geometry_dim="stations")
        assert field.domain.coords.shape == (4, 2)
        sub = field.select([1, 2])
        assert sub.geometry_dim == "stations"
        assert sub.domain.coords.shape == (2, 2)
        out = field.with_data(np.arange(4.0))
        assert out.geometry_dim == "stations"
        np.testing.assert_array_equal(out.ds["_value"].values, np.arange(4.0))

    def test_private_ctor_name_removed(self) -> None:
        from geopatcher._src.fields.xvec import XvecField

        with pytest.raises(TypeError):
            XvecField(_xvec_ds(), _geometry_dim="geometry")  # type: ignore[call-arg]


@needs_xarray
def test_xarray_rectangular_split_merge_with_cached_domain() -> None:
    da = _grid_da()
    field = XarrayField(da)
    patcher = SpatialPatcher(
        geometry=SpatialRectangular(size=(2, 3)),
        sampler=SpatialRegularStride(step=(2, 3)),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )
    recon = patcher.merge_to_xarray(list(patcher.split(field)), field)
    np.testing.assert_allclose(recon.values, da.values)
