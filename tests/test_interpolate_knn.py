"""Tests for sklearn-backed kNN / IDW interpolation."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from scipy.interpolate import griddata

from xr_toolz.interpolate import Grid, fillnan_idw, idw_to_grid, idw_to_points


def test_idw_to_points_returns_exact_value_at_source_location() -> None:
    out = idw_to_points(
        [0.0, 1.0],
        [0.0, 0.0],
        [10.0, 20.0],
        [1.0],
        [0.0],
        k=2,
        eps=0.0,
    )

    assert out[0] == pytest.approx(20.0)


def test_idw_to_grid_preserves_constant_field() -> None:
    grid = Grid(lon=np.linspace(0.0, 2.0, 5), lat=np.linspace(0.0, 2.0, 4))

    out = idw_to_grid(
        [0.0, 2.0, 0.0, 2.0],
        [0.0, 0.0, 2.0, 2.0],
        [7.5, 7.5, 7.5, 7.5],
        grid,
        k=4,
    )

    xr.testing.assert_allclose(out, xr.full_like(out, 7.5))


def test_idw_power_controls_nearest_neighbour_influence() -> None:
    mean = idw_to_points(
        [0.0, 2.0],
        [0.0, 0.0],
        [0.0, 10.0],
        [0.5],
        [0.0],
        k=2,
        power=0.0,
    )[0]
    local = idw_to_points(
        [0.0, 2.0],
        [0.0, 0.0],
        [0.0, 10.0],
        [0.5],
        [0.0],
        k=2,
        power=4.0,
    )[0]

    assert mean == pytest.approx(5.0)
    assert local < mean


def test_idw_max_distance_is_honoured_for_euclidean_and_haversine() -> None:
    euclidean = idw_to_points(
        [0.0],
        [0.0],
        [1.0],
        [2.0],
        [0.0],
        k=1,
        max_distance=1.0,
    )
    haversine = idw_to_points(
        [0.0],
        [0.0],
        [1.0],
        [1.0],
        [0.0],
        k=1,
        metric="haversine",
        max_distance=np.deg2rad(0.5),
    )

    assert np.isnan(euclidean[0])
    assert np.isnan(haversine[0])


def test_haversine_uses_dateline_shortcut_unlike_euclidean() -> None:
    src_lons = [179.0, 0.0]
    src_lats = [0.0, 0.0]
    src_values = [1.0, 10.0]

    euclidean = idw_to_points(src_lons, src_lats, src_values, [-179.0], [0.0], k=1)[0]
    haversine = idw_to_points(
        src_lons,
        src_lats,
        src_values,
        [-179.0],
        [0.0],
        k=1,
        metric="haversine",
    )[0]

    assert euclidean == pytest.approx(10.0)
    assert haversine == pytest.approx(1.0)


def test_fillnan_idw_fills_smooth_hole_with_bounded_rmse() -> None:
    lon = np.linspace(-2.0, 2.0, 31)
    lat = np.linspace(-1.5, 1.5, 25)
    lon_grid, lat_grid = np.meshgrid(lon, lat, indexing="xy")
    values = np.sin(lon_grid) + np.cos(lat_grid)
    expected = xr.DataArray(
        values, dims=("lat", "lon"), coords={"lat": lat, "lon": lon}
    )
    masked = expected.copy()
    hole = np.zeros(expected.shape, dtype=bool)
    hole[9:16, 12:19] = True
    masked.values[hole] = np.nan

    filled = fillnan_idw(masked, k=12, power=2.0)

    rmse = float(np.sqrt(np.mean((filled.values[hole] - expected.values[hole]) ** 2)))
    assert np.isfinite(filled.values[hole]).all()
    assert rmse < 0.2


def test_fillnan_idw_rmse_within_factor_of_griddata_linear() -> None:
    lon = np.linspace(-2.0, 2.0, 31)
    lat = np.linspace(-1.5, 1.5, 25)
    lon_grid, lat_grid = np.meshgrid(lon, lat, indexing="xy")
    values = np.sin(lon_grid) + np.cos(lat_grid)
    finite = np.ones(values.shape, dtype=bool)
    finite[9:16, 12:19] = False
    da = xr.DataArray(
        np.where(finite, values, np.nan),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
    )

    idw = fillnan_idw(da, k=12).values
    linear = griddata(
        np.column_stack([lon_grid[finite], lat_grid[finite]]),
        values[finite],
        np.column_stack([lon_grid[~finite], lat_grid[~finite]]),
        method="linear",
    )
    idw_rmse = np.sqrt(np.mean((idw[~finite] - values[~finite]) ** 2))
    linear_rmse = np.sqrt(np.mean((linear - values[~finite]) ** 2))

    assert idw_rmse < 1.0
    assert linear_rmse < 1.0
    assert idw_rmse <= 3.0 * linear_rmse


def test_fillnan_idw_fills_leading_dask_chunks_and_rejects_core_chunks() -> None:
    pytest.importorskip("dask.array")
    base = xr.DataArray(
        np.arange(18.0).reshape(3, 2, 3),
        dims=("time", "lat", "lon"),
        coords={"time": [0, 1, 2], "lat": [0.0, 1.0], "lon": [0.0, 1.0, 2.0]},
    )
    masked = base.copy()
    masked.values[:, 0, 1] = np.nan

    chunked = masked.chunk({"time": 1, "lat": -1, "lon": -1})
    filled = fillnan_idw(chunked, k=2).compute()
    eager = fillnan_idw(masked, k=2)
    xr.testing.assert_allclose(filled, eager)

    core_chunked = masked.chunk({"lat": 1, "lon": -1})
    with pytest.raises(ValueError, match="Core dimension"):
        fillnan_idw(core_chunked, k=2)


def test_idw_rejects_empty_sources_and_bad_arguments() -> None:
    with pytest.raises(ValueError, match="at least one finite source"):
        idw_to_points([0.0], [0.0], [np.nan], [0.0], [0.0])
    with pytest.raises(ValueError, match="positive integer"):
        idw_to_points([0.0], [0.0], [1.0], [0.0], [0.0], k=0)
