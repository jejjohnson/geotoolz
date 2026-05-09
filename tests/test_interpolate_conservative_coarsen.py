from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xr_toolz.interpolate import coarsen, coarsen_conservative
from xr_toolz.interpolate.operators import Coarsen


def _area_weights(da: xr.DataArray) -> xr.DataArray:
    weights = xr.DataArray(
        np.cos(np.deg2rad(da["lat"])), dims=("lat",), coords={"lat": da["lat"]}
    )
    return xr.ones_like(da) * weights


def test_conservative_coarsen_matches_uniform_at_equator() -> None:
    da = xr.DataArray(
        np.arange(16, dtype=float).reshape(4, 4),
        dims=("lat", "lon"),
        coords={"lat": np.zeros(4), "lon": np.arange(4)},
    )

    result = coarsen_conservative(da, {"lat": 2, "lon": 2})
    expected = coarsen(da, {"lat": 2, "lon": 2})

    xr.testing.assert_allclose(result, expected)


def test_conservative_coarsen_preserves_cos_lat_integral() -> None:
    lat = np.linspace(-87.5, 87.5, 36)
    lon = np.linspace(0.0, 355.0, 72)
    data = np.broadcast_to((1.0 / np.cos(np.deg2rad(lat)))[:, None], (36, 72))
    da = xr.DataArray(data, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    factor = {"lat": 4, "lon": 6}
    area = _area_weights(da)
    coarse_area = area.coarsen(dim=factor, boundary="trim").sum()

    fine_total = (da * area).sum()
    conservative_total = (coarsen_conservative(da, factor) * coarse_area).sum()
    uniform_total = (coarsen(da, factor) * coarse_area).sum()

    xr.testing.assert_allclose(conservative_total, fine_total)
    assert not np.isclose(float(uniform_total), float(fine_total), rtol=1e-3)


def test_conservative_coarsen_has_high_latitude_bias_vs_uniform() -> None:
    lat = np.array([50.0, 55.0, 60.0, 65.0])
    values = 1.0 / np.cos(np.deg2rad(lat))
    da = xr.DataArray(values, dims=("lat",), coords={"lat": lat})

    conservative = coarsen_conservative(da, {"lat": 4})
    uniform = coarsen(da, {"lat": 4})
    uniform_val = float(uniform.item())
    conservative_val = float(conservative.item())
    relative_bias = abs(uniform_val - conservative_val) / abs(conservative_val)

    assert relative_bias > 0.02


def test_conservative_coarsen_renormalizes_with_nan_values() -> None:
    lat = np.array([50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0])
    values = np.array([1.0, np.nan, 3.0, 4.0, np.nan, np.nan, np.nan, np.nan])
    da = xr.DataArray(values, dims=("lat",), coords={"lat": lat})

    result = coarsen_conservative(da, {"lat": 4})
    weights = np.cos(np.deg2rad(lat[:4]))
    finite_indices = [0, 2, 3]
    expected = (values[:4][finite_indices] * weights[finite_indices]).sum() / weights[
        finite_indices
    ].sum()

    assert np.isclose(float(result.isel(lat=0)), expected)
    assert np.isnan(float(result.isel(lat=1)))


def test_conservative_coarsen_without_lat_factor_matches_uniform_mean() -> None:
    da = xr.DataArray(
        np.arange(12, dtype=float).reshape(4, 3),
        dims=("time", "lat"),
        coords={"time": np.arange(4), "lat": [-30.0, 0.0, 30.0]},
    )

    result = coarsen_conservative(da, {"time": 2})
    expected = coarsen(da, {"time": 2})

    xr.testing.assert_allclose(result, expected)


def test_conservative_coarsen_dataset_preserves_unrelated_variables() -> None:
    ds = xr.Dataset(
        {
            "with_lat": (("lat",), np.arange(4, dtype=float)),
            "without_lat": (("time",), np.arange(3, dtype=float)),
        },
        coords={"lat": np.arange(4, dtype=float), "time": np.arange(3)},
    )

    result = coarsen_conservative(ds, {"lat": 2})

    xr.testing.assert_allclose(
        result["with_lat"], coarsen_conservative(ds["with_lat"], {"lat": 2})
    )
    xr.testing.assert_identical(result["without_lat"], ds["without_lat"])


def test_conservative_coarsen_boundary_modes() -> None:
    da = xr.DataArray(
        np.arange(5, dtype=float), dims=("lat",), coords={"lat": np.arange(5)}
    )

    result = coarsen_conservative(da, {"lat": 2}, boundary="trim")

    assert result.sizes["lat"] == 2
    with pytest.raises(ValueError, match="boundary='exact'"):
        coarsen_conservative(da, {"lat": 2}, boundary="exact")


def test_conservative_coarsen_operator_round_trips_config() -> None:
    op = Coarsen(factor={"lat": 2}, conservative=True, lat="latitude")

    cfg = op.get_config()
    round_tripped = Coarsen(**cfg)

    assert round_tripped.get_config() == cfg


def test_conservative_coarsen_operator_uses_custom_lat_name() -> None:
    latitude = np.array([50.0, 55.0, 60.0, 65.0])
    da = xr.DataArray(
        1.0 / np.cos(np.deg2rad(latitude)),
        dims=("latitude",),
        coords={"latitude": latitude},
    )
    op = Coarsen(factor={"latitude": 4}, conservative=True, lat="latitude")

    xr.testing.assert_allclose(
        op(da),
        coarsen_conservative(da, {"latitude": 4}, lat="latitude"),
    )


def test_conservative_coarsen_operator_rejects_non_mean_method() -> None:
    with pytest.raises(ValueError, match="method='mean'"):
        Coarsen(factor={"lat": 2}, method="max", conservative=True)


def test_conservative_coarsen_rejects_misaligned_lat_chunks() -> None:
    pytest.importorskip("dask.array")
    da = xr.DataArray(
        np.arange(8, dtype=float), dims=("lat",), coords={"lat": np.arange(8)}
    ).chunk({"lat": 3})

    with pytest.raises(ValueError, match=r"chunks along 'lat'.*\(3, 3, 2\)"):
        coarsen_conservative(da, {"lat": 2})


def test_conservative_coarsen_accepts_aligned_lat_chunks() -> None:
    pytest.importorskip("dask.array")
    da = xr.DataArray(
        np.arange(8, dtype=float), dims=("lat",), coords={"lat": np.arange(8)}
    ).chunk({"lat": 4})

    result = coarsen_conservative(da, {"lat": 2})
    expected = coarsen_conservative(da.compute(), {"lat": 2})

    xr.testing.assert_allclose(result.compute(), expected)
