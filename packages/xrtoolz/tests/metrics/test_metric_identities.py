"""Metric identities (roadmap gate F1).

Two identities every verification metric must satisfy *exactly*, not to
a tolerance:

- the CRPS of a one-member ensemble is the absolute error ``|x - y|``;
- the RMSE of a field against itself is ``0``.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrtoolz.metrics import CRPS, RMSE, EnergyDistance, crps_ensemble, rmse


# The land box is NaN at every time step, so NaN-skipping reductions over
# ``time`` see empty slices there; numpy warns and returns NaN, as expected.
pytestmark = pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")


@pytest.fixture
def field() -> xr.DataArray:
    """A non-trivial float64 (time, lat, lon) field with a NaN-masked box."""
    rng = np.random.default_rng(20261017)
    values = 3.0 * rng.standard_normal((6, 5, 7)) + 280.0
    values[:, 1:3, 2:5] = np.nan  # land box: NaN at every time step
    values[2, 4, 6] = np.nan  # one isolated missing sample
    return xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={
            "time": np.arange(6),
            "lat": np.linspace(-10.0, 10.0, 5),
            "lon": np.linspace(0.0, 30.0, 7),
        },
        attrs={"units": "K"},
        name="sst",
    )


@pytest.fixture
def single_member(field: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """A one-member forecast and its reference on the same grid."""
    rng = np.random.default_rng(7)
    forecast = (field + rng.standard_normal(field.shape)).expand_dims(member=[0])
    return forecast, field


# ---------- CRPS with one member is the absolute error -------------------


def test_crps_single_member_equals_absolute_error(single_member) -> None:
    forecast, ref = single_member
    out = crps_ensemble(forecast, ref)
    expected = abs(forecast.isel(member=0, drop=True) - ref)
    assert out.dims == ("time", "lat", "lon")
    assert out.dtype == np.float64
    assert "member" not in out.coords
    # Exact, NaN in the same places.
    np.testing.assert_array_equal(out.values, expected.values)
    assert np.isnan(out.values).sum() == 6 * 2 * 3 + 1


def test_crps_single_member_equals_absolute_error_after_reduction(
    single_member,
) -> None:
    forecast, ref = single_member
    dims = ("time", "lat", "lon")
    out = crps_ensemble(forecast, ref, dims=dims)
    expected = abs(forecast.isel(member=0, drop=True) - ref).mean(dim=list(dims))
    assert out.ndim == 0
    assert float(out) == float(expected)


def test_crps_operator_single_member_equals_absolute_error(single_member) -> None:
    forecast, ref = single_member
    ds_e = forecast.to_dataset(name="sst")
    ds_r = ref.to_dataset(name="sst")
    expected = abs(forecast.isel(member=0, drop=True) - ref)

    out = CRPS("sst")(ds_e, ds_r)
    np.testing.assert_array_equal(out.values, expected.values)

    reduced = CRPS("sst", dims=("time", "lat"))(ds_e, ds_r)
    np.testing.assert_array_equal(
        reduced.values, expected.mean(dim=["time", "lat"]).values
    )


def test_crps_single_member_matches_xskillscore_layout(single_member) -> None:
    """Same dims, coords and (dropped) attrs as the xskillscore path."""
    import xskillscore as xs

    forecast, ref = single_member
    # Reference with its dims in another order, to pin the output order.
    ref_t = ref.transpose("lon", "time", "lat")
    out = crps_ensemble(forecast, ref_t)
    expected = xs.crps_ensemble(ref_t, forecast, member_dim="member", dim=[])
    assert out.dims == expected.dims
    assert out.attrs == expected.attrs == {}
    xr.testing.assert_identical(out, expected)


def test_crps_single_member_broadcasts_reference(field) -> None:
    """A reference without the time dim broadcasts as in the multi path."""
    forecast = field.expand_dims(member=[0])
    ref = field.isel(time=0, drop=True)
    out = crps_ensemble(forecast, ref)
    expected = abs(field - ref)
    np.testing.assert_array_equal(out.values, expected.transpose(*out.dims).values)


def test_crps_single_member_is_zero_against_itself(field) -> None:
    out = crps_ensemble(field.expand_dims(member=[0]), field)
    finite = np.isfinite(field.values)
    assert (out.values[finite] == 0.0).all()


def test_crps_multi_member_unchanged(field) -> None:
    """Two or more members still go through xskillscore, bit for bit."""
    import xskillscore as xs

    rng = np.random.default_rng(3)
    forecast = xr.concat(
        [field + rng.standard_normal(field.shape) for _ in range(4)],
        dim="member",
    )
    out = crps_ensemble(forecast, field)
    expected = xs.crps_ensemble(field, forecast, member_dim="member", dim=[])
    xr.testing.assert_identical(out, expected)


@pytest.mark.dask
def test_crps_single_member_dask_matches_eager(single_member) -> None:
    pytest.importorskip("dask.array")
    forecast, ref = single_member
    lazy = crps_ensemble(forecast.chunk({"time": 2}), ref.chunk({"time": 3}))
    assert lazy.chunks is not None
    np.testing.assert_array_equal(
        lazy.compute().values, crps_ensemble(forecast, ref).values
    )


def test_crps_rejects_empty_and_missing_member_dim(field) -> None:
    empty = field.expand_dims(member=[0]).isel(member=slice(0, 0))
    with pytest.raises(ValueError, match="need at least 1"):
        crps_ensemble(empty, field)
    with pytest.raises(ValueError, match="missing ensemble/sample dim"):
        crps_ensemble(field, field)


def test_two_sample_distances_still_need_two_samples(field) -> None:
    one = field.isel(time=slice(0, 1)).rename(time="member").to_dataset(name="x")
    with pytest.raises(ValueError, match="need at least 2"):
        EnergyDistance("x")(one, one)


# ---------- RMSE of a field against itself is zero -----------------------


@pytest.mark.parametrize("dim", ["time", ("lat", "lon"), ("time", "lat", "lon")])
def test_rmse_of_field_against_itself_is_zero(field, dim) -> None:
    out = rmse(field, field.copy(), dim=dim)
    finite = np.isfinite(out.values)
    # Only pixels whose every sample is NaN (the land box) stay NaN.
    assert finite.any()
    assert (out.values[finite] == 0.0).all()
    assert (
        out.isnull().sum()
        == np.isnan(field.values)
        .all(axis=tuple(field.get_axis_num(d) for d in np.atleast_1d(dim)))
        .sum()
    )


def test_rmse_global_of_field_against_itself_is_exactly_zero(field) -> None:
    out = rmse(field, field, dim=("time", "lat", "lon"))
    assert float(out) == 0.0


def test_rmse_operator_of_field_against_itself_is_zero(field) -> None:
    ds = field.to_dataset(name="sst")
    out = RMSE("sst", ("time", "lat", "lon"))(ds, ds.copy(deep=True))
    assert float(out) == 0.0

    per_pixel = RMSE("sst", "time")(ds, ds)
    finite = np.isfinite(per_pixel.values)
    assert (per_pixel.values[finite] == 0.0).all()
