"""Tests for :mod:`xrtoolz.atm._src.vertical`."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrtoolz.atm import (
    column_integral,
    hypsometric_height,
    pbl_height_bulk_richardson,
)
from xrtoolz.atm._src.constants import R_D, G


# ---------- column_integral ----------------------------------------------


@pytest.mark.parametrize("method", ["trapezoid", "sum"])
def test_uniform_field_on_endpoint_inclusive_grid(method):
    # Non-uniform endpoint-inclusive grid on [0, 100]: both rules must give
    # c * (z1 - z0), i.e. no n_z / (n_z - 1) over-count.
    z = np.array([0.0, 10.0, 30.0, 60.0, 100.0])
    field = xr.DataArray(np.full((3, z.size), 2.5), dims=("t", "z"), coords={"z": z})
    out = column_integral(field, method=method)
    assert out.dims == ("t",)
    np.testing.assert_allclose(out, 2.5 * 100.0, rtol=1e-12)


@pytest.mark.parametrize("method", ["trapezoid", "sum"])
def test_linear_field_matches_analytic_integral(method):
    z = np.linspace(0.0, 100.0, 11)
    field = xr.DataArray(3.0 + 0.5 * z, dims="z", coords={"z": z})
    # ∫ (3 + 0.5 z) dz on [0, 100] = 300 + 0.25 * 100²
    np.testing.assert_allclose(
        column_integral(field, method=method), 300.0 + 2500.0, rtol=1e-12
    )


def test_sum_versus_naive_sum_dz_on_endpoint_grid():
    # The retired ``sum * dz`` variant over-counts by n_z / (n_z - 1).
    z = np.linspace(0.0, 100.0, 6)
    field = xr.DataArray(np.ones(6), dims="z", coords={"z": z})
    naive = float(field.sum("z") * 20.0)
    assert naive == pytest.approx(120.0)
    assert float(column_integral(field, method="sum")) == pytest.approx(100.0)


def test_column_integral_with_per_column_coordinate_stays_lazy():
    pytest.importorskip("dask")
    z = np.array([0.0, 10.0, 30.0, 60.0, 100.0])
    z_agl = np.broadcast_to(z[None, :, None], (2, 5, 3)) + np.arange(3)[None, None, :]
    field = xr.DataArray(
        np.ones((2, 5, 3)),
        dims=("t", "z", "x"),
        coords={"z_agl": (("t", "z", "x"), z_agl.copy())},
    ).chunk({"t": 1})
    out = column_integral(field, coord="z_agl")
    assert out.chunks is not None
    np.testing.assert_allclose(out.compute(), 100.0)
    # A DataArray coordinate is accepted directly, too.
    out2 = column_integral(field, coord=field["z_agl"])
    np.testing.assert_allclose(out2.compute(), 100.0)


@pytest.mark.parametrize("method", ["trapezoid", "sum"])
def test_column_integral_nan_level_makes_column_nan(method):
    # A NaN at any level must poison that column (no silent NaN skipping),
    # while untouched columns integrate as before.
    z = np.linspace(0.0, 100.0, 6)
    values = np.ones((2, z.size))
    values[0, 3] = np.nan
    field = xr.DataArray(values, dims=("x", "z"), coords={"z": z})
    out = column_integral(field, method=method)
    assert np.isnan(float(out[0]))
    assert float(out[1]) == pytest.approx(100.0)


def test_column_integral_errors():
    z = np.array([0.0, 1.0])
    field = xr.DataArray(np.ones(2), dims="z")
    with pytest.raises(ValueError, match="no coordinate"):
        column_integral(field)
    with pytest.raises(ValueError, match="not in DataArray dims"):
        column_integral(field, dim="q")
    with pytest.raises(ValueError, match="at least two"):
        column_integral(xr.DataArray([1.0], dims="z", coords={"z": [0.0]}))
    with pytest.raises(ValueError, match="method"):
        column_integral(field.assign_coords(z=z), method="simpson")  # type: ignore[arg-type]


# ---------- hypsometric_height -------------------------------------------


@pytest.fixture
def ds_isothermal() -> xr.Dataset:
    p = np.array([1000.0, 850.0, 700.0, 500.0, 300.0, 100.0])
    return xr.Dataset(
        {"temperature": (("level", "x"), np.full((p.size, 2), 250.0))},
        coords={"level": p},
    ).assign(sp=("x", [1013.0, 980.0]))


def test_isothermal_closed_form(ds_isothermal):
    out = hypsometric_height(ds_isothermal, pressure="level")
    expected = (
        R_D
        * 250.0
        / G
        * np.log(
            ds_isothermal["sp"].values[None, :] / ds_isothermal["level"].values[:, None]
        )
    )
    np.testing.assert_allclose(out["height"], expected, rtol=1e-12)
    assert out["height"].dims == ("level", "x")
    assert out["height"].attrs["standard_name"] == "geopotential_height"
    assert out["height"].attrs["units"] == "m"


def test_height_is_monotone_and_top_first_ordering_matches(ds_isothermal):
    bottom_first = hypsometric_height(ds_isothermal, pressure="level")["height"]
    assert (bottom_first.diff("level") > 0).all()
    flipped = ds_isothermal.isel(level=slice(None, None, -1))
    top_first = hypsometric_height(flipped, pressure="level")["height"]
    xr.testing.assert_allclose(
        top_first, bottom_first.isel(level=slice(None, None, -1))
    )


def test_surface_height_offset_and_humidity(ds_isothermal):
    ds = ds_isothermal.assign(zs=("x", [100.0, 250.0]))
    dry = hypsometric_height(ds, pressure="level")["height"]
    lifted = hypsometric_height(ds, pressure="level", surface_height="zs")["height"]
    np.testing.assert_allclose(lifted - dry, ds["zs"].broadcast_like(dry), rtol=1e-12)
    # Moist air is lighter: T_v = T (1 + 0.61 q) thickens every layer.
    moist = ds.assign(q=(("level", "x"), np.full((6, 2), 0.01)))
    wet = hypsometric_height(moist, pressure="level", specific_humidity="q")["height"]
    np.testing.assert_allclose(wet, dry * (1.0 + 0.61 * 0.01), rtol=1e-12)


def test_hypsometric_nan_temperature_invalidates_levels_above(ds_isothermal):
    # NaN at level 2 of column 0: levels 0-1 are still anchored to the
    # surface, level 2 and everything above are NaN; column 1 is untouched.
    ds = ds_isothermal.copy(deep=True)
    ds["temperature"][2, 0] = np.nan
    clean = hypsometric_height(ds_isothermal, pressure="level")["height"]
    out = hypsometric_height(ds, pressure="level")["height"]
    np.testing.assert_allclose(out[:2, 0], clean[:2, 0], rtol=1e-12)
    assert np.isnan(out[2:, 0]).all()
    np.testing.assert_allclose(out[:, 1], clean[:, 1], rtol=1e-12)


def test_hypsometric_requires_level_dim(ds_isothermal):
    with pytest.raises(ValueError, match="level"):
        hypsometric_height(ds_isothermal, pressure="sp")


# ---------- pbl_height_bulk_richardson -----------------------------------


@pytest.fixture
def ds_pbl() -> xr.Dataset:
    # θ_v = θ0 + Γ z and |V|² = U² z / z_ref make Ri_b linear in z, so the
    # level-wise linear interpolation is exact:
    #   Ri_b(z) = g Γ z_ref z / (θ0 U²)  ->  h = Ri_crit θ0 U² / (g Γ z_ref)
    z = np.linspace(0.0, 2000.0, 41)
    theta0, gamma, u0, z_ref = 300.0, 0.005, 5.0, 1000.0
    ds = xr.Dataset(
        {
            "theta_v": ("level", theta0 + gamma * z),
            "u": ("level", u0 * np.sqrt(z / z_ref)),
            "v": ("level", np.zeros_like(z)),
        },
        coords={"z_agl": ("level", z)},
    )
    ds.attrs["h_closed_form"] = lambda ri: ri * theta0 * u0**2 / (G * gamma * z_ref)
    return ds


def test_pbl_height_closed_form(ds_pbl):
    out = pbl_height_bulk_richardson(ds_pbl, theta_v="theta_v")
    assert "level" not in out["pbl_height"].dims
    assert float(out["pbl_height"]) == pytest.approx(
        ds_pbl.attrs["h_closed_form"](0.25), rel=1e-12
    )
    assert out["pbl_height"].attrs["standard_name"] == (
        "atmosphere_boundary_layer_thickness"
    )


def test_pbl_height_ri_crit_sensitivity(ds_pbl):
    for ri_crit in (0.1, 0.5, 1.0):
        out = pbl_height_bulk_richardson(ds_pbl, theta_v="theta_v", ri_crit=ri_crit)
        assert float(out["pbl_height"]) == pytest.approx(
            ds_pbl.attrs["h_closed_form"](ri_crit), rel=1e-12
        )


def test_pbl_height_nan_when_no_crossing(ds_pbl):
    # Ri_b tops out at ~13 on this column; a threshold of 100 is never reached.
    out = pbl_height_bulk_richardson(ds_pbl, theta_v="theta_v", ri_crit=100.0)
    assert np.isnan(float(out["pbl_height"]))


def test_pbl_height_per_column_and_lazy(ds_pbl):
    pytest.importorskip("dask")
    ds = ds_pbl.expand_dims(x=3).chunk({"x": 1})
    out = pbl_height_bulk_richardson(ds, theta_v="theta_v")["pbl_height"]
    assert out.chunks is not None
    np.testing.assert_allclose(out.compute(), ds_pbl.attrs["h_closed_form"](0.25))


def test_pbl_height_invariant_to_constant_wind_offset(ds_pbl):
    # The shear is referenced to the surface wind, so a uniform offset in
    # (u, v) leaves Ri_b -- and hence h -- unchanged.
    shifted = ds_pbl.assign(u=ds_pbl["u"] + 3.0, v=ds_pbl["v"] - 2.0)
    out = pbl_height_bulk_richardson(shifted, theta_v="theta_v")["pbl_height"]
    assert float(out) == pytest.approx(ds_pbl.attrs["h_closed_form"](0.25), rel=1e-12)


def test_pbl_height_with_vertically_chunked_input(ds_pbl):
    pytest.importorskip("dask")
    ds = ds_pbl.expand_dims(x=3).chunk({"x": 1, "level": 10})
    out = pbl_height_bulk_richardson(ds, theta_v="theta_v")["pbl_height"]
    assert out.chunks is not None
    np.testing.assert_allclose(out.compute(), ds_pbl.attrs["h_closed_form"](0.25))


def test_pbl_height_top_first_profile_matches_flipped(ds_pbl):
    top_first = ds_pbl.isel(level=slice(None, None, -1))
    out = pbl_height_bulk_richardson(top_first, theta_v="theta_v")["pbl_height"]
    assert float(out) == pytest.approx(ds_pbl.attrs["h_closed_form"](0.25), rel=1e-12)


def test_pbl_height_rejects_mixed_orientation(ds_pbl):
    z = ds_pbl["z_agl"].values
    mixed = ds_pbl.expand_dims(x=2).assign_coords(
        z_agl=(("level", "x"), np.stack([z, z[::-1]], axis=-1))
    )
    with pytest.raises(ValueError, match="orientation"):
        pbl_height_bulk_richardson(mixed, theta_v="theta_v")


def test_pbl_unstable_calm_level_does_not_cross(ds_pbl):
    # Level 1 is unstable (theta_v below the surface value) with no shear
    # relative to the surface: Ri_b -> -inf there, which must not count as
    # a crossing of a positive Ri_crit. The rest of the column is the
    # closed-form profile, so h (Ri_crit = 1 sits between levels 3 and 4)
    # is unaffected.
    theta_v = ds_pbl["theta_v"].values.copy()
    u = ds_pbl["u"].values.copy()
    theta_v[1] -= 2.0
    u[1] = u[0]
    ds = ds_pbl.assign(theta_v=("level", theta_v), u=("level", u))
    out = pbl_height_bulk_richardson(ds, theta_v="theta_v", ri_crit=1.0)
    assert float(out["pbl_height"]) == pytest.approx(
        ds_pbl.attrs["h_closed_form"](1.0), rel=1e-12
    )
    # An entirely calm, unstable column never crosses at all.
    calm = ds_pbl.assign(
        theta_v=300.0 - 0.005 * ds_pbl["z_agl"], u=xr.zeros_like(ds_pbl["u"])
    )
    assert np.isnan(
        float(pbl_height_bulk_richardson(calm, theta_v="theta_v")["pbl_height"])
    )


def test_pbl_requires_level_dim(ds_pbl):
    with pytest.raises(ValueError, match="level"):
        pbl_height_bulk_richardson(ds_pbl, theta_v="theta_v", level="z")
