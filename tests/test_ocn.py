"""Tests for :mod:`xr_toolz.ocn` — Layer-0 and Layer-1."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xr_toolz.core import Sequential
from xr_toolz.ocn import (
    absolute_vorticity,
    calculate_ssh_alongtrack,
    coriolis_parameter,
    divergence,
    enstrophy,
    geostrophic_velocities,
    kinetic_energy,
    okubo_weiss,
    relative_vorticity,
    shear_strain,
    strain_magnitude,
    streamfunction,
    tensor_strain,
    validate_ssh,
    validate_velocity,
)
from xr_toolz.ocn.operators import (
    CalculateSSHAlongtrack,
    Divergence,
    GeostrophicVelocities,
    KineticEnergy,
    OkuboWeiss,
    RelativeVorticity,
    Streamfunction,
    ValidateSSH,
)


@pytest.fixture
def ds_ssh_grid() -> xr.Dataset:
    """Smooth 2-D SSH field on a small lon/lat grid."""
    lon = np.linspace(-10.0, 10.0, 11)
    lat = np.linspace(20.0, 40.0, 9)
    lon2, lat2 = np.meshgrid(lon, lat, indexing="xy")
    ssh = 0.1 * np.sin(np.deg2rad(lon2)) * np.cos(np.deg2rad(lat2))
    return xr.Dataset(
        {"ssh": (("lat", "lon"), ssh)},
        coords={"lon": lon, "lat": lat},
    )


@pytest.fixture
def ds_uv_grid() -> xr.Dataset:
    lon = np.linspace(-10.0, 10.0, 11)
    lat = np.linspace(20.0, 40.0, 9)
    lon2, lat2 = np.meshgrid(lon, lat, indexing="xy")
    u = 0.5 * np.cos(np.deg2rad(lat2))
    v = 0.1 * np.sin(np.deg2rad(lon2))
    return xr.Dataset(
        {"u": (("lat", "lon"), u), "v": (("lat", "lon"), v)},
        coords={"lon": lon, "lat": lat},
    )


# ---------- L0 primitives --------------------------------------------------


def test_coriolis_parameter_sign_changes_across_equator():
    lats = xr.DataArray(np.array([-45.0, 0.0, 45.0]), dims="lat")
    f = coriolis_parameter(lats)
    values = f.metpy.dequantify().values if hasattr(f, "metpy") else np.asarray(f)
    assert float(values[0]) < 0.0
    assert float(values[1]) == pytest.approx(0.0, abs=1e-10)
    assert float(values[2]) > 0.0


def test_coriolis_parameter_numerical_value_locks_in_radian_input():
    """Guard against unit-convention regressions.

    ``metpy.calc.coriolis_parameter`` expects the latitude input in
    radians (or a pint-Quantity with angle units). Our API converts
    degrees with ``np.deg2rad`` before calling metpy. At 45° latitude
    the Coriolis parameter is ``2 * Ω * sin(45°) ≈ 1.031e-4 s⁻¹``.
    """
    expected = 2.0 * 7.2921159e-5 * np.sin(np.pi / 4.0)  # ~1.0313e-4
    f = coriolis_parameter(xr.DataArray(np.array([45.0]), dims="lat"))
    values = f.metpy.dequantify().values if hasattr(f, "metpy") else np.asarray(f)
    assert float(values[0]) == pytest.approx(expected, rel=1e-4)


def test_streamfunction_produces_psi(ds_ssh_grid):
    out = streamfunction(ds_ssh_grid)
    assert "psi" in out.data_vars
    # psi = (g / f0) * ssh; for f0 ~ 7e-5 and g ~ 9.81, scale ~ 1.4e5
    assert float(np.abs(out["psi"]).max()) > float(np.abs(ds_ssh_grid["ssh"]).max())


def test_geostrophic_velocities_produces_u_and_v(ds_ssh_grid):
    # geostrophic_velocities takes SSH directly (metpy.geostrophic_wind
    # applies the g/f scaling internally). Passing the stream function
    # would double-apply the scaling.
    uv = geostrophic_velocities(ds_ssh_grid, variable="ssh")
    assert set(uv.data_vars) == {"u", "v"}


def test_geostrophic_velocities_magnitude_bounded_by_gravity_scaling(ds_ssh_grid):
    """Sanity check: with SSH of O(0.1 m) over O(1000 km) scales at
    mid-latitudes, geostrophic speeds are O(0.01–0.1 m/s)."""
    uv = geostrophic_velocities(ds_ssh_grid, variable="ssh")
    speed = np.sqrt(uv["u"] ** 2 + uv["v"] ** 2)
    assert float(speed.max()) < 10.0  # m/s — much smaller than any pathology


def test_kinetic_energy_is_non_negative(ds_uv_grid):
    ke = kinetic_energy(ds_uv_grid)
    assert bool((ke["ke"] >= 0.0).all())


def test_relative_vorticity_shape(ds_uv_grid):
    zeta = relative_vorticity(ds_uv_grid)
    assert "vort_r" in zeta.data_vars
    assert zeta["vort_r"].dims == ("lat", "lon")


def test_absolute_vorticity_shape(ds_uv_grid):
    out = absolute_vorticity(ds_uv_grid)
    assert "vort_a" in out.data_vars


def test_divergence_shape(ds_uv_grid):
    out = divergence(ds_uv_grid)
    assert "div" in out.data_vars


def test_enstrophy_non_negative(ds_uv_grid):
    zeta = relative_vorticity(ds_uv_grid)
    ens = enstrophy(zeta)
    assert bool((ens["ens"] >= 0.0).all())


def test_shear_tensor_strain_shapes(ds_uv_grid):
    sh = shear_strain(ds_uv_grid)
    st = tensor_strain(ds_uv_grid)
    assert "shear_strain" in sh.data_vars
    assert "tensor_strain" in st.data_vars


def test_strain_magnitude_non_negative(ds_uv_grid):
    out = strain_magnitude(ds_uv_grid)
    assert bool((out["strain"] >= 0.0).all())


def test_okubo_weiss_same_shape_as_uv(ds_uv_grid):
    ow = okubo_weiss(ds_uv_grid)
    assert ow["ow"].dims == ds_uv_grid["u"].dims
    assert ow["ow"].shape == ds_uv_grid["u"].shape


# ---------- SSH composition ------------------------------------------------


def test_calculate_ssh_alongtrack_linear_combination():
    ds = xr.Dataset(
        {
            "sla_filtered": ("track", np.array([1.0, 2.0, 3.0])),
            "mdt": ("track", np.array([0.5, 0.5, 0.5])),
            "lwe": ("track", np.array([0.1, 0.1, 0.1])),
        }
    )
    out = calculate_ssh_alongtrack(ds)
    np.testing.assert_allclose(out["ssh"].values, np.array([1.4, 2.4, 3.4]))
    assert out["ssh"].attrs["units"] == "m"


# ---------- validation -----------------------------------------------------


def test_validate_ssh_sets_attrs():
    ds = xr.Dataset({"ssh": ("i", np.arange(3.0))})
    out = validate_ssh(ds)
    assert out["ssh"].attrs["standard_name"] == "sea_surface_height"


def test_validate_velocity_sets_attrs():
    ds = xr.Dataset({"u": ("i", [0.0]), "v": ("i", [0.0])})
    out = validate_velocity(ds)
    assert out["u"].attrs["standard_name"] == "sea_water_x_velocity"
    assert out["v"].attrs["standard_name"] == "sea_water_y_velocity"


# ---------- L1 operators ---------------------------------------------------


def test_streamfunction_operator(ds_ssh_grid):
    psi = Streamfunction()(ds_ssh_grid)
    assert "psi" in psi.data_vars


def test_geostrophic_velocities_operator(ds_ssh_grid):
    uv = GeostrophicVelocities(variable="ssh")(ds_ssh_grid)
    assert set(uv.data_vars) == {"u", "v"}


def test_streamfunction_is_diagnostic_only(ds_ssh_grid):
    """Streamfunction is a separate diagnostic — not a pipeline step
    feeding geostrophic_velocities (the latter takes SSH directly)."""
    psi = Streamfunction()(ds_ssh_grid)
    assert "psi" in psi.data_vars
    uv = GeostrophicVelocities(variable="ssh")(ds_ssh_grid)
    assert set(uv.data_vars) == {"u", "v"}


def test_ocn_pipeline_full_eddy_metrics(ds_uv_grid):
    pipe = Sequential(
        [
            KineticEnergy(),
        ]
    )
    out = pipe(ds_uv_grid)
    assert "ke" in out.data_vars


def test_divergence_vorticity_okubo_in_sequential(ds_uv_grid):
    zeta = RelativeVorticity()(ds_uv_grid)
    div = Divergence()(ds_uv_grid)
    ow = OkuboWeiss()(ds_uv_grid)
    assert "vort_r" in zeta.data_vars
    assert "div" in div.data_vars
    assert "ow" in ow.data_vars


def test_validate_ssh_operator_config_round_trip():
    op = ValidateSSH(variable="ssh")
    assert op.get_config() == {"variable": "ssh"}
    assert repr(op) == "ValidateSSH(variable='ssh')"


def test_calculate_ssh_alongtrack_operator():
    ds = xr.Dataset(
        {
            "sla_filtered": ("track", [1.0]),
            "mdt": ("track", [0.5]),
            "lwe": ("track", [0.1]),
        }
    )
    out = CalculateSSHAlongtrack()(ds)
    assert float(out["ssh"].values[0]) == pytest.approx(1.4)
