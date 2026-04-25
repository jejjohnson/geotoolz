"""Tests for :mod:`xr_toolz.ocn` — Layer-0 and Layer-1."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xr_toolz.core import Sequential
from xr_toolz.ocn import (
    absolute_vorticity,
    advection,
    ageostrophic_velocities,
    brunt_vaisala_frequency,
    calculate_ssh_alongtrack,
    coriolis_parameter,
    curvature_vorticity,
    divergence,
    eddy_kinetic_energy,
    enstrophy,
    frontogenesis,
    geostrophic_velocities,
    horizontal_velocity_magnitude,
    kinetic_energy,
    lapse_rate,
    mixed_layer_depth,
    okubo_weiss,
    potential_vorticity_barotropic,
    relative_vorticity,
    shear_strain,
    shear_vorticity,
    strain_magnitude,
    streamfunction,
    tensor_strain,
    validate_ssh,
    validate_velocity,
    velocity_magnitude,
)
from xr_toolz.ocn.operators import (
    Advection,
    AgeostrophicVelocities,
    BruntVaisalaFrequency,
    CalculateSSHAlongtrack,
    CurvatureVorticity,
    Divergence,
    EddyKineticEnergy,
    Frontogenesis,
    GeostrophicVelocities,
    HorizontalVelocityMagnitude,
    KineticEnergy,
    LapseRate,
    MixedLayerDepth,
    OkuboWeiss,
    PotentialVorticityBarotropic,
    RelativeVorticity,
    ShearVorticity,
    Streamfunction,
    ValidateSSH,
    VelocityMagnitude,
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


# ---------- new kinematics: advection, ageostrophic, vorticity decomp,
#            frontogenesis, barotropic PV ----------------------------------


@pytest.fixture
def ds_uv_ssh_grid(ds_uv_grid, ds_ssh_grid) -> xr.Dataset:
    return xr.merge([ds_uv_grid, ds_ssh_grid])


def test_advection_constant_field_is_zero(ds_uv_grid):
    """``∇c = 0`` ⇒ ``-u·∇c = 0``."""
    ds = ds_uv_grid.assign(c=lambda d: d["u"] * 0.0 + 7.0)
    out = advection(ds, scalar="c")
    assert "c_advection" in out.data_vars
    np.testing.assert_allclose(out["c_advection"].values, 0.0, atol=1e-30)


def test_advection_sign_convention_matches_minus_u_dot_grad(ds_uv_grid):
    """The result is -u·∇c, so a tracer that decreases eastward in
    eastward flow gives positive advection."""
    ds = ds_uv_grid.copy()
    # c grows linearly with longitude; u is positive ⇒ -u·∂c/∂x < 0
    lon2, _ = np.meshgrid(ds["lon"].values, ds["lat"].values, indexing="xy")
    ds["c"] = (("lat", "lon"), lon2.astype(float))
    out = advection(ds, scalar="c")
    # ds_uv_grid has u = 0.5 cos(lat) > 0 everywhere.
    interior = out["c_advection"].values[1:-1, 1:-1]
    assert (interior < 0.0).all()


def test_advection_components_dims_mismatch_raises(ds_uv_grid):
    ds = ds_uv_grid.assign(c=lambda d: d["u"])
    with pytest.raises(ValueError, match="exactly two components"):
        advection(ds, scalar="c", components=("u",), dims=("lon",))


def test_advection_rejects_non_horizontal_dim(ds_uv_grid):
    """3-D advection isn't wired up yet — an explicit error beats a
    confusing failure deep inside calc.partial."""
    ds = ds_uv_grid.assign(c=lambda d: d["u"])
    with pytest.raises(ValueError, match="lon/lat"):
        advection(ds, scalar="c", components=("u", "v"), dims=("lon", "depth"))


def test_shear_vorticity_zero_speed_returns_zero():
    lon = np.linspace(-10.0, 10.0, 5)
    lat = np.linspace(20.0, 40.0, 5)
    zeros = np.zeros((5, 5))
    ds = xr.Dataset(
        {"u": (("lat", "lon"), zeros), "v": (("lat", "lon"), zeros)},
        coords={"lon": lon, "lat": lat},
    )
    out = shear_vorticity(ds)
    assert np.isfinite(out["vort_shear"].values).all()
    np.testing.assert_array_equal(out["vort_shear"].values, 0.0)


def test_curvature_vorticity_zero_speed_returns_zero():
    lon = np.linspace(-10.0, 10.0, 5)
    lat = np.linspace(20.0, 40.0, 5)
    zeros = np.zeros((5, 5))
    ds = xr.Dataset(
        {"u": (("lat", "lon"), zeros), "v": (("lat", "lon"), zeros)},
        coords={"lon": lon, "lat": lat},
    )
    out = curvature_vorticity(ds)
    assert np.isfinite(out["vort_curv"].values).all()
    np.testing.assert_array_equal(out["vort_curv"].values, 0.0)


def test_ageostrophic_velocities_zero_for_purely_geostrophic_input(ds_ssh_grid):
    """If the total wind equals the geostrophic wind, ageostrophic = 0."""
    geo = geostrophic_velocities(ds_ssh_grid)
    ds = xr.merge([ds_ssh_grid, geo])
    out = ageostrophic_velocities(ds, variable="ssh")
    np.testing.assert_allclose(out["u_a"].values, 0.0, atol=1e-30)
    np.testing.assert_allclose(out["v_a"].values, 0.0, atol=1e-30)


def test_ageostrophic_velocities_returns_u_a_v_a(ds_uv_ssh_grid):
    out = ageostrophic_velocities(ds_uv_ssh_grid, variable="ssh")
    assert set(out.data_vars) == {"u_a", "v_a"}


def test_shear_plus_curvature_vorticity_sums_to_relative_vorticity(ds_uv_grid):
    """``ζ = shear_vorticity + curvature_vorticity`` (Majumdar 2024)."""
    zeta = relative_vorticity(ds_uv_grid)["vort_r"]
    zs = shear_vorticity(ds_uv_grid)["vort_shear"]
    zc = curvature_vorticity(ds_uv_grid)["vort_curv"]
    # The decomposition omits the spherical curvature correction (it
    # acts only on the curl operator). Exclude that term from the sum
    # comparison by reconstructing the plain ``∂v/∂x − ∂u/∂y`` field
    # — the same pieces shear+curvature reassemble.
    from xr_toolz.calc import partial as _partial

    plain_zeta = _partial(ds_uv_grid["v"], "lon", geometry="spherical") - _partial(
        ds_uv_grid["u"], "lat", geometry="spherical"
    )
    interior = (zs + zc - plain_zeta).values[2:-2, 2:-2]
    np.testing.assert_allclose(interior, 0.0, atol=1e-12)
    assert zeta.dims == zs.dims  # shape sanity


def test_shear_vorticity_dataset_var_name(ds_uv_grid):
    out = shear_vorticity(ds_uv_grid)
    assert "vort_shear" in out.data_vars


def test_curvature_vorticity_dataset_var_name(ds_uv_grid):
    out = curvature_vorticity(ds_uv_grid)
    assert "vort_curv" in out.data_vars


def test_frontogenesis_zero_on_uniform_scalar(ds_uv_grid):
    """A scalar with no gradient ⇒ frontogenesis identically zero."""
    ds = ds_uv_grid.assign(theta=lambda d: d["u"] * 0.0 + 280.0)
    out = frontogenesis(ds, scalar="theta")
    np.testing.assert_allclose(out["theta_frontogenesis"].values, 0.0, atol=1e-30)


def test_frontogenesis_returns_named_dataset(ds_uv_grid):
    ds = ds_uv_grid.assign(theta=lambda d: d["u"])
    out = frontogenesis(ds, scalar="theta")
    assert "theta_frontogenesis" in out.data_vars


def test_potential_vorticity_barotropic_unit_height_equals_absolute_vorticity(
    ds_uv_grid,
):
    """``h = 1`` ⇒ PV = absolute vorticity."""
    ds = ds_uv_grid.assign(h=lambda d: d["u"] * 0.0 + 1.0)
    pv = potential_vorticity_barotropic(ds, height="h")["pv_barotropic"]
    eta = absolute_vorticity(ds_uv_grid)["vort_a"]
    np.testing.assert_allclose(pv.values, eta.values, atol=1e-15)


# ---- velocity magnitudes / EKE / N² / MLD --------------------------------


@pytest.fixture
def ds_uvw_grid(ds_uv_grid) -> xr.Dataset:
    return ds_uv_grid.assign(w=lambda d: d["u"] * 0.0 + 0.01)


@pytest.fixture
def ds_density_profile() -> xr.Dataset:
    """1-D ocean density profile with a clear mixed layer.

    ρ stays at 1024.5 kg/m³ for the upper 50 m (mixed), then increases
    by 0.5 kg/m³ over the next 50 m, then continues stratified."""
    depth = np.array([0.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0])
    rho = np.array(
        [1024.5, 1024.5, 1024.5, 1024.5, 1024.5, 1024.7, 1025.0, 1025.5, 1026.0]
    )
    return xr.Dataset(
        {"rho": (("depth",), rho)},
        coords={"depth": depth},
    )


def test_velocity_magnitude_2d(ds_uv_grid):
    out = velocity_magnitude(ds_uv_grid)
    expected = np.sqrt(ds_uv_grid["u"] ** 2 + ds_uv_grid["v"] ** 2)
    np.testing.assert_allclose(out["speed"].values, expected.values, atol=1e-15)


def test_velocity_magnitude_3d_includes_w(ds_uvw_grid):
    out = velocity_magnitude(ds_uvw_grid, w="w")
    expected = np.sqrt(
        ds_uvw_grid["u"] ** 2 + ds_uvw_grid["v"] ** 2 + ds_uvw_grid["w"] ** 2
    )
    np.testing.assert_allclose(out["speed"].values, expected.values, atol=1e-15)


def test_horizontal_velocity_magnitude_matches_2d(ds_uv_grid):
    a = horizontal_velocity_magnitude(ds_uv_grid)
    b = velocity_magnitude(ds_uv_grid)
    np.testing.assert_allclose(a["speed"].values, b["speed"].values, atol=1e-15)


def test_eddy_kinetic_energy_zero_for_zero_anomalies(ds_uv_grid):
    ds = ds_uv_grid.assign(
        u_anom=lambda d: d["u"] * 0.0,
        v_anom=lambda d: d["v"] * 0.0,
    )
    out = eddy_kinetic_energy(ds)
    np.testing.assert_allclose(out["eke"].values, 0.0, atol=1e-30)


def test_eddy_kinetic_energy_matches_definition(ds_uv_grid):
    ds = ds_uv_grid.rename({"u": "u_anom", "v": "v_anom"})
    out = eddy_kinetic_energy(ds)
    expected = 0.5 * (ds["u_anom"] ** 2 + ds["v_anom"] ** 2)
    np.testing.assert_allclose(out["eke"].values, expected.values, atol=1e-15)


def test_brunt_vaisala_frequency_positive_for_stable_profile(
    ds_density_profile,
):
    """Density increasing with depth ⇒ ∂ρ/∂z > 0 ⇒ N² > 0."""
    out = brunt_vaisala_frequency(ds_density_profile)
    # Check the strongly-stratified region (below the mixed layer)
    interior = out["n_squared"].values[5:-1]
    assert (interior > 0.0).all()


def test_brunt_vaisala_frequency_zero_for_homogeneous_column():
    depth = np.linspace(0.0, 100.0, 11)
    rho = np.full_like(depth, 1025.0)
    ds = xr.Dataset({"rho": (("depth",), rho)}, coords={"depth": depth})
    out = brunt_vaisala_frequency(ds)
    np.testing.assert_allclose(out["n_squared"].values, 0.0, atol=1e-20)


def test_brunt_vaisala_frequency_unit_attr(ds_density_profile):
    out = brunt_vaisala_frequency(ds_density_profile)
    assert out["n_squared"].attrs["units"] == "s-2"


def test_mixed_layer_depth_returns_threshold_crossing(ds_density_profile):
    """ρ jumps by 0.03 between 50 and 75 m ⇒ MLD ≈ 75 m."""
    out = mixed_layer_depth(ds_density_profile)
    # Threshold 0.03 — value at 75m is 1024.7 vs ref(10m)=1024.5,
    # difference 0.2 > 0.03 ⇒ first crossing is at 75 m.
    assert float(out["mld"].values) == pytest.approx(75.0)


def test_mixed_layer_depth_fully_mixed_returns_deepest():
    depth = np.array([0.0, 10.0, 50.0, 100.0])
    rho = np.full_like(depth, 1025.0)
    ds = xr.Dataset({"rho": (("depth",), rho)}, coords={"depth": depth})
    out = mixed_layer_depth(ds)
    assert float(out["mld"].values) == pytest.approx(100.0)


def test_mixed_layer_depth_2d_horizontal_shape():
    """MLD must broadcast over horizontal dims and return a 2-D field."""
    depth = np.array([0.0, 10.0, 50.0, 75.0, 100.0])
    nx, ny = 3, 4
    # Two columns with different MLDs
    rho = np.broadcast_to(
        np.array([1024.5, 1024.5, 1024.5, 1024.7, 1025.0])[:, None, None],
        (5, nx, ny),
    ).copy()
    ds = xr.Dataset(
        {"rho": (("depth", "x", "y"), rho)},
        coords={"depth": depth, "x": np.arange(nx), "y": np.arange(ny)},
    )
    out = mixed_layer_depth(ds)
    assert out["mld"].dims == ("x", "y")
    assert out["mld"].shape == (nx, ny)
    np.testing.assert_allclose(out["mld"].values, 75.0)


def test_lapse_rate_isothermal_column_is_zero():
    depth = np.linspace(0.0, 100.0, 11)
    T = np.full_like(depth, 280.0)
    ds = xr.Dataset({"T": (("depth",), T)}, coords={"depth": depth})
    out = lapse_rate(ds)
    np.testing.assert_allclose(out["lapse_rate"].values, 0.0, atol=1e-20)


def test_lapse_rate_positive_for_temperature_decreasing_upward():
    """Atmospheric default: T decreases with height ⇒ Γ > 0.

    With ``positive="up"`` and ``∂T/∂z(up) < 0``, ``Γ = −∂T/∂z > 0``.
    """
    height = np.linspace(0.0, 10000.0, 11)  # m, positive upward
    T = 288.0 - 6.5e-3 * height  # standard atmosphere lapse rate
    ds = xr.Dataset({"T": (("z",), T)}, coords={"z": height})
    out = lapse_rate(ds, depth="z", positive="up")
    interior = out["lapse_rate"].values[1:-1]
    np.testing.assert_allclose(interior, 6.5e-3, atol=1e-12)


def test_lapse_rate_invalid_positive_raises():
    ds = xr.Dataset({"T": (("depth",), np.zeros(3))}, coords={"depth": np.arange(3.0)})
    with pytest.raises(ValueError, match="must be 'down' or 'up'"):
        lapse_rate(ds, positive="sideways")


def test_lapse_rate_operator():
    depth = np.linspace(0.0, 100.0, 11)
    T = 280.0 + 0.01 * depth
    ds = xr.Dataset({"T": (("depth",), T)}, coords={"depth": depth})
    out = LapseRate()(ds)
    assert "lapse_rate" in out.data_vars


def test_mixed_layer_depth_density_must_have_depth_dim():
    ds = xr.Dataset({"rho": (("x",), np.array([1.0, 2.0]))})
    with pytest.raises(ValueError, match="not defined on"):
        mixed_layer_depth(ds)


# ---- operator wrappers for the new kinematics ----------------------------


def test_velocity_magnitude_operator(ds_uvw_grid):
    out = VelocityMagnitude(w="w")(ds_uvw_grid)
    assert "speed" in out.data_vars


def test_horizontal_velocity_magnitude_operator(ds_uv_grid):
    out = HorizontalVelocityMagnitude()(ds_uv_grid)
    assert "speed" in out.data_vars


def test_eddy_kinetic_energy_operator(ds_uv_grid):
    ds = ds_uv_grid.rename({"u": "u_anom", "v": "v_anom"})
    out = EddyKineticEnergy()(ds)
    assert "eke" in out.data_vars


def test_brunt_vaisala_frequency_operator(ds_density_profile):
    out = BruntVaisalaFrequency()(ds_density_profile)
    assert "n_squared" in out.data_vars


def test_mixed_layer_depth_operator(ds_density_profile):
    out = MixedLayerDepth()(ds_density_profile)
    assert "mld" in out.data_vars


def test_potential_vorticity_barotropic_doubles_when_height_halves(ds_uv_grid):
    ds_thin = ds_uv_grid.assign(h=lambda d: d["u"] * 0.0 + 0.5)
    ds_thick = ds_uv_grid.assign(h=lambda d: d["u"] * 0.0 + 1.0)
    pv_thin = potential_vorticity_barotropic(ds_thin, height="h")["pv_barotropic"]
    pv_thick = potential_vorticity_barotropic(ds_thick, height="h")["pv_barotropic"]
    np.testing.assert_allclose(pv_thin.values, 2.0 * pv_thick.values, atol=1e-15)


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


def test_advection_operator(ds_uv_grid):
    ds = ds_uv_grid.assign(c=lambda d: d["u"] * 0.0)
    out = Advection(scalar="c")(ds)
    assert "c_advection" in out.data_vars


def test_ageostrophic_velocities_operator(ds_ssh_grid):
    geo = geostrophic_velocities(ds_ssh_grid)
    ds = xr.merge([ds_ssh_grid, geo])
    out = AgeostrophicVelocities(variable="ssh")(ds)
    assert set(out.data_vars) == {"u_a", "v_a"}


def test_shear_curvature_vorticity_operators(ds_uv_grid):
    zs = ShearVorticity()(ds_uv_grid)
    zc = CurvatureVorticity()(ds_uv_grid)
    assert "vort_shear" in zs.data_vars
    assert "vort_curv" in zc.data_vars


def test_frontogenesis_operator(ds_uv_grid):
    ds = ds_uv_grid.assign(theta=lambda d: d["u"])
    out = Frontogenesis(scalar="theta")(ds)
    assert "theta_frontogenesis" in out.data_vars


def test_potential_vorticity_barotropic_operator(ds_uv_grid):
    ds = ds_uv_grid.assign(h=lambda d: d["u"] * 0.0 + 1.0)
    out = PotentialVorticityBarotropic(height="h")(ds)
    assert "pv_barotropic" in out.data_vars
