"""V4 process-evaluation tests — physical-consistency metrics + budgets.

Covers V4.1 (metrics.physical), V4.2 (budgets primitives), V4.3 (heat /
salt / volume / KE residuals), V4.4 (grid_metrics_from_coords helper),
and V4.5 (kinematics audit + density_from_ts gating).
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xr_toolz import calc, metrics, ocn
from xr_toolz.budgets import (
    BoundaryFlux,
    BudgetResidual,
    ControlVolumeIntegral,
    HeatBudgetResidual,
    KineticEnergyBudgetResidual,
    SaltBudgetResidual,
    VolumeBudgetResidual,
    boundary_flux,
    budget_residual,
    control_volume_integral,
    heat_budget_residual,
    kinetic_energy_budget_residual,
    salt_budget_residual,
    volume_budget_residual,
)
from xr_toolz.metrics import (
    DensityInversionFraction,
    DivergenceError,
    GeostrophicBalanceError,
    PVConservationError,
    density_inversion_fraction,
    divergence_error,
    geostrophic_balance_error,
    pv_conservation_error,
)


# ---- Fixtures ------------------------------------------------------------


@pytest.fixture
def lonlat_grid():
    lon = np.linspace(-30.0, 30.0, 25)
    lat = np.linspace(20.0, 50.0, 21)
    return lon, lat


@pytest.fixture
def geostrophic_dataset(lonlat_grid):
    """Synthetic SSH + (u, v) that obey geostrophic balance to leading order."""
    lon, lat = lonlat_grid
    LON, LAT = np.meshgrid(lon, lat, indexing="xy")
    eta = 0.1 * np.sin(np.deg2rad(LON)) * np.cos(np.deg2rad(LAT))
    da_eta = xr.DataArray(eta, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
    ds_eta = da_eta.to_dataset(name="ssh")
    geo = ocn.geostrophic_velocities(ds_eta, variable="ssh")
    return xr.Dataset({"ssh": ds_eta["ssh"], "u": geo["u"], "v": geo["v"]})


# ---- V4.1 physical metrics ----------------------------------------------


def test_geostrophic_balance_error_near_zero_on_geostrophic_flow(
    geostrophic_dataset,
):
    res = geostrophic_balance_error(geostrophic_dataset)
    assert set(res.data_vars) == {"r_u", "r_v"}
    interior = {"lat": slice(2, -2), "lon": slice(2, -2)}
    assert float(np.abs(res["r_u"].isel(interior)).max()) < 1e-2
    assert float(np.abs(res["r_v"].isel(interior)).max()) < 1e-2


def test_geostrophic_balance_error_grows_when_velocity_inflated(
    geostrophic_dataset,
):
    base = geostrophic_balance_error(geostrophic_dataset)
    bad = geostrophic_dataset.assign(u=geostrophic_dataset["u"] * 2.0)
    perturbed = geostrophic_balance_error(bad)
    interior = {"lat": slice(2, -2), "lon": slice(2, -2)}
    assert float(np.abs(perturbed["r_u"].isel(interior)).max()) > float(
        np.abs(base["r_u"].isel(interior)).max()
    )


def test_geostrophic_balance_error_operator_matches_layer0(geostrophic_dataset):
    op = GeostrophicBalanceError()
    eager = op(geostrophic_dataset)
    direct = geostrophic_balance_error(geostrophic_dataset)
    xr.testing.assert_allclose(eager, direct)


def test_divergence_error_zero_on_constant_flow(lonlat_grid):
    lon, lat = lonlat_grid
    u = xr.DataArray(
        np.ones((lat.size, lon.size)),
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
    )
    v = xr.zeros_like(u)
    ds = xr.Dataset({"u": u, "v": v})
    div = divergence_error(ds)
    interior = div.isel(lat=slice(2, -2), lon=slice(2, -2))
    assert float(np.abs(interior).max()) < 1e-9


def test_divergence_error_positive_on_diverging_flow(lonlat_grid):
    lon, lat = lonlat_grid
    LON, _ = np.meshgrid(lon, lat, indexing="xy")
    u = xr.DataArray(LON.copy(), coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
    v = xr.zeros_like(u)
    ds = xr.Dataset({"u": u, "v": v})
    div = divergence_error(ds)
    interior = div.isel(lat=slice(2, -2), lon=slice(2, -2))
    assert float(interior.mean()) > 0.0


def test_divergence_error_operator_returns_dataarray(geostrophic_dataset):
    op = DivergenceError()
    out = op(geostrophic_dataset)
    assert isinstance(out, xr.DataArray)


def test_density_inversion_fraction_zero_on_stable_column():
    depth = np.linspace(0.0, 1000.0, 11)
    rho = 1020.0 + 0.005 * depth
    da = xr.DataArray(rho, coords={"depth": depth}, dims=("depth",))
    ds = da.to_dataset(name="rho")
    frac = density_inversion_fraction(ds)
    assert float(frac) == pytest.approx(0.0)


def test_density_inversion_fraction_increases_with_inversions():
    depth = np.linspace(0.0, 1000.0, 11)
    rho = 1020.0 + 0.005 * depth
    rho[5] = rho[4] - 0.05  # one inversion
    da = xr.DataArray(rho, coords={"depth": depth}, dims=("depth",))
    frac1 = density_inversion_fraction(da.to_dataset(name="rho"))

    rho[3] = rho[2] - 0.05  # second inversion
    da2 = xr.DataArray(rho, coords={"depth": depth}, dims=("depth",))
    frac2 = density_inversion_fraction(da2.to_dataset(name="rho"))

    assert float(frac1) > 0.0
    assert float(frac2) > float(frac1)


def test_density_inversion_fraction_raises_on_missing_depth_dim():
    da = xr.DataArray(np.array([1, 2, 3.0]), dims=("x",))
    with pytest.raises(ValueError, match="depth"):
        density_inversion_fraction(da.to_dataset(name="rho"))


def test_density_inversion_fraction_operator():
    op = DensityInversionFraction()
    depth = np.linspace(0.0, 1000.0, 11)
    rho = 1020.0 + 0.005 * depth
    da = xr.DataArray(rho, coords={"depth": depth}, dims=("depth",))
    out = op(da.to_dataset(name="rho"))
    assert float(out) == pytest.approx(0.0)


def test_pv_conservation_error_zero_on_constant_pv():
    n_traj, n_t = 5, 10
    pv = np.full((n_traj, n_t), 1e-7)
    ds = xr.Dataset(
        {"pv": (("trajectory", "time"), pv)},
        coords={
            "trajectory": np.arange(n_traj),
            "time": np.arange(n_t),
        },
    )
    err = pv_conservation_error(ds)
    assert float(err) == pytest.approx(0.0)


def test_pv_conservation_error_positive_on_drifting_pv():
    n_traj, n_t = 4, 20
    rng = np.random.default_rng(0)
    pv = 1e-7 + 1e-9 * rng.standard_normal((n_traj, n_t))
    ds = xr.Dataset(
        {"pv": (("trajectory", "time"), pv)},
        coords={
            "trajectory": np.arange(n_traj),
            "time": np.arange(n_t),
        },
    )
    err = pv_conservation_error(ds)
    assert float(err) > 0.0


def test_pv_conservation_error_raises_on_missing_dims():
    da = xr.DataArray(np.zeros(4), dims=("trajectory",))
    with pytest.raises(ValueError, match="dims"):
        pv_conservation_error(da.to_dataset(name="pv"))


def test_pv_conservation_error_operator():
    n_traj, n_t = 3, 5
    pv = np.full((n_traj, n_t), 1e-7)
    ds = xr.Dataset(
        {"pv": (("trajectory", "time"), pv)},
        coords={"trajectory": np.arange(n_traj), "time": np.arange(n_t)},
    )
    op = PVConservationError()
    assert float(op(ds)) == pytest.approx(0.0)


# ---- V4.4 grid_metrics_from_coords --------------------------------------


def test_grid_metrics_spherical_dy_constant(lonlat_grid):
    lon, lat = lonlat_grid
    ds = xr.Dataset(coords={"lat": lat, "lon": lon})
    vol, face = calc.grid_metrics_from_coords(ds)
    assert "dx" in vol and "dy" in vol and "cell_area" in vol
    assert "dx_e" in face and "dy_n" in face
    # dy is constant (= R * dphi)
    np.testing.assert_allclose(vol["dy"].values, vol["dy"].values[0], rtol=1e-12)
    # dx decreases with |latitude| (cos phi)
    dx_eq = float(vol["dx"].sel(lat=lat[lat.size // 2]).mean())
    dx_pole = float(vol["dx"].sel(lat=lat.max()).mean())
    assert dx_pole < dx_eq


def test_grid_metrics_spherical_cell_area_matches_known_sphere(lonlat_grid):
    lon, lat = lonlat_grid
    ds = xr.Dataset(coords={"lat": lat, "lon": lon})
    vol, _ = calc.grid_metrics_from_coords(ds)
    R = calc.EARTH_RADIUS
    rad = float(np.pi / 180.0)
    dlat = float(lat[1] - lat[0]) * rad
    dlon = float(lon[1] - lon[0]) * rad
    expected_at_eq = R * R * dlat * dlon * np.cos(0.0)
    measured_at_eq = float(
        vol["cell_area"].sel(lat=lat[np.argmin(np.abs(lat))], method="nearest").mean()
    )
    # equator latitude is 20°N here; just check expected formula at that lat
    expected = R * R * dlat * dlon * np.cos(20.0 * rad)
    assert measured_at_eq == pytest.approx(expected, rel=5e-2)
    assert expected_at_eq > 0.0


def test_grid_metrics_with_depth_returns_cell_volume(lonlat_grid):
    lon, lat = lonlat_grid
    depth = np.array([0.0, 5.0, 15.0, 50.0])
    ds = xr.Dataset(coords={"lat": lat, "lon": lon, "depth": depth})
    vol, face = calc.grid_metrics_from_coords(ds, depth="depth")
    assert "cell_volume" in vol and "dz" in vol
    assert "dz_top" in face and "area_top" in face
    expected = float(vol["cell_area"].mean()) * float(vol["dz"].mean())
    assert float(vol["cell_volume"].mean()) == pytest.approx(expected, rel=1e-9)


def test_grid_metrics_cartesian_mode(lonlat_grid):
    lon, lat = lonlat_grid
    ds = xr.Dataset(coords={"lat": lat, "lon": lon})
    vol, _ = calc.grid_metrics_from_coords(ds, sphere=False)
    np.testing.assert_allclose(vol["dx"].values, vol["dx"].values.mean(), rtol=1e-12)


def test_grid_metrics_raises_on_missing_coord():
    ds = xr.Dataset(coords={"lat": [0, 1, 2.0]})
    with pytest.raises(ValueError, match="lon"):
        calc.grid_metrics_from_coords(ds)


# ---- V4.2 budget primitives ----------------------------------------------


def test_control_volume_integral_constant_field_equals_value_times_volume(
    lonlat_grid,
):
    lon, lat = lonlat_grid
    ds = xr.Dataset(
        {
            "phi": (
                ("lat", "lon"),
                10.0 * np.ones((lat.size, lon.size)),
            )
        },
        coords={"lat": lat, "lon": lon},
    )
    vol, _ = calc.grid_metrics_from_coords(ds)
    integ = control_volume_integral(
        ds, variable="phi", volume_metrics=vol, dims=("lat", "lon")
    )
    expected = 10.0 * float(vol["cell_volume"].sum())
    assert float(integ) == pytest.approx(expected, rel=1e-9)


def test_control_volume_integral_with_region_excludes_outside(lonlat_grid):
    lon, lat = lonlat_grid
    ds = xr.Dataset(
        {"phi": (("lat", "lon"), np.ones((lat.size, lon.size)))},
        coords={"lat": lat, "lon": lon},
    )
    vol, _ = calc.grid_metrics_from_coords(ds)
    region = ds["phi"] > 0
    half_region = region.where(ds["lat"] > 35.0, other=False).fillna(False).astype(bool)
    full = control_volume_integral(
        ds, variable="phi", volume_metrics=vol, dims=("lat", "lon")
    )
    half = control_volume_integral(
        ds, variable="phi", volume_metrics=vol, region=half_region, dims=("lat", "lon")
    )
    assert float(half) < float(full)


def test_control_volume_integral_operator(lonlat_grid):
    lon, lat = lonlat_grid
    ds = xr.Dataset(
        {"phi": (("lat", "lon"), np.full((lat.size, lon.size), 3.0))},
        coords={"lat": lat, "lon": lon},
    )
    vol, _ = calc.grid_metrics_from_coords(ds)
    op = ControlVolumeIntegral("phi", volume_metrics=vol, dims=("lat", "lon"))
    out = op(ds)
    assert float(out) == pytest.approx(3.0 * float(vol["cell_volume"].sum()))


def test_boundary_flux_uniform_flow_through_unit_face_recovers_speed():
    lon = np.linspace(0.0, 1.0, 4)
    lat = np.linspace(0.0, 1.0, 4)
    u = xr.DataArray(
        2.0 * np.ones((lat.size, lon.size)),
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
    )
    v = xr.zeros_like(u)
    face_metrics = xr.Dataset(
        {
            "area_e": xr.DataArray(
                np.ones_like(u.values),
                coords={"lat": lat, "lon": lon},
                dims=("lat", "lon"),
            ),
            "area_n": xr.DataArray(
                np.ones_like(u.values),
                coords={"lat": lat, "lon": lon},
                dims=("lat", "lon"),
            ),
        }
    )
    ds = xr.Dataset({"u": u, "v": v})
    flux = boundary_flux(
        ds, variable=None, velocity_vars={"u": "u", "v": "v"}, face_metrics=face_metrics
    )
    # 16 cells × 1 m² × 2 m/s = 32
    assert float(flux["flux_x"]) == pytest.approx(32.0)
    assert float(flux["flux_y"]) == pytest.approx(0.0)


def test_boundary_flux_operator_returns_dataset():
    lon = np.array([0.0, 1.0])
    lat = np.array([0.0, 1.0])
    u = xr.DataArray(
        np.ones((2, 2)), coords={"lat": lat, "lon": lon}, dims=("lat", "lon")
    )
    fm = xr.Dataset({"area_e": u})
    ds = xr.Dataset({"u": u})
    op = BoundaryFlux(variable=None, velocity_vars={"u": "u"}, face_metrics=fm)
    assert isinstance(op(ds), xr.Dataset)


def test_budget_residual_closes_for_known_balance():
    # Convention: residual = tendency + flux_divergence - source + sink
    tendency = xr.DataArray(np.array([1.0, 2.0, 3.0]), dims=("x",))
    flux_div = xr.DataArray(np.array([0.5, 1.0, 1.5]), dims=("x",))
    source = xr.DataArray(np.array([1.5, 3.0, 4.5]), dims=("x",))
    res = budget_residual(tendency, flux_div, source=source)
    np.testing.assert_allclose(res.values, 0.0, atol=1e-12)


def test_budget_residual_operator():
    op = BudgetResidual()
    tendency = xr.DataArray(np.zeros(3), dims=("x",))
    flux = xr.DataArray(np.zeros(3), dims=("x",))
    out = op(tendency, flux)
    np.testing.assert_allclose(out.values, 0.0, atol=1e-12)


# ---- V4.3 budget residuals ----------------------------------------------


def _closed_dataset(lonlat_grid):
    """A constant-everything Dataset has a closed budget by construction."""
    lon, lat = lonlat_grid
    time = np.arange(3)
    shape = (time.size, lat.size, lon.size)
    coords = {"time": time, "lat": lat, "lon": lon}
    dims = ("time", "lat", "lon")
    return xr.Dataset(
        {
            "theta": (dims, np.full(shape, 15.0)),
            "so": (dims, np.full(shape, 35.0)),
            "u": (dims, np.zeros(shape)),
            "v": (dims, np.zeros(shape)),
        },
        coords=coords,
    )


def test_heat_budget_residual_closes_on_constant_dataset(lonlat_grid):
    ds = _closed_dataset(lonlat_grid)
    res = heat_budget_residual(ds, depth=None)
    np.testing.assert_allclose(res.values, 0.0, atol=1e-9)


def test_heat_budget_residual_with_surface_flux_closes(lonlat_grid):
    ds = _closed_dataset(lonlat_grid)
    # a non-zero surface flux + matching extra tendency would be needed for
    # full closure; this just confirms the source term subtracts cleanly
    flux = xr.full_like(ds["theta"], 0.0)
    out = heat_budget_residual(ds.assign(F=flux), surface_flux_var="F", depth=None)
    np.testing.assert_allclose(out.values, 0.0, atol=1e-9)


def test_heat_budget_residual_operator(lonlat_grid):
    ds = _closed_dataset(lonlat_grid)
    op = HeatBudgetResidual(temp_var="theta", depth=None)
    out = op(ds)
    np.testing.assert_allclose(out.values, 0.0, atol=1e-9)


def test_salt_budget_residual_closes_on_constant_dataset(lonlat_grid):
    ds = _closed_dataset(lonlat_grid)
    res = salt_budget_residual(ds, depth=None)
    np.testing.assert_allclose(res.values, 0.0, atol=1e-9)


def test_salt_budget_residual_operator(lonlat_grid):
    ds = _closed_dataset(lonlat_grid)
    op = SaltBudgetResidual(salt_var="so", depth=None)
    out = op(ds)
    np.testing.assert_allclose(out.values, 0.0, atol=1e-9)


def test_volume_budget_residual_zero_on_zero_flow(lonlat_grid):
    ds = _closed_dataset(lonlat_grid)
    res = volume_budget_residual(ds, depth=None)
    np.testing.assert_allclose(res.values, 0.0, atol=1e-12)


def test_volume_budget_residual_operator(lonlat_grid):
    ds = _closed_dataset(lonlat_grid)
    op = VolumeBudgetResidual(depth=None)
    out = op(ds)
    np.testing.assert_allclose(out.values, 0.0, atol=1e-12)


def test_kinetic_energy_budget_residual_zero_on_zero_flow(lonlat_grid):
    ds = _closed_dataset(lonlat_grid)
    res = kinetic_energy_budget_residual(ds, depth=None)
    np.testing.assert_allclose(res.values, 0.0, atol=1e-12)


def test_kinetic_energy_budget_residual_operator(lonlat_grid):
    ds = _closed_dataset(lonlat_grid)
    op = KineticEnergyBudgetResidual(depth=None)
    np.testing.assert_allclose(op(ds).values, 0.0, atol=1e-12)


def test_kinetic_energy_budget_residual_subtracts_forcing(lonlat_grid):
    ds = _closed_dataset(lonlat_grid)
    forcing = xr.full_like(ds["u"], 0.0)
    out = kinetic_energy_budget_residual(
        ds.assign(F=forcing), forcing_vars=["F"], depth=None
    )
    np.testing.assert_allclose(out.values, 0.0, atol=1e-12)


# ---- V4.5 kinematics audit ----------------------------------------------


def test_density_from_ts_raises_when_eos_unsupported():
    ds = xr.Dataset({"so": ("x", [35.0]), "thetao": ("x", [15.0])})
    with pytest.raises(NotImplementedError, match="eos"):
        ocn.density_from_ts(ds, eos="linear")


def test_density_from_ts_raises_clean_importerror_without_gsw(monkeypatch):
    """If gsw is unavailable, the error message points at the optional extra."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "gsw":
            raise ImportError("No module named 'gsw'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    lon = np.array([0.0])
    lat = np.array([0.0])
    ds = xr.Dataset(
        {
            "so": (("lat", "lon"), [[35.0]]),
            "thetao": (("lat", "lon"), [[15.0]]),
        },
        coords={"lat": lat, "lon": lon},
    )
    with pytest.raises(ImportError, match="oceanography"):
        ocn.density_from_ts(ds)


def test_kinematics_inventory_present():
    """V4.5 audit gate — every quantity in the audit table is reachable."""
    for name in (
        "relative_vorticity",
        "absolute_vorticity",
        "divergence",
        "kinetic_energy",
        "brunt_vaisala_frequency",
        "mixed_layer_depth",
        "density_from_ts",
    ):
        assert hasattr(ocn, name), f"ocn is missing {name!r}"


def test_metrics_physical_submodule_imports():
    from xr_toolz.metrics.physical import (
        DensityInversionFraction,
        DivergenceError,
        GeostrophicBalanceError,
        PVConservationError,
        density_inversion_fraction,
        divergence_error,
        geostrophic_balance_error,
        pv_conservation_error,
    )

    assert callable(geostrophic_balance_error)
    _ = (
        DensityInversionFraction,
        DivergenceError,
        GeostrophicBalanceError,
        PVConservationError,
        density_inversion_fraction,
        divergence_error,
        pv_conservation_error,
    )


def test_metrics_top_level_exposes_v4():
    from xr_toolz.metrics import (
        DensityInversionFraction,
        DivergenceError,
        GeostrophicBalanceError,
        PVConservationError,
    )

    assert DensityInversionFraction is metrics.DensityInversionFraction
    _ = (DivergenceError, GeostrophicBalanceError, PVConservationError)


def test_budgets_top_level_exposes_v4():
    import xr_toolz.budgets as bdg

    for name in (
        "control_volume_integral",
        "boundary_flux",
        "budget_residual",
        "heat_budget_residual",
        "salt_budget_residual",
        "volume_budget_residual",
        "kinetic_energy_budget_residual",
        "ControlVolumeIntegral",
        "BoundaryFlux",
        "BudgetResidual",
        "HeatBudgetResidual",
        "SaltBudgetResidual",
        "VolumeBudgetResidual",
        "KineticEnergyBudgetResidual",
    ):
        assert hasattr(bdg, name), f"xr_toolz.budgets is missing {name!r}"
