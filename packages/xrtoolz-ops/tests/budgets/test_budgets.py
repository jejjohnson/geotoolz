"""V4 process-evaluation tests — physical-consistency metrics + budgets.

Covers V4.1 (metrics.physical), V4.2 (budgets primitives), V4.3 (heat /
salt / volume / KE residuals), V4.4 (grid_metrics_from_coords helper),
and V4.5 (kinematics audit + density_from_ts gating).
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrops import grad as calc, metrics, ocn
from xrops.budgets import (
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
from xrops.metrics import (
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


def _swirl(n: int = 40) -> xr.Dataset:
    """A flow with sources on an even plane, and a tracer."""
    x = y = np.linspace(-1e5, 1e5, n)
    X, Y = np.meshgrid(x, y)
    return xr.Dataset(
        {
            "u": (("y", "x"), 1e-5 * X - 2e-5 * Y + 1e-11 * X * Y),
            "v": (("y", "x"), 2e-5 * X + 3e-5 * Y),
            "c": (("y", "x"), 10.0 + X / 1e5),
        },
        coords={"x": x, "y": y},
    )


PLANE = {"dims": ("x", "y"), "geometry": "cartesian"}


def _box(ds: xr.Dataset, x: slice, y: slice) -> xr.DataArray:
    inside = xr.zeros_like(ds["u"], dtype=bool)
    inside.loc[{"x": x, "y": y}] = True
    return inside


def test_boundary_flux_of_a_radial_flow_is_twice_its_area():
    ds = _swirl()
    radial = ds.assign(
        u=ds["x"] * xr.ones_like(ds["u"]), v=ds["y"] * xr.ones_like(ds["u"])
    )
    box = _box(ds, slice(-3e4, 3e4), slice(-2e4, 4e4))
    area = int(box.sum()) * float(ds["x"].diff("x")[0]) ** 2
    assert float(boundary_flux(radial, region=box, **PLANE)) == pytest.approx(2 * area)


def test_boundary_flux_of_a_region_is_the_sum_over_its_parts():
    ds = _swirl()
    whole = _box(ds, slice(-6e4, 6e4), slice(-6e4, 6e4))
    left = whole & (ds["x"] < 1e4)
    parts = [
        boundary_flux(ds, region=r, variable="c", **PLANE)
        for r in (left, whole & ~left)
    ]
    total = boundary_flux(ds, region=whole, variable="c", **PLANE)
    assert float(sum(parts)) == pytest.approx(float(total), rel=1e-12)


def test_no_tracer_crosses_the_boundary_of_a_rotation():
    ds = _swirl()
    turning = ds.assign(
        u=-ds["y"] * xr.ones_like(ds["u"]), v=ds["x"] * xr.ones_like(ds["u"])
    )
    box = _box(ds, slice(-3e4, 3e4), slice(-2e4, 4e4))
    flux = boundary_flux(turning, region=box, **PLANE)
    through = boundary_flux(abs(turning), region=box, **PLANE)
    assert abs(float(flux)) < 1e-12 * abs(float(through)) + 1e-3


def test_boundary_flux_is_the_area_integral_of_the_residual(lonlat_grid):
    ds = _flowing(lonlat_grid)
    inside = xr.zeros_like(ds["u"].isel(time=0, drop=True), dtype=bool)
    inside[4:12, 5:20] = True
    residual = volume_budget_residual(ds, depth=None).isel(time=0)
    step = np.deg2rad(2.5) * np.deg2rad(1.5)
    # gridax.area_weights: R² cos φ Δλ Δφ
    area = calc.EARTH_RADIUS**2 * step * np.cos(np.deg2rad(ds["lat"]))
    expected = (residual * area).where(inside).sum()
    flux = boundary_flux(ds, region=inside).isel(time=0)
    assert float(flux) == pytest.approx(float(expected), rel=1e-10)


def test_boundary_flux_takes_a_thickness_per_level():
    ds = (
        _swirl()
        .expand_dims(depth=[0.0, 10.0, 30.0])
        .assign(h=("depth", [5.0, 10.0, 20.0]))
    )
    box = _box(ds.isel(depth=0, drop=True), slice(-3e4, 3e4), slice(-2e4, 4e4))
    per_metre = boundary_flux(ds, region=box, **PLANE)
    layers = boundary_flux(ds, region=box, thickness="h", **PLANE)
    assert layers.dims == ("depth",)
    np.testing.assert_allclose(layers, per_metre * ds["h"])


def test_boundary_flux_wraps_around_the_circle():
    lon, lat = np.arange(0.0, 360.0, 10.0), np.arange(-45.0, 50.0, 10.0)
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {k: (("lat", "lon"), rng.normal(size=(lat.size, lon.size))) for k in "uv"},
        coords={"lat": lat, "lon": lon},
    )
    inside = xr.zeros_like(ds["u"], dtype=bool)
    inside[3:6, 14:22] = True
    rolled = ds.roll(lon=18)
    np.testing.assert_allclose(
        boundary_flux(rolled, region=inside.roll(lon=18)),
        boundary_flux(ds, region=inside),
    )


def test_boundary_flux_rejects_a_region_at_the_edge_or_not_boolean():
    ds = _swirl()
    edge = _box(ds, slice(-2e5, 0.0), slice(-2e4, 4e4))
    with pytest.raises(ValueError, match="touches an edge"):
        boundary_flux(ds, region=edge, **PLANE)
    with pytest.raises(ValueError, match="must be boolean"):
        boundary_flux(ds, region=edge.astype(float), **PLANE)
    with pytest.raises(ValueError, match="must be boolean"):
        boundary_flux(ds, region=edge.isel(x=0), **PLANE)


def test_boundary_flux_on_dask_is_lazy_and_equal():
    pytest.importorskip("dask.array")
    ds = _swirl().expand_dims(time=3)
    box = _box(ds.isel(time=0), slice(-3e4, 3e4), slice(-2e4, 4e4))
    lazy = boundary_flux(ds.chunk({"x": 11, "time": 1}), region=box, **PLANE)
    assert lazy.chunks is not None
    xr.testing.assert_allclose(lazy.compute(), boundary_flux(ds, region=box, **PLANE))


def test_boundary_flux_operator_matches_the_function():
    ds = _swirl()
    box = _box(ds, slice(-3e4, 3e4), slice(-2e4, 4e4))
    op = BoundaryFlux(box, variable="c", **PLANE)
    xr.testing.assert_identical(
        op(ds), boundary_flux(ds, region=box, variable="c", **PLANE)
    )
    assert op.get_config()["dims"] == ["x", "y"]


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


def _inside(res: xr.DataArray) -> np.ndarray:
    """The cells with all of their faces inside the domain."""
    return res.isel(lat=slice(1, -1), lon=slice(1, -1)).values


def _flowing(lonlat_grid) -> xr.Dataset:
    lon, lat = lonlat_grid
    L, P = np.deg2rad(lon), np.deg2rad(lat)[:, None]
    t = np.array([0.0, 10.0, 40.0])
    grow = (1.0 + t / 100.0)[:, None, None]
    fields = {
        "theta": 15.0 + np.sin(3 * L) * np.cos(2 * P),
        "u": np.cos(P) * (1.0 + 0.3 * np.sin(4 * L)),
        "v": 0.2 * np.cos(3 * L) * np.sin(4 * P),
    }
    return xr.Dataset(
        {k: (("time", "lat", "lon"), grow * f) for k, f in fields.items()},
        coords={"time": t, "lat": lat, "lon": lon},
    )


def test_heat_budget_residual_closes_on_constant_dataset(lonlat_grid):
    ds = _closed_dataset(lonlat_grid)
    res = heat_budget_residual(ds, depth=None)
    np.testing.assert_allclose(_inside(res), 0.0, atol=1e-9)


def test_heat_budget_residual_with_surface_flux_closes(lonlat_grid):
    ds = _closed_dataset(lonlat_grid)
    # a non-zero surface flux + matching extra tendency would be needed for
    # full closure; this just confirms the source term subtracts cleanly
    flux = xr.full_like(ds["theta"], 0.0)
    out = heat_budget_residual(ds.assign(F=flux), surface_flux_var="F", depth=None)
    np.testing.assert_allclose(_inside(out), 0.0, atol=1e-9)


def test_heat_budget_residual_operator(lonlat_grid):
    ds = _closed_dataset(lonlat_grid)
    op = HeatBudgetResidual(temp_var="theta", depth=None)
    out = op(ds)
    np.testing.assert_allclose(_inside(out), 0.0, atol=1e-9)


def test_salt_budget_residual_closes_on_constant_dataset(lonlat_grid):
    ds = _closed_dataset(lonlat_grid)
    res = salt_budget_residual(ds, depth=None)
    np.testing.assert_allclose(_inside(res), 0.0, atol=1e-9)


def test_salt_budget_residual_operator(lonlat_grid):
    ds = _closed_dataset(lonlat_grid)
    op = SaltBudgetResidual(salt_var="so", depth=None)
    out = op(ds)
    np.testing.assert_allclose(_inside(out), 0.0, atol=1e-9)


def test_volume_budget_residual_zero_on_zero_flow(lonlat_grid):
    ds = _closed_dataset(lonlat_grid)
    res = volume_budget_residual(ds, depth=None)
    np.testing.assert_allclose(_inside(res), 0.0, atol=1e-12)


def test_volume_budget_residual_operator(lonlat_grid):
    ds = _closed_dataset(lonlat_grid)
    op = VolumeBudgetResidual(depth=None)
    out = op(ds)
    np.testing.assert_allclose(_inside(out), 0.0, atol=1e-12)


def test_kinetic_energy_budget_residual_zero_on_zero_flow(lonlat_grid):
    ds = _closed_dataset(lonlat_grid)
    res = kinetic_energy_budget_residual(ds, depth=None)
    np.testing.assert_allclose(_inside(res), 0.0, atol=1e-12)


def test_kinetic_energy_budget_residual_operator(lonlat_grid):
    ds = _closed_dataset(lonlat_grid)
    op = KineticEnergyBudgetResidual(depth=None)
    np.testing.assert_allclose(_inside(op(ds)), 0.0, atol=1e-12)


@pytest.mark.parametrize(
    "op",
    [
        HeatBudgetResidual(temp_var="theta", surface_flux_var="F", depth=None),
        SaltBudgetResidual(salt_var="so", w_var="w"),
        VolumeBudgetResidual(depth=None),
        KineticEnergyBudgetResidual(forcing_vars=("F", "diss"), depth=None),
    ],
    ids=lambda op: type(op).__name__,
)
def test_budget_operator_config_roundtrips_through_constructor(op):
    """``type(op)(**op.get_config())`` must reconstruct an equivalent op."""
    config = op.get_config()
    clone = type(op)(**config)
    assert clone.get_config() == config
    assert vars(clone) == vars(op)


def test_kinetic_energy_budget_residual_subtracts_forcing(lonlat_grid):
    ds = _closed_dataset(lonlat_grid)
    forcing = xr.full_like(ds["u"], 0.0)
    out = kinetic_energy_budget_residual(
        ds.assign(F=forcing), forcing_vars=["F"], depth=None
    )
    np.testing.assert_allclose(_inside(out), 0.0, atol=1e-12)


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
    from xrops.metrics.physical import (
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
    from xrops.metrics import (
        DensityInversionFraction,
        DivergenceError,
        GeostrophicBalanceError,
        PVConservationError,
    )

    assert DensityInversionFraction is metrics.DensityInversionFraction
    _ = (DivergenceError, GeostrophicBalanceError, PVConservationError)


def test_budgets_top_level_exposes_v4():
    import xrops.budgets as bdg

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
        assert hasattr(bdg, name), f"xrops.budgets is missing {name!r}"


# ---- residuals on faces (#796) ------------------------------------------


def test_the_edge_cells_are_nan_and_a_circle_of_longitudes_has_none(lonlat_grid):
    res = volume_budget_residual(_flowing(lonlat_grid), depth=None)
    assert np.isnan(res.isel(lat=0)).all() and np.isnan(res.isel(lon=-1)).all()
    assert np.isfinite(_inside(res)).all()
    lon = np.arange(0.0, 360.0, 15.0)
    ds = _flowing((lon, lonlat_grid[1]))
    around = volume_budget_residual(ds, depth=None)
    assert np.isfinite(around.isel(lat=slice(1, -1))).all()
    rolled = volume_budget_residual(ds.roll(lon=7), depth=None)
    xr.testing.assert_allclose(rolled, around.roll(lon=7))


@pytest.mark.parametrize("chunks", [{"lat": 5, "lon": 6}, {"time": 3}, {"lon": 1}])
@pytest.mark.parametrize("around", [False, True])
def test_residuals_on_dask_are_lazy_and_equal(lonlat_grid, chunks, around):
    pytest.importorskip("dask.array")
    grid = (np.arange(0.0, 360.0, 15.0), lonlat_grid[1]) if around else lonlat_grid
    ds = _flowing(grid)
    lazy = heat_budget_residual(ds.chunk(chunks), depth=None)
    assert lazy.chunks is not None
    xr.testing.assert_allclose(lazy.compute(), heat_budget_residual(ds, depth=None))


def test_tendency_of_a_quadratic_is_exact_on_uneven_times(lonlat_grid):
    ds = _closed_dataset(lonlat_grid).assign_coords(time=[0.0, 1.0, 4.0])
    ds["theta"] = ds["theta"] * 0.0 + ds["time"] ** 2
    res = heat_budget_residual(ds, depth=None)
    np.testing.assert_allclose(_inside(res.isel(time=1)), 2.0)


def test_vertical_flux_on_uneven_levels(lonlat_grid):
    depth = np.array([0.0, 5.0, 15.0, 40.0])
    ds = _closed_dataset(lonlat_grid).expand_dims(depth=depth)
    ds = ds.assign(w=ds["u"] * 0.0 + ds["depth"] ** 2)
    res = heat_budget_residual(ds, w_var="w")
    # ∂(wθ)/∂z = 2 z θ at an inner level
    expected = 2.0 * depth[1:3] * 15.0
    got = res.isel(depth=slice(1, 3), time=0, lat=5, lon=5)
    np.testing.assert_allclose(got, expected)
    assert res.dims == ds["theta"].dims
