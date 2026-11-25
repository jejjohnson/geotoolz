"""The horizontal diagnostics on gridax's grids: spheres, dims, dask (#749, #793)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr

from xrops import grad, ocn
from xrops.metrics import divergence_error, geostrophic_balance_error


OMEGA = 1e-5  # rotation rate of the flow (s⁻¹)


def _solid_body() -> xr.Dataset:
    """u = ωR cos φ, v = 0: a rotation, which has no strain."""
    lat, lon = np.linspace(-70.0, 70.0, 141), np.linspace(0.0, 40.0, 41)
    u = OMEGA * grad.EARTH_RADIUS * np.cos(np.deg2rad(lat))[:, None]
    return xr.Dataset(
        {
            "u": (("lat", "lon"), u * np.ones(41)),
            "v": (("lat", "lon"), 0.0 * u + np.zeros(41)),
        },
        coords={"lat": lat, "lon": lon},
    )


def _flow(*, dtype: str = "float64") -> xr.Dataset:
    lat, lon = np.arange(20.0, 50.0), np.arange(-40.0, 0.0)
    lam, phi = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat))
    swell = np.array([1.0, 2.0, -0.5])[:, None, None]
    fields = {
        "u": np.cos(phi) * (1.0 + 0.3 * np.sin(4 * lam)) * swell,
        "v": 0.3 * np.cos(3 * lam) * np.sin(6 * phi) * swell,
        "ssh": np.sin(3 * lam) * np.cos(4 * phi) * swell,
        "h": 100.0 + np.sin(2 * lam) * swell,
    }
    return xr.Dataset(
        {name: (("time", "lat", "lon"), f.astype(dtype)) for name, f in fields.items()},
        coords={"lat": lat, "lon": lon, "time": np.arange(3)},
    )


CALLS: dict[str, Callable[[xr.Dataset], xr.Dataset | xr.DataArray]] = {
    "geostrophic": ocn.geostrophic_velocities,
    "ageostrophic": ocn.ageostrophic_velocities,
    "vorticity": ocn.relative_vorticity,
    "absolute": ocn.absolute_vorticity,
    "divergence": ocn.divergence,
    "shear_strain": ocn.shear_strain,
    "tensor_strain": ocn.tensor_strain,
    "strain": ocn.strain_magnitude,
    "okubo_weiss": ocn.okubo_weiss,
    "advection": lambda ds: ocn.advection(ds, "ssh"),
    "shear_vorticity": ocn.shear_vorticity,
    "curvature_vorticity": ocn.curvature_vorticity,
    "frontogenesis": lambda ds: ocn.frontogenesis(ds, "ssh"),
    "pv": ocn.potential_vorticity_barotropic,
    "balance": geostrophic_balance_error,
    "divergence_error": divergence_error,
}


def test_a_rotation_has_no_strain() -> None:
    ds = _solid_body()
    for strain in (
        ocn.shear_strain(ds)["shear_strain"],
        ocn.tensor_strain(ds)["tensor_strain"],
        ocn.strain_magnitude(ds)["strain"],
    ):
        # the first and last rows have a first-order difference
        assert float(abs(strain).isel(lat=slice(1, -1)).max()) < 1e-3 * OMEGA


def test_okubo_weiss_of_a_rotation() -> None:
    ds = _solid_body()
    expected = -4.0 * OMEGA**2 * np.sin(np.deg2rad(ds["lat"])) ** 2
    got = ocn.okubo_weiss(ds)["ow"]
    np.testing.assert_allclose(
        got.isel(lat=slice(1, -1), lon=0),
        expected.isel(lat=slice(1, -1)),
        atol=2e-3 * OMEGA**2,
    )


def test_frontogenesis_of_a_rotation_is_zero() -> None:
    ds = _solid_body().assign(
        c=lambda d: np.cos(np.deg2rad(d["lon"])) * xr.ones_like(d["u"])
    )
    front = ocn.frontogenesis(ds, "c")["c_frontogenesis"].isel(lat=slice(1, -1))
    size = ocn.advection(ds, "c")["c_advection"]
    assert float(abs(front).max()) < 1e-3 * float(abs(size).max())


@pytest.mark.parametrize("name", CALLS)
@pytest.mark.parametrize(
    "chunks", [{"lat": 7, "lon": 9}, {"lat": 1, "lon": 40}, {"time": 1, "lat": -1}]
)
def test_dask_equals_numpy_and_stays_lazy(name: str, chunks: dict[str, int]) -> None:
    pytest.importorskip("dask.array")
    ds = _flow()
    lazy = CALLS[name](ds.chunk(chunks))
    variables = lazy.data_vars.values() if isinstance(lazy, xr.Dataset) else [lazy]
    assert all(v.chunks is not None for v in variables)
    xr.testing.assert_allclose(lazy.compute(), CALLS[name](ds), rtol=1e-12, atol=1e-20)


@pytest.mark.parametrize("name", CALLS)
def test_dims_in_any_order(name: str) -> None:
    dims = ("lon", "time", "lat")
    ds = _flow()
    turned = CALLS[name](ds.transpose(*dims))
    variables = (
        turned.data_vars.values() if isinstance(turned, xr.Dataset) else [turned]
    )
    assert all(v.dims == dims for v in variables)
    xr.testing.assert_allclose(turned, CALLS[name](ds).transpose(*dims), rtol=1e-12)


@pytest.mark.parametrize("name", CALLS)
def test_rows_from_north_to_south(name: str) -> None:
    ds = _flow()
    flipped = CALLS[name](ds.isel(lat=slice(None, None, -1)))
    xr.testing.assert_allclose(flipped.sortby("lat"), CALLS[name](ds), rtol=1e-10)


def test_advection_takes_the_pair_either_way_round() -> None:
    ds = _flow()
    xr.testing.assert_identical(
        ocn.advection(ds, "ssh", ("v", "u"), ("lat", "lon")), ocn.advection(ds, "ssh")
    )


def test_an_extra_dim_on_one_component_broadcasts() -> None:
    ds = _flow()
    surface = ds.assign(v=ds["v"].isel(time=0, drop=True))
    got = ocn.divergence(surface)["div"].isel(time=1)
    each = ds.isel(time=1).assign(v=ds["v"].isel(time=0, drop=True))
    xr.testing.assert_allclose(got, ocn.divergence(each)["div"])


def test_an_integer_coordinate_is_read_as_it_is() -> None:
    ds = _flow().isel(time=0)
    whole = ds.assign_coords(lat=ds["lat"].astype(int), lon=ds["lon"].astype(int))
    np.testing.assert_array_equal(
        ocn.divergence(whole)["div"], ocn.divergence(ds)["div"]
    )


def test_float32_fields_stay_float32() -> None:
    ds = _flow(dtype="float32")
    assert ocn.relative_vorticity(ds)["vort_r"].dtype == np.float32
    assert ocn.relative_vorticity(ds.chunk({"lat": 10}))["vort_r"].dtype == np.float32


def test_a_gap_costs_the_cells_whose_stencil_reads_it() -> None:
    ds = _flow().isel(time=0)
    ds["u"][10, 10] = np.nan
    holes = np.argwhere(np.isnan(ocn.divergence(ds)["div"].values))
    assert {tuple(map(int, hole)) for hole in holes} == {(10, 9), (10, 10), (10, 11)}


@pytest.mark.parametrize("geometry", ["spherical", "cartesian"])
def test_an_uneven_coordinate_raises(geometry: str) -> None:
    ds = _flow().isel(time=0)
    uneven = ds.assign_coords(lat=ds["lat"] ** 1.1)
    with pytest.raises(ValueError, match="not uniformly spaced"):
        ocn.divergence(uneven, geometry=geometry)


def test_an_uneven_plane_is_exact_for_a_quadratic() -> None:
    rng = np.random.default_rng(0)
    x, y = (np.cumsum(rng.uniform(0.5, 2.0, 20)) * 1e3 for _ in range(2))
    ds = xr.Dataset(
        {
            "u": (("y", "x"), x[None, :] ** 2 + 0 * y[:, None]),
            "v": (("y", "x"), y[:, None] ** 2 + 0 * x),
        },
        coords={"x": x, "y": y},
    )
    div = ocn.divergence(ds, dims=("x", "y"), geometry="rectilinear")["div"]
    exact = 2.0 * x[None, :] + 2.0 * y[:, None]
    np.testing.assert_allclose(div[1:-1, 1:-1], exact[1:-1, 1:-1], rtol=1e-11)


def test_no_flow_is_geostrophic_at_the_equator() -> None:
    ds = _flow().isel(time=0)
    across = ds.assign_coords(lat=ds["lat"] - 30.0)
    out = ocn.geostrophic_velocities(across)
    assert (
        np.isnan(out["u"].sel(lat=0.0)).all() and np.isnan(out["v"].sel(lat=0.0)).all()
    )
    assert np.isfinite(out["u"].sel(lat=1.0)).all()


def test_an_unknown_geometry_raises() -> None:
    with pytest.raises(ValueError, match="Unknown geometry"):
        ocn.divergence(_flow(), geometry="polar")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "name", [n for n in CALLS if n not in ("geostrophic", "ageostrophic")]
)
def test_a_diagnostic_does_not_have_the_units_of_its_input(name: str) -> None:
    ds = _flow()
    for variable in ds.data_vars.values():
        variable.attrs["units"] = "furlong"
    out = CALLS[name](ds)
    variables = out.data_vars.values() if isinstance(out, xr.Dataset) else [out]
    assert all("units" not in v.attrs for v in variables)


def test_energies_do_not_have_the_units_of_a_velocity() -> None:
    ds = _flow()
    ds["u"].attrs["units"] = ds["v"].attrs["units"] = "m s-1"
    assert "units" not in ocn.kinetic_energy(ds)["ke"].attrs
    assert "units" not in ocn.eddy_kinetic_energy(ds, "u", "v")["eke"].attrs
    assert "units" not in ocn.enstrophy(ds, "u")["ens"].attrs
    assert ocn.kinetic_energy(ds)["ke"].attrs["long_name"] == "Kinetic Energy"
