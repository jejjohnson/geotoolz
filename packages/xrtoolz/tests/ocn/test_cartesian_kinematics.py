"""Cartesian / f-β-plane support in :mod:`xrtoolz.ocn` kinematics (#322)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import xrgrad
from xrtoolz.ocn import (
    AbsoluteVorticity,
    Advection,
    GeostrophicVelocities,
    OkuboWeiss,
    RelativeVorticity,
    absolute_vorticity,
    advection,
    divergence,
    geostrophic_velocities,
    okubo_weiss,
    potential_vorticity_barotropic,
    relative_vorticity,
)


CART = {"dims": ("x", "y"), "geometry": "cartesian"}
F0, BETA = 1e-4, 1.6e-11


def _grid(n: int = 41, half: float = 2e5) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-half, half, n)
    return np.meshgrid(x, x, indexing="xy")


def _uv(u: np.ndarray, v: np.ndarray) -> xr.Dataset:
    n = u.shape[0]
    x = np.linspace(-2e5, 2e5, n)
    return xr.Dataset(
        {"u": (("y", "x"), u), "v": (("y", "x"), v)}, coords={"x": x, "y": x}
    )


def test_solid_body_rotation_vorticity_is_two() -> None:
    X, Y = _grid()
    zeta = relative_vorticity(_uv(-Y, X), **CART)["vort_r"]
    np.testing.assert_allclose(zeta.values, 2.0, rtol=1e-10)


def test_divergence_and_okubo_weiss_of_pure_strain() -> None:
    # u = x, v = -y: non-divergent, irrotational, Sn = 2, Ss = 0 → W = 4.
    X, Y = _grid()
    ds = _uv(X, -Y)
    np.testing.assert_allclose(divergence(ds, **CART)["div"].values, 0.0, atol=1e-12)
    np.testing.assert_allclose(okubo_weiss(ds, **CART)["ow"].values, 4.0, rtol=1e-10)


def test_beta_plane_geostrophy_of_gaussian_ssh() -> None:
    X, Y = _grid(n=201)
    L, A = 5e4, 0.3
    eta = A * np.exp(-(X**2 + Y**2) / L**2)
    x = X[0]
    ds = xr.Dataset({"eta": (("y", "x"), eta)}, coords={"x": x, "y": x})

    out = geostrophic_velocities(ds, "eta", **CART, f=(F0, BETA))

    # y₀ defaults to the domain centre, which is 0 here.
    f = F0 + BETA * Y
    u_true = -(xrgrad.GRAVITY / f) * (-2 * Y / L**2) * eta
    v_true = (xrgrad.GRAVITY / f) * (-2 * X / L**2) * eta
    # Second-order central differences at 25 points per L: ~0.15 % error.
    scale = np.abs(u_true).max()
    np.testing.assert_allclose(out["u"].values, u_true, atol=5e-3 * scale)
    np.testing.assert_allclose(out["v"].values, v_true, atol=5e-3 * scale)


@pytest.mark.parametrize(
    ("f", "expected"),
    [
        (F0, lambda y: np.full_like(y, F0)),
        ((F0, BETA), lambda y: F0 + BETA * y),
        ((F0, BETA, -2e5), lambda y: F0 + BETA * (y + 2e5)),
    ],
)
def test_coriolis_specs_feed_absolute_vorticity(f, expected) -> None:
    X, Y = _grid()
    ds = _uv(np.zeros_like(X), np.zeros_like(X))
    vort_a = absolute_vorticity(ds, **CART, f=f)["vort_a"]
    np.testing.assert_allclose(vort_a.values, expected(Y), rtol=1e-12)


def test_coriolis_dataarray_is_used_as_is() -> None:
    X, _ = _grid()
    ds = _uv(np.zeros_like(X), np.zeros_like(X)).assign(
        h=(("y", "x"), np.full_like(X, 2.0))
    )
    f = xr.full_like(ds["u"], 3e-5)
    pv = potential_vorticity_barotropic(ds, **CART, f=f)["pv_barotropic"]
    np.testing.assert_allclose(pv.values, 1.5e-5)


def test_missing_coriolis_on_cartesian_grid_raises() -> None:
    X, Y = _grid()
    with pytest.raises(ValueError, match="pass f="):
        absolute_vorticity(_uv(-Y, X), **CART)


def test_bad_beta_plane_tuple_raises() -> None:
    X, Y = _grid()
    with pytest.raises(ValueError, match="β-plane"):
        absolute_vorticity(_uv(-Y, X), **CART, f=(F0,))  # ty: ignore[invalid-argument-type]


def test_cartesian_advection_of_linear_tracer() -> None:
    X, _ = _grid()
    ds = _uv(np.ones_like(X), np.zeros_like(X)).assign(c=(("y", "x"), 2.0 * X))
    adv = advection(ds, "c", dims=("x", "y"), geometry="cartesian")["c_advection"]
    np.testing.assert_allclose(adv.values, -2.0, rtol=1e-10)


def test_operators_forward_geometry_and_round_trip_config() -> None:
    X, Y = _grid()
    ds = _uv(-Y, X).assign(eta=(("y", "x"), 1e-6 * (X**2 + Y**2)), c=(("y", "x"), X))
    cases = [
        (RelativeVorticity(**CART), relative_vorticity(ds, **CART)),
        (OkuboWeiss(**CART), okubo_weiss(ds, **CART)),
        (AbsoluteVorticity(**CART, f=F0), absolute_vorticity(ds, **CART, f=F0)),
        (
            GeostrophicVelocities("eta", **CART, f=(F0, BETA)),
            geostrophic_velocities(ds, "eta", **CART, f=(F0, BETA)),
        ),
        (
            Advection("c", dim=("x", "y"), geometry="cartesian"),
            advection(ds, "c", dims=("x", "y"), geometry="cartesian"),
        ),
    ]
    for op, expected in cases:
        out = op(ds)
        for name in expected.data_vars:
            xr.testing.assert_allclose(out[name], expected[name])
        config = op.get_config()
        assert config["geometry"] == "cartesian"
        rebuilt = type(op)(**config)
        assert rebuilt.get_config() == config
