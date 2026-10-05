"""Cartesian / β-plane support in :mod:`xrtoolz.metrics.physical` (#323).

The analytic cases mirror somax's ``tests/eval/test_diagnostics.py`` so the
scores somax hand-rolls in ``somax.eval`` can come from xrtoolz instead.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import xrgrad
from xrtoolz.metrics import (
    DivergenceError,
    GeostrophicBalanceError,
    GeostrophicImbalance,
    divergence_error,
    geostrophic_balance_error,
    geostrophic_imbalance,
)


F0 = 1e-4
CART = {"dims": ("x", "y"), "geometry": "cartesian"}


def _state(eta, u, v, n: int = 32, length: float = 1e6) -> xr.Dataset:
    x = np.arange(n) * (length / n)
    shape = (n, n)
    data = {
        name: (("y", "x"), np.broadcast_to(np.asarray(val, dtype=float), shape))
        for name, val in {"ssh": eta, "u": u, "v": v}.items()
    }
    return xr.Dataset(data, coords={"x": x, "y": x})


def _balanced_jet(slope: float = 1e-3, n: int = 32) -> xr.Dataset:
    # η linear in x, v = (g/f) ∂η/∂x, u = 0: exact discrete geostrophy.
    x = np.arange(n) * (1e6 / n)
    eta = np.broadcast_to(slope * x[None, :], (n, n))
    return _state(eta, 0.0, (xrgrad.GRAVITY / F0) * slope, n=n)


def test_imbalance_is_unity_for_pure_inertial_flow() -> None:
    ds = _state(0.0, 0.5, -0.3)
    score = geostrophic_imbalance(ds, **CART, f=F0)
    assert float(score) == pytest.approx(1.0, rel=1e-6)


def test_imbalance_near_zero_for_balanced_jet() -> None:
    assert float(geostrophic_imbalance(_balanced_jet(), **CART, f=F0)) < 1e-6


def test_imbalance_is_scale_invariant() -> None:
    rng = np.random.default_rng(0)
    base = _state(rng.normal(size=(32, 32)), 0.5, -0.3)
    scaled = base * 3.0
    a = geostrophic_imbalance(base, **CART, f=(F0, 1.6e-11))
    b = geostrophic_imbalance(scaled, **CART, f=(F0, 1.6e-11))
    assert float(b) == pytest.approx(float(a), rel=1e-10)


def test_balance_residual_fields_vanish_for_balanced_jet() -> None:
    res = geostrophic_balance_error(_balanced_jet(), **CART, f=F0)
    np.testing.assert_allclose(res["r_u"].values, 0.0, atol=1e-12)
    np.testing.assert_allclose(res["r_v"].values, 0.0, atol=1e-12)


def test_cartesian_balance_requires_explicit_f() -> None:
    with pytest.raises(ValueError, match="pass f="):
        geostrophic_balance_error(_balanced_jet(), **CART)


def test_rms_divergence_zero_for_divergence_free_field() -> None:
    n = 32
    x = np.arange(n) * 1e4
    X, Y = np.meshgrid(x, x)
    ds = _state(0.0, -Y * 1e-5, X * 1e-5, n=n).assign_coords(x=x, y=x)
    score = divergence_error(ds, **CART, reduce="rms")
    assert score.name == "rms_divergence"
    assert score.ndim == 0
    assert float(score) < 1e-15


def test_rms_divergence_of_uniform_expansion() -> None:
    # u = a x, v = a y → ∇·u = 2a everywhere.
    n, a = 32, 1e-5
    x = np.arange(n) * 1e4
    X, Y = np.meshgrid(x, x)
    ds = _state(0.0, a * X, a * Y, n=n).assign_coords(x=x, y=x)
    assert float(divergence_error(ds, **CART, reduce="rms")) == pytest.approx(2 * a)
    field = divergence_error(ds, **CART)
    assert field.name == "divergence"
    np.testing.assert_allclose(field.values, 2 * a, rtol=1e-10)


def test_divergence_error_rejects_unknown_reduce() -> None:
    with pytest.raises(ValueError, match="reduce"):
        divergence_error(_balanced_jet(), **CART, reduce="mean")  # ty: ignore[invalid-argument-type]


def test_operators_forward_and_round_trip_config() -> None:
    ds = _balanced_jet()
    cases = [
        (GeostrophicImbalance(**CART, f=F0), geostrophic_imbalance(ds, **CART, f=F0)),
        (
            DivergenceError(**CART, reduce="rms"),
            divergence_error(ds, **CART, reduce="rms"),
        ),
    ]
    for op, expected in cases:
        xr.testing.assert_identical(op(ds), expected)
        assert type(op)(**op.get_config()).get_config() == op.get_config()
    op = GeostrophicBalanceError(**CART, f=F0)
    xr.testing.assert_identical(op(ds), geostrophic_balance_error(ds, **CART, f=F0))
    assert op.get_config()["geometry"] == "cartesian"
