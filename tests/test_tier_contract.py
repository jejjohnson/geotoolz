"""Three-tier type contract tests (D11).

For each pilot module (metrics / transforms / calc), assert:

(a) **Tier A is reachable** — the ``<module>.array`` namespace re-exports
    the duck-array kernels.
(b) **Tier B numerically agrees with Tier A on numpy arrays** — wrapping
    a numpy array in :class:`xr.Dataset` and calling the Tier B function
    matches the Tier A kernel called on the raw array.
(c) **Tier C numerically agrees with Tier B** — the Operator wrapper
    produces the same value as the underlying Tier B function.

Numerical equivalence is the simplest contract — it catches both
"reimplemented Tier B math" and "broken delegation" without needing to
spy on call sites.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


_PIXEL_NAMES: tuple[str, ...] = (
    "mse",
    "rmse",
    "mae",
    "bias",
    "nrmse",
    "correlation",
    "r2_score",
)


@pytest.mark.parametrize("name", _PIXEL_NAMES)
def test_metrics_array_namespace_exports(name: str) -> None:
    """Tier A is reachable via ``xr_toolz.metrics.array``."""
    from xr_toolz.metrics import array as ma

    assert hasattr(ma, name), f"xr_toolz.metrics.array.{name} missing"
    assert callable(getattr(ma, name))


@pytest.mark.parametrize("name", _PIXEL_NAMES)
def test_metrics_tier_b_matches_tier_a(name: str) -> None:
    """Tier B (Dataset, ``dim=``) numerically matches Tier A (axis=)."""
    from xr_toolz.metrics import array as ma
    from xr_toolz.metrics._src import pixel as tier_b

    rng = np.random.default_rng(42)
    pred = rng.standard_normal((5, 8))
    ref = rng.standard_normal((5, 8))

    a_out = getattr(ma, name)(pred, ref, axis=-1)
    ds_pred = xr.Dataset({"x": (("a", "b"), pred)})
    ds_ref = xr.Dataset({"x": (("a", "b"), ref)})
    b_out = getattr(tier_b, name)(ds_pred, ds_ref, "x", "b").values

    np.testing.assert_allclose(a_out, b_out, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "op_name,fn_name",
    [
        ("MSE", "mse"),
        ("RMSE", "rmse"),
        ("MAE", "mae"),
        ("Bias", "bias"),
        ("NRMSE", "nrmse"),
        ("Correlation", "correlation"),
        ("R2Score", "r2_score"),
    ],
)
def test_metrics_tier_c_matches_tier_b(op_name: str, fn_name: str) -> None:
    """Tier C Operator output equals Tier B function output."""
    from xr_toolz.metrics import operators as ops
    from xr_toolz.metrics._src import pixel as tier_b

    rng = np.random.default_rng(7)
    pred = rng.standard_normal((3, 6))
    ref = rng.standard_normal((3, 6))
    ds_pred = xr.Dataset({"x": (("a", "b"), pred)})
    ds_ref = xr.Dataset({"x": (("a", "b"), ref)})

    op = getattr(ops, op_name)(variable="x", dims="b")
    c_out = op(ds_pred, ds_ref).values
    b_out = getattr(tier_b, fn_name)(ds_pred, ds_ref, "x", "b").values

    np.testing.assert_allclose(c_out, b_out, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------


def test_transforms_array_namespace_exports() -> None:
    """Tier A is reachable via ``xr_toolz.transforms.array``."""
    from xr_toolz.transforms import array as ta

    for name in ("fft", "ifft", "power_spectrum"):
        assert hasattr(ta, name)
        assert callable(getattr(ta, name))


def test_transforms_array_fft_roundtrip() -> None:
    """``ifft(fft(x)) == x`` exercises the Tier A kernel on its own."""
    from xr_toolz.transforms import array as ta

    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 16))
    np.testing.assert_allclose(ta.ifft(ta.fft(x, axis=-1), axis=-1).real, x, atol=1e-10)


def test_transforms_array_power_spectrum_matches_manual_fft() -> None:
    """Tier A ``power_spectrum`` matches a hand-rolled ``|fft|**2``."""
    from xr_toolz.transforms import array as ta

    rng = np.random.default_rng(1)
    x = rng.standard_normal((8,))
    power, (freqs,) = ta.power_spectrum(x, axis=-1, d=0.5, norm="ortho")
    expected = np.abs(np.fft.fftn(x, axes=(-1,), norm="ortho")) ** 2
    np.testing.assert_allclose(power, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(freqs, np.fft.fftfreq(8, d=0.5))


# ---------------------------------------------------------------------------
# calc
# ---------------------------------------------------------------------------


def test_calc_array_namespace_exports() -> None:
    """Tier A is reachable via ``xr_toolz.calc.array``."""
    from xr_toolz.calc import array as ca

    for name in ("partial", "gradient"):
        assert hasattr(ca, name)
        assert callable(getattr(ca, name))


def test_calc_array_partial_matches_tier_b_cartesian() -> None:
    """Tier A ``partial`` (numpy central diff) matches the Tier B cartesian
    partial for the default 2nd-order central scheme.

    The Tier B path runs through ``finitediffx``; the equivalence here
    checks the two numerical engines agree on a smooth field at default
    accuracy, which is the contract D11 cares about for the array tier.
    """
    from xr_toolz.calc import array as ca, partial as tier_b_partial

    x_coord = np.linspace(0.0, 2.0 * np.pi, 64)
    y_coord = np.linspace(0.0, 2.0 * np.pi, 32)
    field = np.sin(x_coord)[:, None] + np.cos(y_coord)[None, :]
    da = xr.DataArray(field, dims=("x", "y"), coords={"x": x_coord, "y": y_coord})

    dx = float(x_coord[1] - x_coord[0])
    a_dfdx = ca.partial(field, axis=0, spacing=dx)
    b_dfdx = tier_b_partial(da, "x", geometry="cartesian").values

    # Compare on the interior to avoid any boundary-stencil divergence.
    np.testing.assert_allclose(a_dfdx[1:-1], b_dfdx[1:-1], rtol=1e-6, atol=1e-6)


def test_calc_array_gradient_returns_per_axis() -> None:
    """Tier A ``gradient`` returns one component per requested axis."""
    from xr_toolz.calc import array as ca

    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 5, 6))
    components = ca.gradient(x, axes=(0, 2), spacing=(0.5, 2.0))
    assert len(components) == 2
    for comp in components:
        assert comp.shape == x.shape
