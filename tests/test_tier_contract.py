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
    """Tier A is reachable via ``xrtoolz.metrics.array``."""
    from xrtoolz.metrics import array as ma

    assert hasattr(ma, name), f"xrtoolz.metrics.array.{name} missing"
    assert callable(getattr(ma, name))


@pytest.mark.parametrize("name", _PIXEL_NAMES)
def test_metrics_tier_b_matches_tier_a(name: str) -> None:
    """Tier B (Dataset, ``dim=``) numerically matches Tier A (axis=)."""
    from xrtoolz.metrics import array as ma
    from xrtoolz.metrics._src import pixel as tier_b

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
    from xrtoolz.metrics import operators as ops
    from xrtoolz.metrics._src import pixel as tier_b

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
    """Tier A is reachable via ``xrtoolz.transforms.array``."""
    from xrtoolz.transforms import array as ta

    for name in ("fft", "ifft", "power_spectrum"):
        assert hasattr(ta, name)
        assert callable(getattr(ta, name))


def test_transforms_array_fft_roundtrip() -> None:
    """``ifft(fft(x)) == x`` exercises the Tier A kernel on its own."""
    from xrtoolz.transforms import array as ta

    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 16))
    np.testing.assert_allclose(ta.ifft(ta.fft(x, axis=-1), axis=-1).real, x, atol=1e-10)


def test_transforms_array_power_spectrum_matches_manual_fft() -> None:
    """Tier A ``power_spectrum`` matches a hand-rolled ``|fft|**2``."""
    from xrtoolz.transforms import array as ta

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
    """Tier A is reachable via ``xrtoolz.calc.array``."""
    from xrtoolz.calc import array as ca

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
    from xrtoolz.calc import array as ca, partial as tier_b_partial

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
    from xrtoolz.calc import array as ca

    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 5, 6))
    components = ca.gradient(x, axes=(0, 2), spacing=(0.5, 2.0))
    assert len(components) == 2
    for comp in components:
        assert comp.shape == x.shape


# ---------------------------------------------------------------------------
# interpolate (Tier A array kernels — F3.2 / F3.3)
# ---------------------------------------------------------------------------


_INTERPOLATE_ARRAY_NAMES: tuple[str, ...] = (
    "moving_average",
    "gaussian_smooth",
    "gaussian_smooth_nd",
    "lowpass_filter",
    "remap_axis",
)


@pytest.mark.parametrize("name", _INTERPOLATE_ARRAY_NAMES)
def test_interpolate_array_namespace_exports(name: str) -> None:
    """Tier A is reachable via ``xrtoolz.interpolate.array``."""
    from xrtoolz.interpolate import array as ia

    assert hasattr(ia, name), f"xrtoolz.interpolate.array.{name} missing"
    assert callable(getattr(ia, name))


def test_interpolate_smooth_tier_b_matches_tier_a() -> None:
    """Tier B (Dataset, ``dim=``) numerically matches Tier A (``axis=``)."""
    from xrtoolz.interpolate import array as ia
    from xrtoolz.interpolate._src import smooth as tier_b

    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 64))
    ds = xr.Dataset({"x": (("a", "time"), x)})

    b_ma = tier_b.moving_average(ds, dim="time", window=5)["x"].values
    a_ma = ia.moving_average(x, axis=-1, window=5)
    np.testing.assert_allclose(a_ma, b_ma)

    b_g = tier_b.gaussian_smooth(ds, dim="time", sigma=2.0)["x"].values
    a_g = ia.gaussian_smooth(x, axis=-1, sigma=2.0)
    np.testing.assert_allclose(a_g, b_g)

    b_lp = tier_b.lowpass_filter(ds, dim="time", cutoff=0.1)["x"].values
    a_lp = ia.lowpass_filter(x, axis=-1, cutoff=0.1)
    np.testing.assert_allclose(a_lp, b_lp)


def test_interpolate_smooth_tier_c_matches_tier_b() -> None:
    """Tier C ``Operator`` output equals Tier B function output."""
    from xrtoolz.interpolate._src import smooth as tier_b
    from xrtoolz.interpolate.operators import (
        GaussianSmooth,
        LowpassFilter,
        MovingAverage,
    )

    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 64))
    ds = xr.Dataset({"x": (("a", "time"), x)})

    np.testing.assert_allclose(
        MovingAverage("time", window=5)(ds)["x"].values,
        tier_b.moving_average(ds, dim="time", window=5)["x"].values,
    )
    np.testing.assert_allclose(
        GaussianSmooth("time", sigma=2.0)(ds)["x"].values,
        tier_b.gaussian_smooth(ds, dim="time", sigma=2.0)["x"].values,
    )
    np.testing.assert_allclose(
        LowpassFilter("time", cutoff=0.1)(ds)["x"].values,
        tier_b.lowpass_filter(ds, dim="time", cutoff=0.1)["x"].values,
    )


def test_interpolate_coord_remap_tier_b_matches_tier_a() -> None:
    """Tier B ``remap_axis`` matches the Tier A kernel on the same data."""
    from xrtoolz.interpolate import array as ia
    from xrtoolz.interpolate._src import coord_remap as tier_b

    src = np.linspace(0.0, 100.0, 21)
    tgt = np.linspace(0.0, 100.0, 11)
    rng = np.random.default_rng(0)
    field = rng.standard_normal((src.size, 4))
    ds = xr.Dataset(
        {"f": (("depth", "x"), field)},
        coords={"depth": src, "x": np.arange(4)},
    )
    b_out = tier_b.remap_axis(ds, source_dim="depth", target_coords=tgt)["f"].values
    a_out = ia.remap_axis(field, axis=0, source_coords=src, target_coords=tgt)
    np.testing.assert_allclose(a_out, b_out)


def test_interpolate_coord_remap_tier_c_matches_tier_b() -> None:
    """Tier C ``RemapAxis`` Operator output equals Tier B function output."""
    from xrtoolz.interpolate._src import coord_remap as tier_b
    from xrtoolz.interpolate.operators import RemapAxis

    src = np.linspace(0.0, 100.0, 21)
    tgt = np.linspace(0.0, 100.0, 11)
    rng = np.random.default_rng(1)
    field = rng.standard_normal((src.size, 4))
    ds = xr.Dataset(
        {"f": (("depth", "x"), field)},
        coords={"depth": src, "x": np.arange(4)},
    )
    np.testing.assert_allclose(
        RemapAxis("depth", tgt)(ds)["f"].values,
        tier_b.remap_axis(ds, source_dim="depth", target_coords=tgt)["f"].values,
    )
