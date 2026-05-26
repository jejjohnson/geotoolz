"""Tests for ``xrtoolz.interpolate`` coord_remap (F3.2, D12)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrtoolz.interpolate import remap_axis, to_phase
from xrtoolz.interpolate.operators import (
    FromSigma,
    RemapAxis,
    ToHeight,
    ToIsopycnal,
    ToPhase,
    ToPressureLevels,
    ToSigma,
)
from xrtoolz.transforms import (
    remap_axis as transforms_remap_axis,
    to_phase as transforms_to_phase,
)
from xrtoolz.transforms._src import _coord_remap_kernels as ia


# ---------------------------------------------------------------------------
# Private kernels — remap_axis
# ---------------------------------------------------------------------------


def test_coord_remap_reexport_from_transforms() -> None:
    assert remap_axis is transforms_remap_axis
    assert to_phase is transforms_to_phase


def test_array_remap_identity_when_target_equals_source():
    src = np.linspace(0.0, 10.0, 11)
    values = np.sin(src)
    out = ia.remap_axis(values, axis=-1, source_coords=src, target_coords=src)
    np.testing.assert_allclose(out, values)


def test_array_remap_linear_recovers_known_function():
    src = np.linspace(0.0, 1.0, 21)
    values = 2.0 * src + 3.0  # exactly representable by linear interp
    tgt = np.linspace(0.0, 1.0, 7)
    out = ia.remap_axis(values, axis=-1, source_coords=src, target_coords=tgt)
    np.testing.assert_allclose(out, 2.0 * tgt + 3.0, rtol=1e-12)


def test_array_remap_2d_along_first_axis():
    src = np.linspace(0.0, 1.0, 11)
    field = np.outer(src, np.arange(4, dtype=float))  # shape (11, 4)
    tgt = np.array([0.25, 0.5, 0.75])
    out = ia.remap_axis(field, axis=0, source_coords=src, target_coords=tgt)
    assert out.shape == (3, 4)
    np.testing.assert_allclose(out, np.outer(tgt, np.arange(4, dtype=float)))


def test_array_remap_handles_descending_source():
    src = np.linspace(10.0, 0.0, 11)  # descending
    values = src.copy()  # f(x)=x
    tgt = np.array([2.0, 5.0, 8.0])
    out = ia.remap_axis(values, axis=-1, source_coords=src, target_coords=tgt)
    np.testing.assert_allclose(out, tgt, rtol=1e-12)


def test_array_remap_outside_range_returns_nan():
    src = np.linspace(0.0, 1.0, 11)
    values = np.ones_like(src)
    tgt = np.array([-0.5, 0.5, 1.5])
    out = ia.remap_axis(values, axis=-1, source_coords=src, target_coords=tgt)
    assert np.isnan(out[0])
    assert out[1] == 1.0
    assert np.isnan(out[2])


def test_array_remap_nearest_method():
    src = np.array([0.0, 1.0, 2.0, 3.0])
    values = np.array([10.0, 20.0, 30.0, 40.0])
    tgt = np.array([0.4, 1.6, 2.9])
    out = ia.remap_axis(
        values, axis=-1, source_coords=src, target_coords=tgt, method="nearest"
    )
    # 0.4 → 0.0; 1.6 → 2.0; 2.9 → 3.0
    np.testing.assert_array_equal(out, np.array([10.0, 30.0, 40.0]))


def test_array_remap_invalid_method_raises():
    src = np.array([0.0, 1.0])
    with pytest.raises(ValueError):
        ia.remap_axis(
            np.array([1.0, 2.0]),
            axis=-1,
            source_coords=src,
            target_coords=src,
            method="bogus",
        )


def test_array_remap_non_monotonic_source_raises():
    with pytest.raises(ValueError):
        ia.remap_axis(
            np.array([1.0, 2.0, 3.0]),
            axis=-1,
            source_coords=np.array([0.0, 2.0, 1.0]),
            target_coords=np.array([0.5]),
        )


# ---------------------------------------------------------------------------
# Layer 0 — Dataset wrappers
# ---------------------------------------------------------------------------


@pytest.fixture
def ds_profile():
    """Synthetic ocean profile: T(z) = 20 - 0.1 * z, with a horizontal axis."""
    z = np.linspace(0.0, 100.0, 21)
    x = np.arange(4)
    T = 20.0 - 0.1 * z[:, None] + 0.0 * x[None, :]
    return xr.Dataset(
        {"T": (("depth", "x"), T)},
        coords={"depth": z, "x": x},
    )


def test_tier_b_remap_axis_replaces_dim(ds_profile):
    new_z = np.linspace(0.0, 100.0, 11)
    out = remap_axis(
        ds_profile["T"],
        source_dim="depth",
        target_coords=new_z,
    )
    assert "depth" in out.dims
    assert out.sizes["depth"] == 11
    np.testing.assert_allclose(out.values[:, 0], 20.0 - 0.1 * new_z)


def test_tier_b_remap_axis_renames_via_target_name(ds_profile):
    out = remap_axis(
        ds_profile["T"],
        source_dim="depth",
        target_coords=np.linspace(0.0, 100.0, 11),
        target_name="z_new",
    )
    assert "z_new" in out.dims
    assert "depth" not in out.dims


def test_tier_b_remap_axis_uses_dataarray_name(ds_profile):
    target = xr.DataArray(np.linspace(0.0, 100.0, 11), dims=("z2",), name="z2")
    out = remap_axis(ds_profile["T"], source_dim="depth", target_coords=target)
    assert "z2" in out.dims


def test_tier_b_unknown_source_dim_raises(ds_profile):
    with pytest.raises(ValueError):
        remap_axis(
            ds_profile["T"],
            source_dim="bogus",
            target_coords=np.array([0.0, 1.0]),
        )


# ---------------------------------------------------------------------------
# Layer 0 — to_phase
# ---------------------------------------------------------------------------


def test_to_phase_recovers_diurnal_sinusoid():
    """A sinusoid with period 24 sampled over many days should fold cleanly."""
    period = 24.0
    n_days = 30
    n_per_day = 48  # half-hourly
    t = np.arange(n_days * n_per_day) * (period / n_per_day)
    signal = np.sin(2 * np.pi * t / period)
    da = xr.DataArray(signal, dims=("time",), coords={"time": t}, name="x")
    out = to_phase(da, time_dim="time", period=period, n_bins=24)

    assert out.sizes["phase"] == 24
    phases = out["phase"].values
    expected = np.sin(2 * np.pi * phases)
    np.testing.assert_allclose(out.values, expected, atol=0.05)


def test_to_phase_invalid_args_raise():
    da = xr.DataArray(
        np.zeros(10), dims=("time",), coords={"time": np.arange(10)}, name="x"
    )
    with pytest.raises(ValueError):
        to_phase(da, time_dim="time", period=0.0, n_bins=8)
    with pytest.raises(ValueError):
        to_phase(da, time_dim="time", period=1.0, n_bins=0)
    with pytest.raises(ValueError):
        to_phase(da, time_dim="bogus", period=1.0, n_bins=8)


# ---------------------------------------------------------------------------
# Layer 0 → Layer 1 — round-trip identity (ToSigma → FromSigma)
# ---------------------------------------------------------------------------


def test_to_sigma_from_sigma_round_trip():
    """ToSigma then FromSigma on a synthetic profile recovers the original.

    Builds a simple linear depth → sigma mapping and remaps T(depth) →
    T(sigma) → T(depth).
    """
    z = np.linspace(0.0, 100.0, 41)
    T = 20.0 - 0.1 * z + 0.001 * z**2
    ds = xr.Dataset({"T": (("depth",), T)}, coords={"depth": z})

    # Linear mapping: sigma = -depth / H, so sigma in [-1, 0].
    H = 100.0
    sigma_levels = np.linspace(-1.0, 0.0, 51)

    # Forward: build a target that carries depth values at chosen sigma levels.
    # Since sigma = -depth/H, depth at each sigma is -sigma * H.
    depth_at_sigma = -sigma_levels * H
    # Use generic remap_axis: source=depth, target=depth_at_sigma but rename to sigma.
    sig = ToSigma(
        target_axis=xr.DataArray(depth_at_sigma, dims=("sigma",), name="sigma")
    )
    in_sigma = sig(ds)
    assert "sigma" in in_sigma.dims
    assert in_sigma.sizes["sigma"] == 51

    # Reverse: from the sigma-frame Dataset, remap back to a regular depth
    # grid. Source dim is now "sigma" (with values = depth_at_sigma — note
    # that ToSigma puts the *target* values into the new "sigma" coord, so
    # the sigma coord here actually holds depth_at_sigma).
    back = FromSigma(target_axis=z)(in_sigma)
    assert "depth" in back.dims
    # Quadratic profile through two linear interpolations introduces a
    # small interpolation error proportional to the squared step size.
    np.testing.assert_allclose(back["T"].values, T, atol=1e-2)


# ---------------------------------------------------------------------------
# Layer 1 — Operator wrappers
# ---------------------------------------------------------------------------


def test_remap_axis_operator_matches_function(ds_profile):
    new_z = np.linspace(0.0, 100.0, 11)
    op = RemapAxis("depth", new_z)
    np.testing.assert_allclose(
        op(ds_profile)["T"].values,
        remap_axis(ds_profile["T"], source_dim="depth", target_coords=new_z).values,
    )


def test_to_phase_operator_matches_function():
    period = 24.0
    t = np.arange(240) * 0.5
    signal = np.cos(2 * np.pi * t / period)
    ds = xr.Dataset({"x": (("time",), signal)}, coords={"time": t})
    op = ToPhase("time", period=period, n_bins=12)
    np.testing.assert_allclose(
        op(ds)["x"].values,
        to_phase(ds["x"], time_dim="time", period=period, n_bins=12).values,
    )


def test_vertical_presets_default_dim_names():
    """Each preset names the output dim conventionally."""
    z = np.linspace(0.0, 100.0, 11)
    ds = xr.Dataset({"T": (("depth",), z)}, coords={"depth": z})
    target = np.linspace(0.0, 100.0, 5)

    out_sigma = ToSigma(target_axis=target)(ds)
    assert "sigma" in out_sigma.dims

    out_iso = ToIsopycnal(target_axis=target)(ds)
    assert "sigma_theta" in out_iso.dims

    ds_p = xr.Dataset({"T": (("level",), z)}, coords={"level": z})
    out_p = ToPressureLevels(target_axis=target)(ds_p)
    assert "pressure" in out_p.dims

    out_h = ToHeight(target_axis=target)(ds_p)
    assert "height" in out_h.dims


def test_remap_axis_operator_rejects_non_numeric_var_with_source_dim():
    """A non-numeric variable carrying source_dim must raise at the operator boundary.

    After the PR γ primitive flip, the Dataset loop / non-numeric guard
    lives on ``RemapAxis._apply``; the primitive itself is DataArray-only.
    """
    z = np.linspace(0.0, 100.0, 5)
    ds = xr.Dataset(
        {
            "T": (("depth",), 20.0 - 0.1 * z),
            "flag": (("depth",), np.array(["a", "b", "c", "d", "e"])),
        },
        coords={"depth": z},
    )
    op = RemapAxis("depth", np.linspace(0.0, 100.0, 11))
    with pytest.raises(TypeError, match="non-numeric"):
        op(ds)


def test_remap_axis_operator_passes_through_non_dim_non_numeric_var():
    z = np.linspace(0.0, 100.0, 5)
    ds = xr.Dataset(
        {
            "T": (("depth",), 20.0 - 0.1 * z),
            "label": ((), "site_42"),
        },
        coords={"depth": z},
    )
    op = RemapAxis("depth", np.linspace(0.0, 100.0, 11))
    out = op(ds)
    assert str(out["label"].values) == "site_42"


def test_to_phase_primitive_requires_time_coord():
    """DataArray must carry a coordinate named time_dim, not just a dimension."""
    da = xr.DataArray(np.zeros(10), dims=("time",), name="x")  # no time coord
    with pytest.raises(ValueError, match="coordinate named"):
        to_phase(da, time_dim="time", period=1.0, n_bins=8)


def test_to_phase_operator_rejects_non_numeric_var_with_time_dim():
    """Operator boundary handles non-numeric Dataset variables after the flip."""
    t = np.arange(10, dtype=float)
    ds = xr.Dataset(
        {
            "x": (("time",), np.zeros(10)),
            "label": (("time",), np.array(["a"] * 10)),
        },
        coords={"time": t},
    )
    op = ToPhase("time", period=1.0, n_bins=8)
    with pytest.raises(TypeError, match="non-numeric"):
        op(ds)


def test_array_remap_preserves_complex_dtype():
    """Complex inputs must not silently lose their imaginary component."""
    src = np.linspace(0.0, 1.0, 11)
    z = src + 1j * src**2
    tgt = np.array([0.25, 0.5, 0.75])
    out = ia.remap_axis(z, axis=-1, source_coords=src, target_coords=tgt)
    assert np.iscomplexobj(out)
    np.testing.assert_allclose(out.real, tgt, atol=1e-12)
    # Linear interpolation of x^2 incurs O(h^2) discretization error.
    np.testing.assert_allclose(out.imag, tgt**2, atol=5e-3)


def test_array_remap_nan_target_returns_nan_linear():
    src = np.linspace(0.0, 1.0, 5)
    out = ia.remap_axis(
        np.arange(5, dtype=float),
        axis=-1,
        source_coords=src,
        target_coords=np.array([0.5, np.nan, 0.75]),
    )
    assert not np.isnan(out[0])
    assert np.isnan(out[1])
    assert not np.isnan(out[2])


def test_array_remap_nan_target_returns_nan_nearest():
    src = np.linspace(0.0, 1.0, 5)
    out = ia.remap_axis(
        np.arange(5, dtype=float),
        axis=-1,
        source_coords=src,
        target_coords=np.array([0.1, np.nan, 0.9]),
        method="nearest",
    )
    assert np.isnan(out[1])
    assert not np.isnan(out[0])
    assert not np.isnan(out[2])


def test_to_phase_preserves_complex_signal():
    """Complex time series must fold without dropping the imaginary part."""
    period = 24.0
    # Sample much finer than the bin width so the per-bin mean is
    # close to ``exp(i 2π * bin_center)``.
    n_per_period = 240
    n = n_per_period * 30
    t = np.arange(n, dtype=float) * (period / n_per_period)
    z = np.exp(1j * 2 * np.pi * t / period)
    da = xr.DataArray(z, dims=("time",), coords={"time": t}, name="z")
    out = to_phase(da, time_dim="time", period=period, n_bins=24)
    assert np.iscomplexobj(out.values)
    phases = out["phase"].values
    expected = np.exp(1j * 2 * np.pi * phases)
    np.testing.assert_allclose(out.values, expected, atol=0.05)


def test_to_phase_drops_nan_time_samples():
    """Samples with NaN time must not contribute to phase-bin means."""
    period = 1.0
    t = np.array([0.1, 0.2, np.nan, 0.6, 0.7])
    # Bin 0 should average (10, 20); bin 1 (60, 70). The NaN-time sample
    # has value 1000 — if it leaked into a bin, the mean would explode.
    values = np.array([10.0, 20.0, 1000.0, 60.0, 70.0])
    da = xr.DataArray(values, dims=("time",), coords={"time": t}, name="x")
    out = to_phase(da, time_dim="time", period=period, n_bins=2)
    np.testing.assert_allclose(out.values, [15.0, 65.0])


def test_to_phase_accepts_integer_dtype_input():
    """Regression: ``to_phase`` used ``np.isnan`` for the finite check,
    which raises ``TypeError`` on integer arrays. ``np.isfinite`` treats
    integer dtypes as all-finite and is the documented contract."""
    period = 1.0
    t = np.array([0.1, 0.2, 0.6, 0.7])
    values = np.array([10, 20, 60, 70], dtype=np.int64)
    da = xr.DataArray(values, dims=("time",), coords={"time": t}, name="x")
    out = to_phase(da, time_dim="time", period=period, n_bins=2)
    np.testing.assert_allclose(out.values, [15.0, 65.0])


def test_remap_axis_get_config_is_serializable():
    op = RemapAxis("depth", np.array([0.0, 50.0, 100.0]), target_name="z")
    cfg = op.get_config()
    assert cfg == {
        "source_axis": "depth",
        "target_axis": [0.0, 50.0, 100.0],
        "target_name": "z",
        "method": "linear",
    }
