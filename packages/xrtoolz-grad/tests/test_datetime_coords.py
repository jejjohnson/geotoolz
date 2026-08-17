"""Tests for time derivatives on datetime64 / timedelta64 coordinates."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrgrad import gradient, partial


def _hourly(n: int = 24) -> xr.DataArray:
    """A smooth signal on a uniform hourly ``datetime64`` axis."""
    time = np.datetime64("2026-01-01T00:00:00") + np.arange(n) * np.timedelta64(1, "h")
    seconds = np.arange(n) * 3600.0
    return xr.DataArray(
        np.sin(seconds / 86400.0),
        dims=("time",),
        coords={"time": time},
        name="f",
    )


def test_datetime_partial_matches_xarray_differentiate():
    """Both are 2nd-order centred on a uniform axis, in per-second units."""
    da = _hourly()
    got = partial(da, "time")
    expected = da.differentiate("time", datetime_unit="s")
    np.testing.assert_allclose(got.values[1:-1], expected.values[1:-1], rtol=1e-9)


def test_datetime_partial_matches_analytic_derivative():
    """Accurate to the 2nd-order truncation error of an hourly stencil.

    With ``h = 3600 s`` on a signal of timescale ``86400 s`` the centred
    stencil carries a relative error of order ``(h/T)² / 6 ≈ 3e-4``, so
    ``rtol=1e-3`` is the tightest bound that is not merely fitting noise.
    """
    da = _hourly()
    seconds = np.arange(da.sizes["time"]) * 3600.0
    exact = np.cos(seconds / 86400.0) / 86400.0
    np.testing.assert_allclose(partial(da, "time").values[1:-1], exact[1:-1], rtol=1e-3)


def test_datetime_derivative_is_per_second():
    """A ramp of 1 unit per hour differentiates to 1/3600 per second."""
    n = 12
    time = np.datetime64("2026-03-01") + np.arange(n) * np.timedelta64(1, "h")
    da = xr.DataArray(np.arange(n, dtype=float), dims=("time",), coords={"time": time})
    np.testing.assert_allclose(partial(da, "time").values, 1.0 / 3600.0, rtol=1e-12)


def test_timedelta_coordinate_works():
    """Lead-time axes are timedelta64, not datetime64."""
    lead = np.timedelta64(1, "h") * np.arange(12)
    da = xr.DataArray(
        np.arange(12, dtype=float), dims=("lead",), coords={"lead": lead}, name="f"
    )
    np.testing.assert_allclose(partial(da, "lead").values, 1.0 / 3600.0, rtol=1e-12)


def test_non_uniform_datetime_routes_through_rectilinear():
    """A skipped day makes the axis non-uniform; rectilinear handles it."""
    days = np.array([0, 1, 2, 4, 5, 6, 7], dtype="timedelta64[D]")
    time = np.datetime64("2026-01-01") + days
    slope = 3.0 / 86400.0  # per second
    values = slope * days.astype("timedelta64[s]").astype(float)
    da = xr.DataArray(values, dims=("time",), coords={"time": time}, name="f")
    got = partial(da, "time", geometry="rectilinear")
    np.testing.assert_allclose(got.values, slope, rtol=1e-9)


def test_uniform_datetime_rejected_by_cartesian_when_non_uniform():
    days = np.array([0, 1, 2, 4], dtype="timedelta64[D]")
    time = np.datetime64("2026-01-01") + days
    da = xr.DataArray(np.arange(4.0), dims=("time",), coords={"time": time}, name="f")
    with pytest.raises(ValueError, match="not uniformly spaced"):
        partial(da, "time", geometry="cartesian")


def test_datetime_gradient_names_follow_convention():
    grad = gradient(_hourly(), dims=("time",))
    assert set(grad.data_vars) == {"df_dtime"}


def test_datetime_coordinate_is_preserved_on_output():
    da = _hourly()
    out = partial(da, "time")
    assert out["time"].dtype == da["time"].dtype
    np.testing.assert_array_equal(out["time"].values, da["time"].values)


def test_nanosecond_resolution_is_not_lost_to_epoch_offset():
    """Anchoring at the first sample keeps small steps representable."""
    time = np.datetime64("2026-01-01T00:00:00") + np.arange(8) * np.timedelta64(1, "ns")
    da = xr.DataArray(np.arange(8.0), dims=("time",), coords={"time": time}, name="f")
    np.testing.assert_allclose(partial(da, "time").values, 1e9, rtol=1e-9)


def test_cftime_object_coordinate_raises_clear_error():
    cftime = pytest.importorskip("cftime")
    time = np.array(
        [cftime.DatetimeNoLeap(2026, 1, d) for d in range(1, 6)], dtype=object
    )
    da = xr.DataArray(np.arange(5.0), dims=("time",), coords={"time": time}, name="f")
    with pytest.raises(ValueError, match="object dtype"):
        partial(da, "time")


def test_numeric_coordinates_are_unaffected():
    x = np.linspace(0.0, 3.0, 8)
    da = xr.DataArray(x**2, dims=("x",), coords={"x": x}, name="f")
    np.testing.assert_allclose(partial(da, "x").values[1:-1], (2 * x)[1:-1], atol=1e-9)


def test_integer_coordinate_still_works():
    idx = np.arange(10)
    da = xr.DataArray(idx.astype(float) ** 2, dims=("i",), coords={"i": idx}, name="f")
    np.testing.assert_allclose(
        partial(da, "i").values[1:-1], (2.0 * idx)[1:-1], atol=1e-9
    )


def test_millennium_span_does_not_overflow_nanoseconds():
    """A 1000-year daily axis exceeds the int64 nanosecond range.

    ``timedelta64[ns]`` saturates at roughly ±292 years, so casting the
    whole span to nanoseconds used to wrap it to a *negative* step and
    silently flip the sign of every derivative.
    """
    steps = np.arange(1000) * 365
    time = np.datetime64("1000-01-01", "D") + steps.astype("timedelta64[D]")
    slope_per_second = 1.0 / (365.0 * 86400.0)
    values = steps.astype(np.float64) * 86400.0 * slope_per_second
    da = xr.DataArray(values, dims=("time",), coords={"time": time}, name="f")
    got = partial(da, "time")
    assert float(got.values[5]) > 0.0, "derivative sign flipped — ns overflow"
    np.testing.assert_allclose(got.values[1:-1], slope_per_second, rtol=1e-9)


@pytest.mark.parametrize("unit", ["s", "m", "h", "D"])
def test_coarse_time_units_convert_exactly(unit):
    """Second/minute/hour/day resolutions all scale from their own unit."""
    per_unit_seconds = {"s": 1.0, "m": 60.0, "h": 3600.0, "D": 86400.0}[unit]
    time = np.datetime64("2026-01-01", unit) + np.arange(10).astype(
        f"timedelta64[{unit}]"
    )
    da = xr.DataArray(
        np.arange(10, dtype=float), dims=("time",), coords={"time": time}, name="f"
    )
    np.testing.assert_allclose(
        partial(da, "time").values, 1.0 / per_unit_seconds, rtol=1e-12
    )


def test_single_gap_wider_than_the_nanosecond_range():
    """One adjacent gap over ~292 years overflows a nanosecond subtraction.

    The axis itself is representable — 1800 and 2200 both fit
    ``datetime64[ns]`` — but their difference does not, so the gap has to
    be measured at a coarser resolution.
    """
    time = np.array(["1800-01-01", "2200-01-01"], dtype="datetime64[ns]")
    da = xr.DataArray([0.0, 1.0], dims=("time",), coords={"time": time}, name="f")
    got = partial(da, "time")
    span_s = 400 * 365.2425 * 86400.0
    assert float(got.values[0]) > 0.0, "derivative sign flipped — ns overflow"
    np.testing.assert_allclose(got.values, 1.0 / span_s, rtol=1e-3)


def test_microsecond_axis_keeps_its_resolution():
    """The safe-gap threshold scales per unit, not off the ns limit."""
    time = np.datetime64("2026-01-01", "us") + np.arange(6).astype("timedelta64[us]")
    da = xr.DataArray(
        np.arange(6, dtype=float), dims=("time",), coords={"time": time}, name="f"
    )
    np.testing.assert_allclose(partial(da, "time").values, 1e6, rtol=1e-12)
