"""Tests for NaN/land-mask-aware differentiation (``nan_policy``)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrgrad import curl, divergence, gradient, laplacian, partial


NLON, NLAT = 41, 21
COAST_DEG = 10.0


def _field() -> tuple[xr.DataArray, np.ndarray, np.ndarray]:
    """``sin(λ)cos(φ)`` plus its land mask and exact ``∂/∂λ`` per degree."""
    lon = np.linspace(0.0, 40.0, NLON)
    lat = np.linspace(0.0, 20.0, NLAT)
    grid_lon, grid_lat = np.meshgrid(lon, lat, indexing="xy")
    values = np.sin(np.deg2rad(grid_lon)) * np.cos(np.deg2rad(grid_lat))
    exact = np.cos(np.deg2rad(grid_lon)) * np.cos(np.deg2rad(grid_lat)) * np.pi / 180.0
    da = xr.DataArray(
        values, dims=("lat", "lon"), coords={"lat": lat, "lon": lon}, name="f"
    )
    land = grid_lon < COAST_DEG
    return da.where(~xr.DataArray(land, dims=da.dims, coords=da.coords)), land, exact


def _first_ocean_column(land: np.ndarray) -> np.ndarray:
    """Boolean mask selecting the ocean cells adjacent to the coast."""
    column = int(land[0].sum())
    sel = np.zeros_like(land)
    sel[:, column] = True
    return sel


# ---------- core behaviour --------------------------------------------------


def test_adaptive_recovers_the_coastal_cells():
    masked, land, _ = _field()
    coast = _first_ocean_column(land)
    propagate = partial(masked, "lon")
    adaptive = partial(masked, "lon", nan_policy="adaptive")
    assert np.isnan(propagate.values[coast]).all()
    assert np.isfinite(adaptive.values[coast]).all()


def test_adaptive_coastal_values_are_accurate():
    """One-sided, so one order down — but the right order of magnitude."""
    masked, land, exact = _field()
    coast = _first_ocean_column(land)
    adaptive = partial(masked, "lon", nan_policy="adaptive", accuracy=2)
    scale = np.abs(exact).mean()
    rel = np.abs(adaptive.values[coast] - exact[coast]) / scale
    assert rel.max() < 1e-2


def test_higher_accuracy_improves_the_coastal_cells():
    """Raising accuracy buys back the order lost to the one-sided stencil."""
    masked, land, exact = _field()
    coast = _first_ocean_column(land)
    errs = [
        np.abs(
            partial(masked, "lon", nan_policy="adaptive", accuracy=a).values[coast]
            - exact[coast]
        ).max()
        for a in (1, 2)
    ]
    assert errs[1] < errs[0]


def test_masked_cells_stay_nan():
    """Land has no data; the answer there is NaN regardless of arithmetic."""
    masked, land, _ = _field()
    adaptive = partial(masked, "lon", nan_policy="adaptive")
    assert np.isnan(adaptive.values[land]).all()


def test_interior_is_bit_identical_to_propagate():
    masked, land, _ = _field()
    lon = masked["lon"].values
    interior = np.broadcast_to(lon[None, :] > COAST_DEG + 5.0, land.shape)
    propagate = partial(masked, "lon")
    adaptive = partial(masked, "lon", nan_policy="adaptive")
    np.testing.assert_array_equal(propagate.values[interior], adaptive.values[interior])


def test_unmasked_field_is_unchanged_by_the_policy():
    """With nothing to recover, adaptive must not perturb the answer."""
    x = np.linspace(0.0, 4.0, 24)
    da = xr.DataArray(np.sin(x), dims=("x",), coords={"x": x}, name="f")
    np.testing.assert_array_equal(
        partial(da, "x").values, partial(da, "x", nan_policy="adaptive").values
    )


def test_default_is_propagate():
    masked, _, _ = _field()
    np.testing.assert_array_equal(
        partial(masked, "lon").values,
        partial(masked, "lon", nan_policy="propagate").values,
    )


# ---------- speckle and narrow strips ---------------------------------------


def test_single_nan_speckle_neighbours_recover():
    x = np.linspace(0.0, 2.0, 21)
    values = x**2
    values[10] = np.nan
    da = xr.DataArray(values, dims=("x",), coords={"x": x}, name="f")
    adaptive = partial(da, "x", nan_policy="adaptive")
    assert np.isnan(adaptive.values[10]), "the speckle itself has no data"
    assert np.isfinite(adaptive.values[9]) and np.isfinite(adaptive.values[11])
    np.testing.assert_allclose(adaptive.values[[9, 11]], (2 * x)[[9, 11]], atol=0.2)


def test_strip_too_narrow_for_any_stencil_stays_nan():
    """A lone valid cell cannot support even a one-sided difference."""
    x = np.linspace(0.0, 4.0, 9)
    values = np.full(9, np.nan)
    values[4] = 1.0
    da = xr.DataArray(values, dims=("x",), coords={"x": x}, name="f")
    assert np.isnan(partial(da, "x", nan_policy="adaptive").values).all()


def _channel(width: int, accuracy: int) -> np.ndarray:
    """``x**2`` on a ``width``-cell channel walled in by NaN."""
    x = np.linspace(0.0, 10.0, 11)
    values = np.full(11, np.nan)
    values[3 : 3 + width] = x[3 : 3 + width] ** 2
    da = xr.DataArray(values, dims=("x",), coords={"x": x}, name="f")
    got = partial(da, "x", nan_policy="adaptive", accuracy=accuracy).values
    return got[3 : 3 + width]


@pytest.mark.parametrize("accuracy", [1, 2, 3])
def test_narrow_channel_falls_back_to_a_lower_accuracy(accuracy):
    """Two cells support a first-order difference and nothing wider."""
    np.testing.assert_allclose(_channel(2, accuracy), 7.0)


@pytest.mark.parametrize("width", [2, 3, 4, 5])
def test_raising_accuracy_never_loses_cells(width):
    """Recovery must be monotone in accuracy, not just error size.

    Without the accuracy descent a channel narrower than the requested
    one-sided stencil vanished precisely when the caller asked for *more*
    accuracy.
    """
    recovered = [int(np.isfinite(_channel(width, a)).sum()) for a in (1, 2, 3)]
    assert recovered == [width, width, width]


def test_wide_stencil_still_wins_where_it_fits():
    """The descent is a fallback: it must not demote a resolvable point."""
    np.testing.assert_allclose(_channel(3, 2), [6.0, 8.0, 10.0])


@pytest.mark.parametrize("accuracy", [1, 2, 3])
def test_axis_shorter_than_the_stencil_descends(accuracy):
    """A short axis must behave like an embedded narrow strip.

    Two samples with ``accuracy=2`` used to raise finitediffx's size
    error, while the *same* two samples embedded in a longer masked axis
    resolved — the outcome hung on array length rather than on the data.
    """
    x = np.array([0.0, 1.0])
    da = xr.DataArray(x**2, dims=("x",), coords={"x": x}, name="f")
    got = partial(da, "x", nan_policy="adaptive", accuracy=accuracy).values
    np.testing.assert_allclose(got, 1.0)


def test_undifferentiable_axis_still_reports_itself():
    """Descending must not swallow an axis that no stencil can handle."""
    da = xr.DataArray([1.0], dims=("x",), coords={"x": [0.0]}, name="f")
    with pytest.raises(ValueError):
        partial(da, "x", nan_policy="adaptive", accuracy=2)


def test_pass_count_stays_within_three_times_accuracy(monkeypatch):
    """The documented upper bound, pinned on a maximally fragmented mask."""
    from xrgrad._src import cartesian

    calls: list[tuple[str, int]] = []
    original = cartesian._difference_maybe_periodic

    def counted(*args, **kwargs):
        calls.append((kwargs["method"], kwargs["accuracy"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(cartesian, "_difference_maybe_periodic", counted)

    accuracy = 3
    x = np.linspace(0.0, 20.0, 21)
    values = x**2
    values[2::3] = np.nan  # two-cell channels, narrower than the stencil
    da = xr.DataArray(values, dims=("x",), coords={"x": x}, name="f")
    partial(da, "x", nan_policy="adaptive", accuracy=accuracy)
    assert len(calls) <= 3 * accuracy
    assert [acc for _, acc in calls] == sorted(
        (acc for _, acc in calls), reverse=True
    ), "accuracies must be tried widest-first"


def test_all_nan_input_stays_all_nan():
    x = np.linspace(0.0, 1.0, 8)
    da = xr.DataArray(np.full(8, np.nan), dims=("x",), coords={"x": x}, name="f")
    assert np.isnan(partial(da, "x", nan_policy="adaptive").values).all()


# ---------- geometry / option interplay -------------------------------------


@pytest.mark.parametrize("geometry", ["cartesian", "rectilinear", "spherical"])
def test_all_geometries_recover_the_coast(geometry):
    masked, land, _ = _field()
    coast = _first_ocean_column(land)
    adaptive = partial(masked, "lon", geometry=geometry, nan_policy="adaptive")
    assert np.isfinite(adaptive.values[coast]).all()
    assert np.isnan(adaptive.values[land]).all()


def test_non_uniform_rectilinear_recovers_the_coast():
    """The chain-rule path differentiates the field, so the policy applies."""
    x = np.array([0.0, 0.5, 1.5, 3.5, 6.0, 9.5, 14.0])
    values = x**2
    values[:2] = np.nan
    da = xr.DataArray(values, dims=("x",), coords={"x": x}, name="f")
    adaptive = partial(da, "x", geometry="rectilinear", nan_policy="adaptive")
    assert np.isnan(adaptive.values[:2]).all()
    assert np.isfinite(adaptive.values[2:]).all()


def test_non_uniform_chain_rule_uses_one_stencil_for_both_terms():
    """``df/di`` and ``dx/di`` must be the same stencil at every point.

    A one-sided numerator over a centred denominator is silently wrong on
    a stretched grid: here it returned ``(3-1)/((3-0)/2) = 4/3`` at index 1
    instead of 1. A linear field pins the quotient exactly.
    """
    x = np.array([0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0])
    values = x.copy()
    values[0] = np.nan
    da = xr.DataArray(values, dims=("x",), coords={"x": x}, name="f")
    adaptive = partial(da, "x", geometry="rectilinear", nan_policy="adaptive")
    np.testing.assert_allclose(adaptive.values[1:], 1.0, atol=1e-12)


def test_non_uniform_interior_matches_propagate():
    """Away from the mask the chain rule must be untouched by the policy."""
    x = np.array([0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0])
    values = x**2
    masked = values.copy()
    masked[0] = np.nan
    coords = {"x": x}
    propagate = partial(
        xr.DataArray(values, dims=("x",), coords=coords, name="f"),
        "x",
        geometry="rectilinear",
    )
    adaptive = partial(
        xr.DataArray(masked, dims=("x",), coords=coords, name="f"),
        "x",
        geometry="rectilinear",
        nan_policy="adaptive",
    )
    np.testing.assert_array_equal(propagate.values[2:], adaptive.values[2:])


def test_clean_field_costs_a_single_stencil_pass(monkeypatch):
    """Nothing to fill, so the forward/backward passes must not run."""
    from xrgrad._src import cartesian

    calls: list[str] = []
    original = cartesian._difference_maybe_periodic

    def counted(*args, **kwargs):
        calls.append(kwargs["method"])
        return original(*args, **kwargs)

    monkeypatch.setattr(cartesian, "_difference_maybe_periodic", counted)

    x = np.linspace(0.0, 4.0, 24)
    da = xr.DataArray(np.sin(x), dims=("x",), coords={"x": x}, name="f")
    partial(da, "x", nan_policy="adaptive")
    assert calls == ["central"]


def test_adaptive_composes_with_periodic():
    """A masked global axis wraps and selects stencils in the same call."""
    lon = np.linspace(0.0, 360.0, 72, endpoint=False)
    lat = np.array([0.0, 10.0, 20.0])
    grid_lon, _ = np.meshgrid(lon, lat, indexing="xy")
    values = np.sin(np.deg2rad(grid_lon))
    values[:, 30:33] = np.nan  # an island away from the seam
    da = xr.DataArray(
        values, dims=("lat", "lon"), coords={"lat": lat, "lon": lon}, name="f"
    )
    adaptive = partial(
        da, "lon", geometry="spherical", periodic=True, nan_policy="adaptive"
    )
    assert np.isnan(adaptive.values[:, 30:33]).all(), "island keeps no data"
    assert np.isfinite(adaptive.values[:, 0]).all(), "seam still wraps"
    assert np.isfinite(adaptive.values[:, 29]).all(), "island edge recovered"


def test_adaptive_with_order_two():
    x = np.linspace(0.0, 5.0, 26)
    values = x**2
    values[:3] = np.nan
    da = xr.DataArray(values, dims=("x",), coords={"x": x}, name="f")
    adaptive = partial(da, "x", order=2, accuracy=2, nan_policy="adaptive")
    np.testing.assert_allclose(adaptive.values[3:], 2.0, atol=1e-6)


def test_float32_input_preserves_dtype():
    """The coord stays float64 — float32 spacing is not uniform to 1e-6."""
    x = np.linspace(0.0, 4.0, 24)
    values = (x**2).astype(np.float32)
    values[:2] = np.nan
    da = xr.DataArray(values, dims=("x",), coords={"x": x}, name="f")
    adaptive = partial(da, "x", nan_policy="adaptive")
    assert adaptive.dtype == np.float32
    assert np.isfinite(adaptive.values[2:]).all()


# ---------- vector operators ------------------------------------------------


def test_vector_operators_accept_the_policy():
    masked, land, _ = _field()
    ds = xr.Dataset({"u": masked, "v": masked})
    cases = {
        "divergence": lambda **kw: divergence(
            ds, ("u", "v"), dims=("lon", "lat"), geometry="spherical", **kw
        ),
        "curl": lambda **kw: curl(
            ds, ("u", "v"), dims=("lon", "lat"), geometry="spherical", **kw
        ),
        "laplacian": lambda **kw: laplacian(masked, dims=("lon", "lat"), **kw),
    }
    for name, call in cases.items():
        lost_propagate = int(np.isnan(call().values).sum())
        lost_adaptive = int(np.isnan(call(nan_policy="adaptive").values).sum())
        assert lost_adaptive < lost_propagate, name
        assert lost_adaptive == int(land.sum()), f"{name}: only the mask should remain"


def test_gradient_accepts_the_policy():
    masked, land, _ = _field()
    grad = gradient(
        masked, dims=("lon", "lat"), geometry="spherical", nan_policy="adaptive"
    )
    for name, component in grad.data_vars.items():
        assert int(np.isnan(component.values).sum()) == int(land.sum()), name


# ---------- validation ------------------------------------------------------


def test_unknown_policy_raises():
    masked, _, _ = _field()
    with pytest.raises(ValueError, match="Unknown nan_policy"):
        partial(masked, "lon", nan_policy="fill-extrapolate")


@pytest.mark.parametrize("method", ["forward", "backward"])
def test_adaptive_with_explicit_method_raises(method):
    """The fallback chain *is* method selection; the two would contradict."""
    masked, _, _ = _field()
    with pytest.raises(ValueError, match="cannot be combined with method"):
        partial(masked, "lon", nan_policy="adaptive", method=method)


def test_propagate_still_allows_an_explicit_method():
    masked, _, _ = _field()
    assert partial(masked, "lon", method="forward").shape == masked.shape
