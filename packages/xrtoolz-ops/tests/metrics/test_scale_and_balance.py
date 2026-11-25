"""Resolved scale and physical scores on diagnostix's kernels (#792, #801)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrops import grad, ocn
from xrops.metrics import (
    GeostrophicImbalance,
    density_inversion_fraction,
    divergence_error,
    geostrophic_imbalance,
    resolved_scale,
)


K = np.array([0.01, 0.02, 0.04, 0.08, 0.16])  # cycles per km


def _score(*values: float) -> xr.DataArray:
    return xr.DataArray(np.array(values), dims="k", coords={"k": K})


# ---------- resolved scale -------------------------------------------------------


def test_the_first_crossing_is_the_resolved_scale() -> None:
    """A later bump was taken for it, and a score above gave −6.25 km (#792)."""
    # 0.7 at 0.02 and 0.3 at 0.04: halfway in the logarithm
    expected = pytest.approx(1.0 / np.sqrt(0.02 * 0.04))
    assert float(resolved_scale(_score(0.9, 0.7, 0.3, 0.1, 0.0), "k")) == expected
    assert float(resolved_scale(_score(0.9, 0.7, 0.3, 0.45, 0.6), "k")) == expected
    assert np.isnan(resolved_scale(_score(0.99, 0.95, 0.9, 0.8, 0.7), "k"))
    below = resolved_scale(_score(0.4, 0.3, 0.2, 0.1, 0.0), "k")
    assert float(below) == pytest.approx(100.0)


def test_resolved_scale_takes_any_order_zero_and_negative_frequencies() -> None:
    score = _score(0.9, 0.7, 0.3, 0.1, 0.0)
    expected = float(resolved_scale(score, "k"))
    mirrored = score[::-1].assign_coords(k=-K[::-1])
    mean = score[:1].assign_coords(k=[0.0])
    two_sided = xr.concat([mirrored, mean, score], dim="k")
    assert float(resolved_scale(two_sided, "k")) == pytest.approx(expected)
    assert float(resolved_scale(score[::-1], "k")) == pytest.approx(expected)
    assert float(resolved_scale(score.to_dataset(name="score"), "k")) == expected


def test_resolved_scale_of_each_score_and_on_dask() -> None:
    scores = xr.concat(
        [_score(0.9, 0.7, 0.3, 0.1, 0.0), _score(0.9, 0.3, 0.2, 0.1, 0.0)], dim="lead"
    )
    out = resolved_scale(scores, "k")
    assert out.dims == ("lead",)
    assert float(out[1]) == pytest.approx(
        float(resolved_scale(scores.isel(lead=1), "k"))
    )
    pytest.importorskip("dask.array")
    lazy = resolved_scale(scores.chunk({"lead": 1, "k": 2}), "k")
    assert lazy.chunks is not None
    xr.testing.assert_allclose(lazy.compute(), out)


# ---------- physical ---------------------------------------------------------------


def _flow() -> xr.Dataset:
    lat, lon = np.arange(20.0, 60.0), np.arange(-40.0, 0.0)
    lam, phi = np.deg2rad(lon)[None, :], np.deg2rad(lat)[:, None]
    ssh = xr.DataArray(
        0.3 * np.sin(6 * lam) * np.cos(5 * phi),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
    )
    ds = ocn.geostrophic_velocities(ssh.to_dataset(name="ssh"))
    return ds.assign(ssh=ssh)


def test_a_geostrophic_flow_has_no_imbalance_and_rest_has_none_to_speak_of() -> None:
    ds = _flow()
    assert float(geostrophic_imbalance(ds)) < 1e-12
    assert np.isnan(geostrophic_imbalance(ds * 0.0))
    # no pressure gradient: all of the Coriolis force is left over
    assert float(geostrophic_imbalance(ds.assign(ssh=0.0 * ds["ssh"]))) == 1.0


def test_each_plane_has_its_own_imbalance_by_area() -> None:
    ds = _flow()
    several = xr.concat([ds, ds.assign(u=1.5 * ds["u"])], dim="time")
    out = geostrophic_imbalance(several)
    assert out.dims == ("time",)
    # f u is half as large again: r_u = f u / 2 of the first
    area = np.cos(np.deg2rad(ds["lat"]))
    f = ocn.coriolis_parameter(ds["lat"])
    mean = lambda x: float((x * area).sum() / (xr.ones_like(x) * area).sum())
    scale = mean((f * ds["v"]) ** 2) + mean((1.5 * f * ds["u"]) ** 2)
    assert float(out[1]) == pytest.approx(
        np.sqrt(mean((0.5 * f * ds["u"]) ** 2) / scale), rel=1e-6
    )
    op = GeostrophicImbalance()
    xr.testing.assert_identical(op(several), out)
    assert "eps" not in op.get_config()


def test_land_is_not_read() -> None:
    ds = _flow()
    wet = xr.ones_like(ds["ssh"], dtype=bool)
    wet[10:15, 20:26] = False
    for score in (
        geostrophic_imbalance,
        lambda d: divergence_error(d, reduce="rms"),
    ):
        out = score(ds.where(wet))
        assert np.isfinite(out)
        xr.testing.assert_allclose(out, score(ds.where(wet, 1e6).where(wet)))
    # the flow on either side of the land is as balanced as before, to truncation
    assert float(geostrophic_imbalance(ds.where(wet))) < 0.05


def test_a_row_at_a_pole_is_left_out() -> None:
    lat, lon = np.linspace(50.0, 90.0, 21), np.arange(0.0, 360.0, 10.0)
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {k: (("lat", "lon"), rng.normal(size=(21, 36))) for k in ("u", "v", "ssh")},
        coords={"lat": lat, "lon": lon},
    )
    assert np.isfinite(divergence_error(ds, reduce="rms"))
    assert np.isfinite(geostrophic_imbalance(ds))


def test_rms_divergence_of_a_radial_flow() -> None:
    x = y = np.linspace(0.0, 9.0, 10)
    X, Y = np.meshgrid(x, y)
    ds = xr.Dataset(
        {"u": (("y", "x"), 2.0 * X), "v": (("y", "x"), 3.0 * Y)},
        coords={"x": x, "y": y},
    )
    plane = {"dims": ("x", "y"), "geometry": "cartesian"}
    out = divergence_error(ds.expand_dims(time=2), reduce="rms", **plane)
    assert out.dims == ("time",)
    np.testing.assert_allclose(out, 5.0)


def test_physical_scores_on_dask_are_lazy_and_equal() -> None:
    pytest.importorskip("dask.array")
    ds = xr.concat([_flow(), 2.0 * _flow()], dim="time")
    chunked = ds.chunk({"time": 1, "lat": 11, "lon": 13})
    for score in (geostrophic_imbalance, lambda d: divergence_error(d, reduce="rms")):
        lazy = score(chunked)
        assert lazy.chunks is not None
        xr.testing.assert_allclose(lazy.compute(), score(ds), atol=1e-12)


def test_inversions_of_each_column_with_land_and_a_shallow_bottom() -> None:
    depth = np.array([0.0, 10.0, 25.0, 50.0, 100.0])
    rho = np.full((5, 3), np.nan)
    rho[:, 0] = [1025.0, 1025.5, 1025.3, 1026.0, 1026.5]
    rho[:3, 1] = [1025.0, 1024.0, 1026.0]
    ds = xr.Dataset({"rho": (("depth", "x"), rho)}, coords={"depth": depth})
    out = density_inversion_fraction(ds)
    np.testing.assert_allclose(out, [0.25, 0.5, np.nan])
    xr.testing.assert_allclose(
        density_inversion_fraction(ds.isel(depth=slice(None, None, -1))), out
    )


def test_gravity_is_that_of_the_caller() -> None:
    ds = _flow()
    assert float(geostrophic_imbalance(ds, g=grad.GRAVITY)) < 1e-12
    assert float(geostrophic_imbalance(ds, g=2.0 * grad.GRAVITY)) > 0.5
