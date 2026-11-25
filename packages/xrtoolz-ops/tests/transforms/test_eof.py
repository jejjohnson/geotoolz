"""EOFs: weights, land, and the way back (#798)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xrops.transforms import eof, eof_project, eof_reconstruct


@pytest.fixture
def field() -> xr.DataArray:
    rng = np.random.default_rng(0)
    time = pd.date_range("2020-01-01", periods=24, freq="1ME")
    lat, lon = np.linspace(-60.0, 60.0, 5), np.linspace(-20.0, 20.0, 6)
    return xr.DataArray(
        15.0 + rng.standard_normal((24, 5, 6)),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
        name="sst",
    )


def _area(da: xr.DataArray) -> xr.DataArray:
    return np.cos(np.deg2rad(da["lat"]))


def test_without_weights_it_is_a_principal_component_analysis(field) -> None:
    decomposition = pytest.importorskip("sklearn.decomposition")
    oracle = decomposition.PCA(n_components=3).fit(field.values.reshape(24, -1))
    out = eof(field, n_modes=3)
    modes = out["modes"].values.reshape(3, -1)
    sign = np.sign(np.sum(modes * oracle.components_, axis=-1))[:, None]
    np.testing.assert_allclose(modes, sign * oracle.components_, atol=1e-10)
    np.testing.assert_allclose(
        out["explained_variance"], oracle.explained_variance_ratio_
    )


def test_modes_are_orthonormal_under_the_weights(field) -> None:
    out = eof(field, n_modes=4, weights=_area(field))
    gram = xr.dot(
        out["modes"] * out["weights"],
        out["modes"].rename(mode="other"),
        dim=["lat", "lon"],
    )
    np.testing.assert_allclose(gram, np.eye(4), atol=1e-12)
    assert out["modes"].dims == ("mode", "lat", "lon")
    assert out["principal_components"].dims == ("time", "mode")
    xr.testing.assert_identical(out["lat"], field["lat"])


def test_weights_favour_the_cells_they_weigh(field) -> None:
    polar = field.copy()
    polar[:, 0] += 5.0 * np.random.default_rng(1).standard_normal((24, 6))
    tropics = (abs(field["lat"]) < 50.0).astype(float)
    plain, weighted = eof(polar, n_modes=1), eof(polar, n_modes=1, weights=tropics)
    assert int(abs(plain["modes"]).max("lon").isel(mode=0).argmax("lat")) == 0
    assert np.isnan(weighted["modes"].isel(lat=0)).all()


def test_all_modes_rebuild_the_field_and_project_to_their_amplitudes(field) -> None:
    out = eof(field, n_modes=24, weights=_area(field))
    xr.testing.assert_allclose(eof_reconstruct(out), field)
    xr.testing.assert_allclose(
        eof_project(field, out), out["principal_components"], atol=1e-10
    )


def test_a_mode_set_to_zero_is_left_out(field) -> None:
    out = eof(field, n_modes=3)
    first = out["principal_components"].where(out["mode"] == 0, 0.0)
    expected = out["mean"] + (out["modes"] * out["principal_components"]).isel(mode=0)
    xr.testing.assert_allclose(
        eof_reconstruct(out, first), expected.transpose("time", ...).drop_vars("mode")
    )


def test_land_is_left_out_and_nan(field) -> None:
    wet = xr.ones_like(field.isel(time=0, drop=True), dtype=bool)
    wet[2, 3:] = False
    out = eof(field.where(wet), n_modes=2, weights=_area(field))
    assert (np.isnan(out["modes"]) == ~wet).all()
    np.testing.assert_allclose(
        eof_project(field.where(wet), out), out["principal_components"], atol=1e-10
    )
    rebuilt = eof_reconstruct(out)
    assert (np.isnan(rebuilt) == ~wet).all()


def test_a_gap_raises(field) -> None:
    field[3, 1, 1] = np.nan
    with pytest.raises(ValueError, match="some samples only"):
        eof(field, n_modes=2)


def test_dims_in_any_order_and_dask(field) -> None:
    expected = eof(field, n_modes=2, weights=_area(field))
    turned = eof(field.transpose("lon", "time", "lat"), n_modes=2, weights=_area(field))
    assert turned["modes"].dims == ("mode", "lon", "lat")
    for name, variable in expected.data_vars.items():
        xr.testing.assert_allclose(turned[name].transpose(*variable.dims), variable)
    pytest.importorskip("dask.array")
    chunked = field.chunk({"time": 5, "lat": 2})
    xr.testing.assert_allclose(eof(chunked, n_modes=2, weights=_area(field)), expected)
    lazy = eof_project(chunked, expected)
    assert lazy.chunks is not None
    xr.testing.assert_allclose(
        lazy.compute(), expected["principal_components"], atol=1e-10
    )


def test_a_one_dim_grid_and_another_sample_dim(field) -> None:
    ensemble = field.isel(lon=0).rename(time="member")
    out = eof(ensemble, "member", n_modes=2)
    assert out["modes"].dims == ("mode", "lat")
    xr.testing.assert_allclose(
        eof_project(ensemble, out), out["principal_components"], atol=1e-10
    )


def test_bad_arguments_raise(field) -> None:
    with pytest.raises(ValueError, match="needs 'member'"):
        eof(field, "member", n_modes=1)
    with pytest.raises(ValueError, match="n_modes"):
        eof(field, n_modes=25)
    with pytest.raises(ValueError, match="that the grid has not"):
        eof(field, n_modes=1, weights=field)
