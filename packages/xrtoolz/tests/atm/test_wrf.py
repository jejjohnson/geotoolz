"""Tests for :mod:`xrtoolz.atm._src.wrf`."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrtoolz.atm import (
    destagger,
    open_wrfout,
    wrf_height_agl,
    wrf_temperature,
    wrf_time,
    wrf_wind,
)
from xrtoolz.atm._src.wrf import _DIMS, KAPPA, P_REFERENCE

from .conftest import write_synthetic_wrfout


# ---------- destagger ----------------------------------------------------


def test_destagger_linear_field_is_exact():
    x_stag = 10.0 * (np.arange(7) - 0.5)
    da = xr.DataArray(
        3.0 + 0.25 * x_stag[None, :] + np.arange(2)[:, None],
        dims=("t", "x_stag"),
        coords={"x_stag": x_stag, "t": [0, 1], "flag": ("x_stag", list("abcdefg"))},
        attrs={"units": "m s-1"},
        name="U",
    )
    out = destagger(da, "x_stag")
    x_mass = 10.0 * np.arange(6)
    assert out.dims == ("t", "x")
    assert out.sizes["x"] == da.sizes["x_stag"] - 1
    np.testing.assert_allclose(
        out, 3.0 + 0.25 * x_mass[None, :] + np.arange(2)[:, None], rtol=1e-14
    )
    np.testing.assert_allclose(out["x"], x_mass)  # coordinate averaged
    assert "flag" not in out.coords  # non-numeric coord on the dim is dropped
    assert list(out["t"]) == [0, 1]
    assert out.attrs == {"units": "m s-1"}
    assert out.name == "U"


def test_destagger_explicit_new_dim_and_errors():
    da = xr.DataArray(np.arange(4.0), dims="face")
    out = destagger(da, "face", new_dim="centre")
    assert out.dims == ("centre",)
    np.testing.assert_allclose(out, [0.5, 1.5, 2.5])
    with pytest.raises(ValueError, match="not in DataArray dims"):
        destagger(da, "x_stag")


def test_destagger_stays_lazy():
    pytest.importorskip("dask")
    da = xr.DataArray(np.ones((3, 5)), dims=("t", "x_stag")).chunk({"t": 1})
    out = destagger(da, "x_stag")
    assert out.chunks is not None
    np.testing.assert_allclose(out.compute(), 1.0)


# ---------- wrf_time -----------------------------------------------------


STAMPS = np.array(
    ["2024-06-01T00:00:00", "2024-06-01T01:00:00", "2024-06-01T02:00:00"],
    dtype="datetime64[s]",
)


def _times_rows() -> np.ndarray:
    return np.array([str(t).replace("T", "_").encode() for t in STAMPS], dtype="S19")


def test_wrf_time_from_times_only():
    ds = xr.Dataset({"Times": (("Time",), _times_rows())})
    out = wrf_time(ds)
    assert out.dims == ("Time",)
    assert out.dtype == np.dtype("datetime64[s]")
    np.testing.assert_array_equal(out.values, STAMPS)


def test_wrf_time_from_times_char_matrix():
    rows = np.array(
        [np.frombuffer(str(t).replace("T", "_").encode(), dtype="S1") for t in STAMPS]
    )
    assert rows.shape == (3, 19)
    ds = xr.Dataset({"Times": (("Time", "DateStrLen"), rows)})
    np.testing.assert_array_equal(wrf_time(ds).values, STAMPS)


def test_wrf_time_from_xtime_minutes():
    ds = xr.Dataset(
        {
            "XTIME": (
                ("Time",),
                [0.0, 60.0, 120.0],
                {"units": "minutes since 2024-06-01 00:00:00"},
            )
        },
        attrs={"SIMULATION_START_DATE": "2024-06-01_00:00:00"},
    )
    np.testing.assert_array_equal(wrf_time(ds).values, STAMPS)


def test_wrf_time_from_xtime_datetime64():
    ds = xr.Dataset({"XTIME": (("Time",), STAMPS.astype("datetime64[ns]"))})
    out = wrf_time(ds)
    assert out.dtype == np.dtype("datetime64[s]")
    np.testing.assert_array_equal(out.values, STAMPS)


def test_wrf_time_both_present_agree():
    ds = xr.Dataset(
        {"Times": (("Time",), _times_rows()), "XTIME": (("Time",), [0.0, 60.0, 120.0])},
        attrs={"SIMULATION_START_DATE": "2024-06-01_00:00:00"},
    )
    both = wrf_time(ds)
    np.testing.assert_array_equal(both.values, wrf_time(ds.drop_vars("XTIME")).values)
    np.testing.assert_array_equal(both.values, wrf_time(ds.drop_vars("Times")).values)


def test_wrf_time_errors():
    with pytest.raises(ValueError, match="neither"):
        wrf_time(xr.Dataset({"U": (("Time",), [0.0])}))
    with pytest.raises(ValueError, match="SIMULATION_START_DATE"):
        wrf_time(xr.Dataset({"XTIME": (("Time",), [0.0, 60.0])}))


# ---------- open_wrfout on the synthetic file ------------------------------


def test_open_wrfout_dims_coords_and_attrs(synthetic_wrf):
    ds = open_wrfout(synthetic_wrf.path)
    spec = synthetic_wrf
    assert dict(ds.sizes) == {
        "time": spec.n_time,
        "level": spec.n_lev,
        "y": spec.n_y,
        "x": spec.n_x,
    }
    for name in ("u", "v", "w", "temperature", "pressure"):
        assert ds[name].dims == ("time", "level", "y", "x")
    assert ds["pbl_height"].dims == ("time", "y", "x")
    assert ds["terrain"].dims == ("y", "x")
    assert ds["z_agl"].dims == ("time", "level", "y", "x")
    assert ds["time"].dtype == np.dtype("datetime64[s]")
    np.testing.assert_array_equal(ds["time"].values, spec.times)
    np.testing.assert_allclose(ds["x"], spec.x_mass)
    np.testing.assert_allclose(ds["y"], spec.y_mass)
    np.testing.assert_array_equal(ds["level"], np.arange(spec.n_lev))
    assert ds["lon"].dims == ("y", "x") and ds["lat"].dims == ("y", "x")
    np.testing.assert_allclose(ds["lon"].isel(y=0), -3.0 + 0.01 * np.arange(spec.n_x))
    np.testing.assert_allclose(ds["lat"].isel(x=0), 40.0 + 0.009 * np.arange(spec.n_y))
    # Raw WRF coordinate variables do not leak through.
    assert not {"XLONG", "XLAT", "XTIME", "Times"} & set(ds.variables)
    for key, value in (
        ("DX", 1000.0),
        ("DY", 1000.0),
        ("MAP_PROJ", 1),
        ("CEN_LAT", 40.05),
        ("STAND_LON", -2.945),
    ):
        assert ds.attrs[key] == value
    assert "TRUELAT1" in ds.attrs and "TRUELAT2" in ds.attrs and "CEN_LON" in ds.attrs
    assert "TITLE" not in ds.attrs


def test_open_wrfout_physical_fields(synthetic_wrf):
    ds = open_wrfout(synthetic_wrf.path)
    spec = synthetic_wrf
    tt, zz, yy, xx = spec.grid()
    np.testing.assert_allclose(ds["z_agl"], zz, rtol=1e-12)
    np.testing.assert_allclose(ds["pressure"], P_REFERENCE - 8.0 * zz, rtol=1e-12)
    np.testing.assert_allclose(
        ds["temperature"],
        spec.theta(zz) * ((P_REFERENCE - 8.0 * zz) / P_REFERENCE) ** KAPPA,
        rtol=1e-12,
    )
    np.testing.assert_allclose(ds["u"], spec.u(xx, yy, zz, tt), rtol=1e-12)
    np.testing.assert_allclose(ds["v"], spec.v(xx, yy, zz, tt), rtol=1e-12)
    np.testing.assert_allclose(ds["w"], spec.w(xx, yy, zz, tt), rtol=1e-12)
    np.testing.assert_allclose(
        ds["pbl_height"], spec.pbl_height(xx[:, 0], yy[:, 0], tt[:, 0]), rtol=1e-12
    )
    np.testing.assert_allclose(ds["terrain"], spec.terrain)


def test_open_wrfout_cf_attrs(synthetic_wrf):
    ds = open_wrfout(synthetic_wrf.path)
    expected = {
        "u": ("eastward_wind", "m s-1"),
        "v": ("northward_wind", "m s-1"),
        "w": ("upward_air_velocity", "m s-1"),
        "temperature": ("air_temperature", "K"),
        "pressure": ("air_pressure", "Pa"),
        "pbl_height": ("atmosphere_boundary_layer_thickness", "m"),
        "terrain": ("surface_altitude", "m"),
        "z_agl": ("height", "m"),
        "lon": ("longitude", "degrees_east"),
        "lat": ("latitude", "degrees_north"),
    }
    for name, (standard_name, units) in expected.items():
        assert ds[name].attrs["standard_name"] == standard_name, name
        assert ds[name].attrs["units"] == units, name
        assert "stagger" not in ds[name].attrs
    assert ds["z_agl"].attrs["positive"] == "up"
    assert ds["x"].attrs["units"] == "m" and ds["y"].attrs["units"] == "m"


def test_open_wrfout_times_slice_and_raw_variables(synthetic_wrf):
    ds = open_wrfout(synthetic_wrf.path, times=slice(1, 3), variables=["QVAPOR"])
    spec = synthetic_wrf
    assert ds.sizes["time"] == 2
    np.testing.assert_array_equal(ds["time"].values, spec.times[1:3])
    _, zz, _, _ = spec.grid()
    np.testing.assert_allclose(
        ds["u"].isel(time=0), open_wrfout(spec.path)["u"].isel(time=1)
    )
    assert ds["QVAPOR"].dims == ("time", "level", "y", "x")
    np.testing.assert_allclose(ds["QVAPOR"], spec.qvapor(zz)[1:3], rtol=1e-12)
    with pytest.raises(KeyError):
        open_wrfout(spec.path, variables=["NOPE"])


def test_open_wrfout_variables_accepts_bare_str(synthetic_wrf):
    # A bare name must not be iterated character by character.
    ds = open_wrfout(synthetic_wrf.path, variables="QVAPOR")
    assert "QVAPOR" in ds.data_vars
    assert not {"Q", "V", "A", "P", "O", "R"} & set(ds.data_vars)
    xr.testing.assert_identical(
        ds["QVAPOR"], open_wrfout(synthetic_wrf.path, variables=["QVAPOR"])["QVAPOR"]
    )


def test_open_wrfout_rotates_grid_relative_wind(tmp_path):
    alpha = 0.6  # radians; the fixture writes U / V rotated by -alpha
    spec = write_synthetic_wrfout(tmp_path / "wrfout.nc", alpha=alpha)
    ds = open_wrfout(spec.path)
    tt, zz, yy, xx = spec.grid()
    u_e, v_e = spec.u(xx, yy, zz, tt), spec.v(xx, yy, zz, tt)
    np.testing.assert_allclose(ds["u"], u_e, rtol=1e-12)
    np.testing.assert_allclose(ds["v"], v_e, rtol=1e-12)
    assert ds["u"].attrs["standard_name"] == "eastward_wind"
    assert ds["v"].attrs["standard_name"] == "northward_wind"
    assert "grid_relative" not in ds["u"].attrs
    # The rotation is not a no-op: the raw destaggered components differ.
    raw = xr.open_dataset(spec.path, decode_times=False)
    u_grid = destagger(raw["U"], "west_east_stag")
    assert not np.allclose(u_grid, u_e)
    np.testing.assert_allclose(u_grid, u_e * np.cos(alpha) + v_e * np.sin(alpha))


def test_wrf_wind_helper_on_raw_file(tmp_path):
    spec = write_synthetic_wrfout(tmp_path / "wrfout.nc", alpha=-0.3)
    raw = xr.open_dataset(spec.path, decode_times=False).rename(_DIMS)
    u, v = wrf_wind(raw)
    tt, zz, yy, xx = spec.grid()
    np.testing.assert_allclose(u, spec.u(xx, yy, zz, tt), rtol=1e-12)
    np.testing.assert_allclose(v, spec.v(xx, yy, zz, tt), rtol=1e-12)


def test_open_wrfout_without_cosalpha_keeps_grid_relative_wind(tmp_path):
    spec = write_synthetic_wrfout(tmp_path / "wrfout.nc", with_alpha=False)
    ds = open_wrfout(spec.path)
    tt, zz, yy, xx = spec.grid()
    # alpha == 0 so grid-relative == the analytic wind; values are untouched.
    np.testing.assert_allclose(ds["u"], spec.u(xx, yy, zz, tt), rtol=1e-12)
    np.testing.assert_allclose(ds["v"], spec.v(xx, yy, zz, tt), rtol=1e-12)
    for name, component in (("u", "U"), ("v", "V")):
        attrs = ds[name].attrs
        assert "standard_name" not in attrs, name
        assert attrs["long_name"] == f"grid-relative {component} wind component"
        assert attrs["grid_relative"] == "true"
        assert attrs["units"] == "m s-1"


def test_open_wrfout_moving_nest_keeps_time_axis_on_static_fields(tmp_path):
    spec = write_synthetic_wrfout(tmp_path / "wrfout.nc", moving_nest=True)
    ds = open_wrfout(spec.path)
    for name in ("lon", "lat", "terrain"):
        assert ds[name].dims == ("time", "y", "x"), name
    steps = np.arange(spec.n_time)
    np.testing.assert_allclose(
        ds["lon"].isel(y=0), -3.0 + 0.01 * (np.arange(spec.n_x) + steps[:, None])
    )
    np.testing.assert_allclose(
        ds["lat"].isel(x=0), 40.0 + 0.009 * (np.arange(spec.n_y) + steps[:, None])
    )
    np.testing.assert_allclose(
        ds["terrain"].isel(y=0, x=0), spec.terrain + 10.0 * steps
    )
    # The static file still collapses (default fixture is time-invariant).
    static = open_wrfout(write_synthetic_wrfout(tmp_path / "static.nc").path)
    assert static["lon"].dims == ("y", "x") and static["terrain"].dims == ("y", "x")


def test_open_wrfout_context_manager_closes_file(synthetic_wrf, monkeypatch):
    calls: list[int] = []
    real_open = xr.open_dataset

    def spy(*args, **kwargs):
        ds = real_open(*args, **kwargs)
        inner = ds._close

        def closer():
            calls.append(1)
            inner()

        ds.set_close(closer)
        return ds

    monkeypatch.setattr(xr, "open_dataset", spy)
    with open_wrfout(synthetic_wrf.path) as ds:
        np.testing.assert_allclose(ds["terrain"], synthetic_wrf.terrain)
        assert calls == []
    assert calls == [1]
    ds.close()  # idempotent: xarray clears the callback after the first close
    assert calls == [1]


@pytest.mark.parametrize("hgt_time_axis", [True, False])
def test_open_wrfout_hgt_with_and_without_time_axis(tmp_path, hgt_time_axis):
    spec = write_synthetic_wrfout(tmp_path / "wrfout.nc", hgt_time_axis=hgt_time_axis)
    ds = open_wrfout(spec.path)
    _, zz, _, _ = spec.grid()
    assert ds["terrain"].dims == ("y", "x")
    np.testing.assert_allclose(ds["terrain"], spec.terrain)
    np.testing.assert_allclose(ds["z_agl"], zz, rtol=1e-12)
    np.testing.assert_allclose(
        wrf_height_agl(xr.open_dataset(spec.path, decode_times=False)), zz, rtol=1e-12
    )


@pytest.mark.parametrize(("with_times", "with_xtime"), [(True, False), (False, True)])
def test_open_wrfout_time_axis_from_either_source(tmp_path, with_times, with_xtime):
    spec = write_synthetic_wrfout(
        tmp_path / "wrfout.nc", with_times=with_times, with_xtime=with_xtime
    )
    np.testing.assert_array_equal(open_wrfout(spec.path)["time"].values, spec.times)


def test_wrf_temperature_and_height_helpers_on_raw_file(synthetic_wrf):
    raw = xr.open_dataset(synthetic_wrf.path, decode_times=False)
    _, zz, _, _ = synthetic_wrf.grid()
    np.testing.assert_allclose(
        wrf_temperature(raw), synthetic_wrf.temperature(zz), rtol=1e-12
    )
    np.testing.assert_allclose(wrf_height_agl(raw), zz, rtol=1e-12)


@pytest.mark.dask
def test_open_wrfout_chunks_keep_everything_lazy(synthetic_wrf):
    pytest.importorskip("dask")
    lazy = open_wrfout(synthetic_wrf.path, chunks={"time": 1})
    for name, da in lazy.data_vars.items():
        assert da.chunks is not None, name
        if "time" in da.dims:
            assert da.chunks[da.dims.index("time")] == (1,) * synthetic_wrf.n_time, name
    assert lazy["z_agl"].chunks is not None
    eager = open_wrfout(synthetic_wrf.path)
    xr.testing.assert_allclose(lazy.compute(), eager)
