"""Tests for :mod:`xr_toolz.geo.operators` — Layer-1 wrappers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xr_toolz.core import Graph, Input, Sequential
from xr_toolz.geo.operators import (
    MAE,
    NRMSE,
    PSD,
    RMSE,
    Bias,
    CalculateClimatology,
    Correlation,
    PSDScore,
    R2Score,
    RemoveClimatology,
    ResampleTime,
    SelectVariables,
    SubsetBBox,
    SubsetTime,
    ValidateCoords,
    ValidateLatitude,
    ValidateLongitude,
)


@pytest.fixture
def ds_global() -> xr.Dataset:
    time = pd.date_range("2020-01-01", "2021-12-31", freq="1D")
    lat = np.linspace(-60.0, 60.0, 7)
    lon = np.linspace(-170.0, 170.0, 9)
    rng = np.random.default_rng(11)
    data = rng.standard_normal((len(time), len(lat), len(lon)))
    return xr.Dataset(
        {"ssh": (("time", "lat", "lon"), data)},
        coords={"time": time, "lat": lat, "lon": lon},
    )


def test_validate_coords_applies_both():
    ds = xr.Dataset(coords={"longitude": [200.0], "latitude": [100.0]})
    out = ValidateCoords()(ds)
    assert "lon" in out.coords
    assert "lat" in out.coords


def test_pipeline_validate_subset_select(ds_global):
    ds_global["sst"] = ds_global["ssh"] * 0.5
    pipe = Sequential(
        [
            ValidateCoords(),
            SubsetBBox(lon_bnds=(-30.0, 30.0), lat_bnds=(-20.0, 20.0)),
            SelectVariables("ssh"),
        ]
    )
    out = pipe(ds_global)
    assert list(out.data_vars) == ["ssh"]
    assert float(out.lon.min()) >= -30.0


def test_subset_time_operator(ds_global):
    out = SubsetTime("2020-06-01", "2020-08-31")(ds_global)
    assert pd.Timestamp(out.time.min().values) >= pd.Timestamp("2020-06-01")


def test_climatology_operators_round_trip(ds_global):
    clim_op = CalculateClimatology(freq="day")
    clim = clim_op(ds_global)
    anomaly = RemoveClimatology(clim)(ds_global)
    # anomaly climatology should be ~zero
    residual = CalculateClimatology(freq="day")(anomaly)["ssh"]
    np.testing.assert_allclose(residual.values, 0.0, atol=1e-10)


def test_pipe_syntax_composes_validation():
    ds = xr.Dataset(coords={"longitude": [200.0], "latitude": [100.0]})
    pipe = ValidateLongitude() | ValidateLatitude()
    out = pipe(ds)
    assert "lon" in out.coords and "lat" in out.coords


def test_get_config_round_trip_sequential(ds_global):
    pipe = Sequential(
        [
            ValidateCoords(),
            SubsetBBox((-30.0, 30.0), (-20.0, 20.0)),
        ]
    )
    cfg = pipe.get_config()
    assert cfg["operators"][0]["class"] == "ValidateCoords"
    assert cfg["operators"][1]["class"] == "SubsetBBox"
    # round-trip through repr for readability
    assert "SubsetBBox" in repr(pipe)


def test_pixel_metric_operators_two_inputs(ds_global):
    pred = ds_global
    ref = ds_global.copy()
    assert float(RMSE("ssh", "time")(pred, ref).max()) == pytest.approx(0.0)
    assert float(MAE("ssh", "time")(pred, ref).max()) == pytest.approx(0.0)
    assert float(Bias("ssh", "time")(pred, ref).max()) == pytest.approx(0.0)
    assert float(Correlation("ssh", "time")(pred, ref).min()) == pytest.approx(1.0)
    assert float(NRMSE("ssh", "time")(pred, ref).min()) == pytest.approx(1.0)
    assert float(R2Score("ssh", "time")(pred, ref).min()) == pytest.approx(1.0)


def test_psd_operator(ds_global):
    # 1-D PSD along time is well-defined and fast.
    ds1d = ds_global.isel(lat=3, lon=4)
    out = PSD("ssh", dims=["time"])(ds1d)
    assert "ssh" in out.data_vars


def test_psd_score_operator_perfect_prediction(ds_global):
    ds1d = ds_global.isel(lat=3)
    out = PSDScore("ssh", psd_dims=["time", "lon"])(ds1d, ds1d)
    np.testing.assert_allclose(out["score"].values, np.ones_like(out["score"].values))


def test_resample_operator_monthly(ds_global):
    out = ResampleTime(freq="1ME", method="mean")(ds_global)
    assert out.sizes["time"] == 24


def test_graph_with_rmse_and_bias(ds_global):
    pred = Input("pred")
    ref = Input("ref")
    rmse_node = RMSE("ssh", "time")(pred, ref)
    bias_node = Bias("ssh", "time")(pred, ref)
    graph = Graph(
        inputs={"pred": pred, "ref": ref},
        outputs={"rmse": rmse_node, "bias": bias_node},
    )
    out = graph(pred=ds_global, ref=ds_global)
    assert "rmse" in out and "bias" in out
