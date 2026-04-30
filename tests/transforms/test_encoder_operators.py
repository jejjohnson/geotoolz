"""Behavioral tests for the Tier-C encoder operators (#95)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xr_toolz.transforms.encoders import (
    cyclical_encode,
    encode_time_cyclical,
    encode_time_ordinal,
    fourier_features,
    positional_encoding,
    random_fourier_features,
    time_rescale,
    time_unrescale,
)
from xr_toolz.transforms.operators import (
    CyclicalEncode,
    EncodeTimeCyclical,
    EncodeTimeOrdinal,
    FourierFeatures,
    PositionalEncoding,
    RandomFourierFeatures,
    TimeRescale,
    TimeUnrescale,
)


# ---------- fixtures ------------------------------------------------------


@pytest.fixture
def ds_scalar() -> xr.Dataset:
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {"angle": (("sample",), rng.uniform(0, 2 * np.pi, size=8))},
        coords={"sample": np.arange(8)},
    )


@pytest.fixture
def ds_time() -> xr.Dataset:
    time = pd.date_range("2020-01-01", periods=10, freq="6h")
    return xr.Dataset(
        {"x": (("time",), np.arange(10, dtype=float))},
        coords={"time": time},
    )


# ---------- parity --------------------------------------------------------


def test_cyclical_encode_parity(ds_scalar: xr.Dataset) -> None:
    op = CyclicalEncode(variable="angle", period=2 * np.pi)
    out = op(ds_scalar)
    sin_e, cos_e = cyclical_encode(ds_scalar["angle"].values, period=2 * np.pi)
    np.testing.assert_allclose(out["angle_sin"].values, sin_e)
    np.testing.assert_allclose(out["angle_cos"].values, cos_e)
    assert "angle" in out  # original preserved


def test_fourier_features_parity(ds_scalar: xr.Dataset) -> None:
    op = FourierFeatures(variable="angle", num_freqs=4, scale=2.0)
    out = op(ds_scalar)
    expected = fourier_features(ds_scalar["angle"].values, num_freqs=4, scale=2.0)
    np.testing.assert_allclose(out["angle_fourier"].values, expected)
    assert out["angle_fourier"].dims == ("sample", "feature")
    assert out.sizes["feature"] == 8


def test_random_fourier_features_parity(ds_scalar: xr.Dataset) -> None:
    op = RandomFourierFeatures(variable="angle", num_features=10, sigma=1.5, seed=42)
    out = op(ds_scalar)
    expected = random_fourier_features(
        ds_scalar["angle"].values, num_features=10, sigma=1.5, seed=42
    )
    np.testing.assert_allclose(out["angle_rff"].values, expected)
    assert out["angle_rff"].dims == ("sample", "feature")


def test_random_fourier_features_rejects_scalar_input() -> None:
    ds = xr.Dataset({"x": ((), np.float64(1.5))})
    op = RandomFourierFeatures(variable="x", num_features=4, seed=0)
    with pytest.raises(ValueError, match="non-scalar"):
        op(ds)


def test_random_fourier_features_replaces_trailing_axis_for_vector_input() -> None:
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {"x": (("sample", "channel"), rng.standard_normal((6, 3)))},
        coords={"sample": np.arange(6), "channel": np.arange(3)},
    )
    op = RandomFourierFeatures(variable="x", num_features=8, sigma=1.0, seed=1)
    out = op(ds)
    expected = random_fourier_features(
        ds["x"].values, num_features=8, sigma=1.0, seed=1
    )
    # Output should be 2-D (sample, feature) — trailing channel axis replaced.
    assert out["x_rff"].dims == ("sample", "feature")
    assert out["x_rff"].shape == (6, 8)
    np.testing.assert_allclose(out["x_rff"].values, expected)


def test_positional_encoding_parity(ds_scalar: xr.Dataset) -> None:
    op = PositionalEncoding(variable="angle", num_freqs=3, include_input=True)
    out = op(ds_scalar)
    expected = positional_encoding(
        ds_scalar["angle"].values, num_freqs=3, include_input=True
    )
    np.testing.assert_allclose(out["angle_posenc"].values, expected)


def test_encode_time_cyclical_parity(ds_time: xr.Dataset) -> None:
    op = EncodeTimeCyclical(components=("hour", "dayofyear"))
    out = op(ds_time)
    expected = encode_time_cyclical(ds_time, components=("hour", "dayofyear"))
    xr.testing.assert_identical(out, expected)


def test_encode_time_ordinal_parity(ds_time: xr.Dataset) -> None:
    op = EncodeTimeOrdinal(unit="h")
    out = op(ds_time)
    expected = encode_time_ordinal(ds_time, unit="h")
    xr.testing.assert_identical(out, expected)


def test_time_rescale_unrescale_round_trip(ds_time: xr.Dataset) -> None:
    rescale = TimeRescale(freq_dt=6.0, freq_unit="h")
    unrescale = TimeUnrescale()
    rescaled = rescale(ds_time)
    np.testing.assert_array_equal(
        time_rescale(ds_time, freq_dt=6.0, freq_unit="h")["time"].values,
        rescaled["time"].values,
    )
    restored = unrescale(rescaled)
    xr.testing.assert_allclose(
        restored["time"].astype("datetime64[ns]").astype(np.int64),
        ds_time["time"].astype("datetime64[ns]").astype(np.int64),
    )
    # And via direct round-trip:
    direct = time_unrescale(time_rescale(ds_time, freq_dt=6.0, freq_unit="h"))
    xr.testing.assert_identical(restored, direct)


# ---------- get_config ----------------------------------------------------


@pytest.mark.parametrize(
    "op",
    [
        CyclicalEncode(variable="x", period=24.0),
        FourierFeatures(variable="x", num_freqs=4),
        RandomFourierFeatures(variable="x", num_features=8, seed=0),
        PositionalEncoding(variable="x", num_freqs=3),
        EncodeTimeCyclical(components=("hour",)),
        EncodeTimeOrdinal(unit="D"),
        TimeRescale(freq_dt=1.0, freq_unit="h"),
        TimeUnrescale(),
    ],
    ids=lambda op: type(op).__name__,
)
def test_get_config_json_round_trips(op) -> None:
    cfg = op.get_config()
    assert json.loads(json.dumps(cfg)) == cfg


def test_time_ops_stringify_datetime64_in_config() -> None:
    t0 = np.datetime64("2020-06-15T00:00:00")
    rescale = TimeRescale(freq_dt=1.0, freq_unit="D", t0=t0)
    ordinal = EncodeTimeOrdinal(reference_date=t0)
    for cfg in (rescale.get_config(), ordinal.get_config()):
        assert json.loads(json.dumps(cfg)) == cfg
