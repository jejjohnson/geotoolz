"""Tests for `XarrayField.time_coord` — the coord-aware temporal helper."""

from __future__ import annotations

import numpy as np
import pytest


xr = pytest.importorskip("xarray")

from geopatcher._src.fields.xarray import XarrayField


def _hourly_da() -> xr.DataArray:
    time = np.arange("2020-01-01", "2020-01-02", dtype="datetime64[h]").astype(
        "datetime64[ns]"
    )
    return xr.DataArray(
        np.arange(time.size, dtype=np.float32),
        dims=("time",),
        coords={"time": time},
    )


def test_returns_numpy_array_of_correct_dtype() -> None:
    da = _hourly_da()
    coord = XarrayField(da).time_coord()
    assert isinstance(coord, np.ndarray)
    assert np.issubdtype(coord.dtype, np.datetime64)
    np.testing.assert_array_equal(coord, da["time"].values)


def test_custom_name_resolves() -> None:
    da = _hourly_da().rename({"time": "valid_time"})
    coord = XarrayField(da).time_coord(name="valid_time")
    np.testing.assert_array_equal(coord, da["valid_time"].values)


def test_missing_coord_raises_keyerror() -> None:
    da = _hourly_da().drop_vars("time")
    with pytest.raises(KeyError, match="has no coord named 'time'"):
        XarrayField(da).time_coord()


def test_object_dtype_raises_typeerror() -> None:
    # Surrogate for cftime: an object-typed coord triggers the typed error
    # without needing a cftime dependency in the test suite.
    da = xr.DataArray(
        np.zeros(3),
        dims=("time",),
        coords={"time": np.array(["a", "b", "c"], dtype=object)},
    )
    with pytest.raises(TypeError, match="cftime"):
        XarrayField(da).time_coord()


def test_grid_domain_bounds_datetime() -> None:
    # Bounds keep the coordinate's own type — no float() coercion.
    lo, hi = XarrayField(_hourly_da()).domain.bounds["time"]
    assert lo == np.datetime64("2020-01-01T00", "ns")
    assert hi == np.datetime64("2020-01-01T23", "ns")


def test_grid_domain_bounds_mixed_dims() -> None:
    da = xr.DataArray(
        np.zeros((2, 3), dtype=np.float32),
        dims=("time", "lat"),
        coords={
            "time": np.array(["2021-06-01", "2021-06-02"], dtype="datetime64[ns]"),
            "lat": np.array([10.0, -5.0, 2.5]),
        },
    )
    bounds = XarrayField(da).domain.bounds
    assert bounds["lat"] == (-5.0, 10.0)
    assert bounds["time"][0] < bounds["time"][1]
