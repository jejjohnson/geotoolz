"""Tests for :mod:`xrtoolz.atm._src.wind`."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrtoolz.atm import wind_components, wind_direction, wind_speed


@pytest.fixture
def ds_uv() -> xr.Dataset:
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {
            "u": (("y", "x"), rng.normal(size=(4, 5))),
            "v": (("y", "x"), rng.normal(size=(4, 5))),
        }
    )


def test_wind_speed_is_hypot(ds_uv):
    out = wind_speed(ds_uv)
    np.testing.assert_allclose(
        out["wind_speed"], np.hypot(ds_uv["u"], ds_uv["v"]), rtol=1e-12
    )
    assert out["wind_speed"].attrs["standard_name"] == "wind_speed"
    assert out["wind_speed"].attrs["units"] == "m s-1"


@pytest.mark.parametrize(
    ("u", "v", "expected_from"),
    [
        (1.0, 0.0, 270.0),  # westerly blows from the west
        (0.0, 1.0, 180.0),  # southerly
        (-1.0, 0.0, 90.0),  # easterly
        (0.0, -1.0, 0.0),  # northerly
        (1.0, 1.0, 225.0),
    ],
)
def test_wind_direction_cardinal_points(u, v, expected_from):
    ds = xr.Dataset({"u": ((), u), "v": ((), v)})
    assert float(wind_direction(ds)["wind_direction"]) == pytest.approx(expected_from)
    assert float(wind_direction(ds, convention="to")["wind_direction"]) == (
        pytest.approx((expected_from + 180.0) % 360.0)
    )


def test_wind_direction_attrs_follow_convention(ds_uv):
    out_from = wind_direction(ds_uv)["wind_direction"]
    out_to = wind_direction(ds_uv, convention="to")["wind_direction"]
    assert out_from.attrs["standard_name"] == "wind_from_direction"
    assert out_to.attrs["standard_name"] == "wind_to_direction"
    assert out_from.attrs["units"] == "degree"
    assert float(out_from.min()) >= 0.0 and float(out_from.max()) < 360.0


@pytest.mark.parametrize("convention", ["from", "to"])
def test_wind_components_round_trip(ds_uv, convention):
    polar = wind_direction(wind_speed(ds_uv), convention=convention)
    back = wind_components(polar.drop_vars(["u", "v"]), convention=convention)
    np.testing.assert_allclose(back["u"], ds_uv["u"], atol=1e-12)
    np.testing.assert_allclose(back["v"], ds_uv["v"], atol=1e-12)
    assert back["u"].attrs["standard_name"] == "eastward_wind"
    assert back["v"].attrs["standard_name"] == "northward_wind"


def test_wind_components_documented_signs():
    # A "from 270°" wind (westerly) is (u, v) = (|V|, 0).
    ds = xr.Dataset({"wind_speed": ((), 3.0), "wind_direction": ((), 270.0)})
    out = wind_components(ds)
    assert float(out["u"]) == pytest.approx(3.0)
    assert float(out["v"]) == pytest.approx(0.0, abs=1e-12)


def test_bad_convention_raises(ds_uv):
    with pytest.raises(ValueError, match="convention"):
        wind_direction(ds_uv, convention="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="convention"):
        wind_components(wind_direction(wind_speed(ds_uv)), convention="bogus")  # type: ignore[arg-type]
