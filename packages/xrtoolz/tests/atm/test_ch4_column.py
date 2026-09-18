"""Tests for :mod:`xrtoolz.atm.gas.ch4._src.column`."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrtoolz.atm._src.constants import M_DRY, N_A, G
from xrtoolz.atm.gas.ch4 import (
    apply_column_averaging_kernel,
    column_mass_to_delta_vmr,
    dry_air_column,
    mixing_ratio_to_column,
)


# ---------- apply_column_averaging_kernel ---------------------------------


@pytest.fixture
def ds_ak() -> xr.Dataset:
    """A 12-layer TROPOMI-style averaging-kernel row (values in ppb).

    Representative of a TROPOMI CH4 L2 pixel (equal-pressure layers, AK
    rising from ~0.6 at the surface to ~1.05 aloft); not copied from a
    specific granule.
    """
    layer = np.arange(12)
    return xr.Dataset(
        {
            "h": ("layer", np.full(12, 1.0 / 12.0)),
            "ak": (
                "layer",
                [
                    0.63,
                    0.72,
                    0.80,
                    0.88,
                    0.95,
                    1.00,
                    1.03,
                    1.05,
                    1.06,
                    1.05,
                    1.02,
                    0.97,
                ],
            ),
            "xa": (
                "layer",
                [
                    1930.0,
                    1925.0,
                    1918.0,
                    1910.0,
                    1900.0,
                    1888.0,
                    1872.0,
                    1850.0,
                    1815.0,
                    1760.0,
                    1680.0,
                    1560.0,
                ],
                {"units": "1e-9"},
            ),
            "x": (
                "layer",
                [
                    1985.0,
                    1960.0,
                    1938.0,
                    1922.0,
                    1908.0,
                    1892.0,
                    1874.0,
                    1851.0,
                    1815.0,
                    1759.0,
                    1679.0,
                    1560.0,
                ],
                {"units": "1e-9"},
            ),
        },
        coords={"layer": layer},
    )


def _smooth(ds: xr.Dataset, **kw) -> xr.DataArray:
    kw = {
        "profile": "x",
        "prior": "xa",
        "averaging_kernel": "ak",
        "pressure_weights": "h",
        **kw,
    }
    return apply_column_averaging_kernel(ds, **kw)["xch4_smoothed"]


def test_identity_kernel_gives_column_average(ds_ak):
    ds = ds_ak.assign(ak=xr.ones_like(ds_ak["ak"]))
    assert float(_smooth(ds)) == pytest.approx(float((ds["h"] * ds["x"]).sum()))


def test_profile_equal_to_prior_gives_prior_column(ds_ak):
    ds = ds_ak.assign(x=ds_ak["xa"])
    assert float(_smooth(ds)) == pytest.approx(float((ds["h"] * ds["xa"]).sum()))


def test_tropomi_style_row_matches_reference_within_1ppb(ds_ak):
    # h·x_a = 1834.0 ppb; the AK-weighted enhancement adds 8.254 ppb.
    out = _smooth(ds_ak)
    assert float(out) == pytest.approx(1842.254, abs=1.0)
    assert "layer" not in out.dims
    assert out.attrs["standard_name"] == "dry_atmosphere_mole_fraction_of_methane"
    assert out.attrs["units"] == "1e-9"


def test_ak_applies_per_pixel(ds_ak):
    ds = ds_ak.expand_dims(pixel=3)
    out = _smooth(ds)
    assert out.dims == ("pixel",)
    np.testing.assert_allclose(out, 1842.2541666666668, rtol=1e-12)


def test_ak_all_nan_pixel_stays_nan(ds_ak):
    # A QA-masked pixel (all layers NaN) must come out NaN, not 0.
    ds = ds_ak.expand_dims(pixel=2).copy(deep=True)
    ds["x"][0, :] = np.nan
    out = _smooth(ds)
    assert np.isnan(float(out[0]))
    assert float(out[1]) == pytest.approx(1842.2541666666668, rel=1e-12)


def test_ak_requires_layer_dim(ds_ak):
    with pytest.raises(ValueError, match="layer"):
        _smooth(ds_ak, level="lev")


# ---------- dry_air_column / mixing_ratio_to_column -----------------------


def test_dry_air_column_reference_value():
    out = dry_air_column(xr.Dataset({"sp": ((), 1.0e5)}))["dry_air_column"]
    # N_A p_s / (g M_dry) for p_s = 1e5 Pa
    assert float(out) == pytest.approx(2.12e29, rel=2e-3)
    assert float(out) == pytest.approx(N_A * 1.0e5 / (G * M_DRY), rel=1e-12)
    assert out.attrs["units"] == "m-2"


def test_dry_air_column_removes_water_vapour():
    ds = xr.Dataset({"sp": ("x", [1.0e5, 9.0e4]), "tcwv": ("x", [20.0, 5.0])})
    dry = dry_air_column(ds)["dry_air_column"]
    wet = dry_air_column(ds, water_vapour_column="tcwv")["dry_air_column"]
    np.testing.assert_allclose(dry - wet, ds["tcwv"] * N_A / M_DRY, rtol=1e-12)


@pytest.fixture
def ds_profile() -> xr.Dataset:
    p = np.linspace(1.0e5, 0.0, 11)  # full column, surface first
    return xr.Dataset(
        {"chi": (("level", "x"), np.full((p.size, 2), 1.9e-6))},
        coords={"level": p},
    ).assign(sp=("x", [1.0e5, 1.0e5]))


def test_uniform_vmr_gives_vmr_times_dry_air_column(ds_profile):
    column = mixing_ratio_to_column(ds_profile, vmr="chi", pressure="level")["column"]
    n_dry = dry_air_column(ds_profile)["dry_air_column"]
    np.testing.assert_allclose(column, 1.9e-6 * n_dry, rtol=1e-12)
    assert column.dims == ("x",)
    # Molecules m^-2, not the CF mol m^-2 mole content: no standard_name.
    assert "standard_name" not in column.attrs
    assert column.attrs["units"] == "m-2"
    assert "molecules per square metre" in column.attrs["long_name"]


def test_mixing_ratio_to_column_is_linear_and_orientation_free(ds_profile):
    one = mixing_ratio_to_column(ds_profile, vmr="chi", pressure="level")["column"]
    two = mixing_ratio_to_column(
        ds_profile.assign(chi=2.0 * ds_profile["chi"]), vmr="chi", pressure="level"
    )["column"]
    np.testing.assert_allclose(two, 2.0 * one, rtol=1e-12)
    flipped = ds_profile.isel(level=slice(None, None, -1))
    np.testing.assert_allclose(
        mixing_ratio_to_column(flipped, vmr="chi", pressure="level")["column"], one
    )


def test_specific_humidity_reduces_column(ds_profile):
    ds = ds_profile.assign(q=xr.full_like(ds_profile["chi"], 0.01))
    dry = mixing_ratio_to_column(ds, vmr="chi", pressure="level")["column"]
    moist = mixing_ratio_to_column(
        ds, vmr="chi", pressure="level", specific_humidity="q"
    )["column"]
    np.testing.assert_allclose(moist, 0.99 * dry, rtol=1e-12)


# ---------- column_mass_to_delta_vmr --------------------------------------


def test_column_mass_to_delta_vmr_matches_plumax_twin():
    # Reference from plumax ``coupled.rtm.column_mass_to_delta_vmr`` with
    # its constants (N_A = 6.02214076e23, k_B = 1.380649e-23,
    # M_CH4 = 0.0160425 kg/mol) for m = 0.01 kg/m², p = 0.9 atm
    # (= 91192.5 Pa), T = 288.15 K, L = 500 m (= 5e4 cm):
    #   n_gas = m N_A / M_CH4 / 1e4            [molecules/cm²]
    #   n_air = p / (k_B T) / 1e6              [molecules/cm³]
    #   ΔVMR  = n_gas / (n_air L_cm)
    reference = 3.2753041450630834e-05
    out = column_mass_to_delta_vmr(
        xr.DataArray(0.01),
        pressure_pa=0.9 * 101325.0,
        temperature_k=288.15,
        path_length_m=500.0,
    )
    assert float(out) == pytest.approx(reference, rel=1e-12)
    assert out.attrs["units"] == "1"


def test_column_mass_to_delta_vmr_broadcasts_layer_parameters():
    mass = xr.DataArray([0.0, 0.01, 0.02], dims="x")
    temperature = xr.DataArray([280.0, 290.0, 300.0], dims="x")
    out = column_mass_to_delta_vmr(
        mass, pressure_pa=1.0e5, temperature_k=temperature, path_length_m=100.0
    )
    assert out.dims == ("x",)
    assert float(out[0]) == 0.0
    # Linear in mass; the per-x temperature enters through n_air ∝ 1/T.
    assert float(out[2]) == pytest.approx(
        2.0 * float(out[1]) * 300.0 / 290.0, rel=1e-12
    )
