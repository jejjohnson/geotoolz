"""Variable registry + CF metadata basics."""

from __future__ import annotations

import dataclasses

import pytest

from xrreader.types import REGISTRY, SST, T2M, Variable, register, resolve


def test_variable_is_frozen_and_hashable():
    v1 = Variable(name="foo", units="m")
    v2 = Variable(name="foo", units="m")
    assert v1 == v2
    assert hash(v1) == hash(v2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        v1.name = "bar"  # type: ignore[misc]


def test_variable_for_source_aliases():
    assert SST.for_source("cmems") == "thetao"
    assert SST.for_source("cds") == "sea_surface_temperature"
    assert SST.for_source("unknown") == "sst"  # falls back to canonical name


def test_variable_cf_attrs_round_trip():
    attrs = SST.cf_attrs()
    assert attrs["standard_name"] == "sea_surface_temperature"
    assert attrs["units"] == "K"
    assert attrs["long_name"].startswith("Sea surface")


def test_resolve_returns_registry_entry_for_known_name():
    assert resolve("sst") is SST
    assert resolve("t2m") is T2M


def test_resolve_returns_variable_unchanged():
    v = Variable(name="x", units="m")
    assert resolve(v) is v


def test_resolve_raises_with_helpful_message():
    with pytest.raises(KeyError, match="Unknown variable"):
        resolve("not_a_real_var")


def test_register_inserts_into_registry():
    v = Variable(name="my_custom_var", units="kg", standard_name="custom_stuff")
    try:
        register(v)
        assert resolve("my_custom_var") is v
    finally:
        del REGISTRY["my_custom_var"]


def test_registry_covers_key_ocn_and_atm_vars():
    for n in ["sst", "ssh", "sla", "uo", "vo", "so", "t2m", "u10", "v10", "msl"]:
        assert n in REGISTRY


def test_registry_covers_ocean_colour_and_bgc_vars():
    for n in [
        # Altimetry derivatives
        "adt",
        "ugos",
        "vgos",
        # Salinity companions
        "sos",
        "dens",
        # Sea ice
        "ice_conc",
        # Ocean colour
        "chl",
        "kd490",
        "zsd",
        "spm",
        "bbp443",
        "pp",
        # Remote-sensing reflectance wavelengths
        "rrs412",
        "rrs443",
        "rrs490",
        "rrs510",
        "rrs555",
        "rrs670",
        # Biogeochemistry
        "no3",
        "po4",
        "si",
        "o2",
        "phyc",
        "zooc",
        "ph",
        "spco2",
    ]:
        assert n in REGISTRY, f"{n} missing from REGISTRY"


def test_rrs_wavelengths_share_standard_name_and_units():
    attrs = [resolve(f"rrs{wl}") for wl in (412, 443, 490, 510, 555, 670)]
    assert len({v.standard_name for v in attrs}) == 1
    assert len({v.units for v in attrs}) == 1
    # But long_names and aliases differ (per-wavelength metadata).
    assert len({v.long_name for v in attrs}) == len(attrs)


def test_chl_has_cf_standard_name():
    chl = resolve("chl")
    assert chl.standard_name == "mass_concentration_of_chlorophyll_a_in_sea_water"
    assert chl.for_source("cmems") == "CHL"


# ---- ERA5 boundary layer / pressure levels + methane L2 (gh-298) -----------


ERA5_PRESSURE_LEVEL_NAMES = ["u", "v", "w", "t", "z", "q"]
METHANE_NAMES = [
    "xch4",
    "xch4_bias_corrected",
    "xch4_precision",
    "ch4_enhancement",
    "column_averaging_kernel",
    "ch4_profile_apriori",
    "dry_air_subcolumns",
    "qa_value",
    "ch4_column",
]


@pytest.mark.parametrize("name", ["blh", *ERA5_PRESSURE_LEVEL_NAMES, *METHANE_NAMES])
def test_era5_and_methane_entries_resolve_with_units(name):
    var = resolve(name)
    attrs = var.cf_attrs()
    assert var.name == name
    assert attrs["units"] == var.units
    if var.standard_name is not None:
        assert attrs["standard_name"] == var.standard_name


def test_era5_pressure_level_cds_aliases():
    assert resolve("u").for_source("cds") == "u_component_of_wind"
    assert resolve("v").for_source("cds") == "v_component_of_wind"
    assert resolve("w").for_source("cds") == "vertical_velocity"
    assert resolve("blh").for_source("cds") == "boundary_layer_height"
    # WRF raw names ride along for the wrfout opener.
    assert resolve("u").for_source("wrf") == "U"
    assert resolve("blh").for_source("wrf") == "PBLH"
    # QVAPOR is a dry-air mixing ratio, not specific humidity: no wrf alias.
    assert "wrf" not in resolve("q").aliases


def test_methane_tropomi_aliases():
    assert resolve("xch4").for_source("tropomi") == "methane_mixing_ratio"
    assert (
        resolve("xch4_bias_corrected").for_source("tropomi")
        == "methane_mixing_ratio_bias_corrected"
    )
    assert resolve("ch4_profile_apriori").for_source("tropomi") == (
        "methane_profile_apriori"
    )
    assert resolve("ch4_enhancement").for_source("emit") == "ch4_enhancement"
    assert resolve("xch4").standard_name == "dry_atmosphere_mole_fraction_of_methane"
    assert resolve("ch4_column").standard_name == "atmosphere_mass_content_of_methane"
