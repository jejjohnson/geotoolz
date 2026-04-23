"""Variable registry + CF metadata basics."""

from __future__ import annotations

import dataclasses

import pytest

from xr_toolz.types import REGISTRY, SST, T2M, Variable, register, resolve


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
