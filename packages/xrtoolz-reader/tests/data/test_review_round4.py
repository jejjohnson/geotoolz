"""Regression tests for the fourth review round on the migration PR."""

from __future__ import annotations

import pytest
import xarray as xr

from xrreader import (
    AEMETCredentials,
    AemetSource,
    BBox,
    CDSCredentials,
    CDSSource,
    CMEMSCredentials,
    CMEMSSource,
)
from xrreader._src.aemet.source import _subset_variables
from xrreader.types import SEA_SURFACE_SALINITY, PressureLevels, TimeRange, resolve


def _cds() -> CDSSource:
    return CDSSource(credentials=CDSCredentials("u", "k"), client=object())


# ---- 1. AEMET raw field-name filters translate through the field map ---------


def test_subset_variables_accepts_raw_aemet_field_names():
    ds = xr.Dataset(
        {
            "air_temperature_daily_mean": ("time", [1.0]),
            "precipitation_amount": ("time", [2.0]),
        },
        coords={"time": [0]},
        attrs={"endpoint": "daily"},
    )
    # "tmed" is AEMET's raw daily field; data var is the canonical name.
    out = _subset_variables(ds, ["tmed"])
    assert list(out.data_vars) == ["air_temperature_daily_mean"]


# ---- 2. catalog-only CMEMS variables are registered --------------------------


@pytest.mark.parametrize("name", ["sst_obs", "analysed_sst", "sea_surface_salinity"])
def test_catalog_variables_registered(name):
    assert resolve(name).name == name


def test_cmems_encodes_registered_catalog_variable():
    src = CMEMSSource(credentials=CMEMSCredentials("u", "p"), client=object())
    # Registered now → translated to the CMEMS alias, not passed raw.
    assert src._encode_variables(["sea_surface_salinity"]) == [
        SEA_SURFACE_SALINITY.for_source("cmems")
    ]


# ---- 3. hourly endpoint honours the time window ------------------------------


def test_open_forwards_time_to_hourly(monkeypatch):
    src = AemetSource(credentials=AEMETCredentials("k"))
    seen: dict[str, object] = {}
    monkeypatch.setattr(src, "_resolve_station_ids", lambda s, b: ("1",))
    monkeypatch.setattr(
        src,
        "get_hourly",
        lambda ids, *, time=None, variables=None: (
            seen.update(time=time) or xr.Dataset()
        ),
    )
    tr = TimeRange.parse("2024-01-01", "2024-01-02")
    src.open("aemet_hourly", stations=["1"], time=tr)
    assert seen["time"] is tr


# ---- 4. ERA5 pressure_level is per-dataset, not per-family --------------------


def test_pressure_level_only_for_pressure_dataset():
    levels = PressureLevels((500, 850))
    single = _cds()._build_form(
        dataset_id="reanalysis-era5-single-levels",
        variables=["t"],
        bbox=None,
        time=None,
        levels=levels,
        extras={},
    )
    assert "pressure_level" not in single
    pressure = _cds()._build_form(
        dataset_id="reanalysis-era5-pressure-levels",
        variables=["t"],
        bbox=None,
        time=None,
        levels=levels,
        extras={},
    )
    assert pressure["pressure_level"] == ["500", "850"]


# ---- 5. antimeridian boxes are rejected by service serializers ---------------


def test_as_cmems_rejects_antimeridian():
    with pytest.raises(ValueError, match="antimeridian"):
        BBox(170.0, -170.0, -10.0, 10.0).as_cmems()


def test_as_cds_area_rejects_antimeridian():
    with pytest.raises(ValueError, match="antimeridian"):
        BBox(170.0, -170.0, -10.0, 10.0).as_cds_area()


def test_as_cmems_normal_box_ok():
    assert BBox(-10.0, 10.0, -5.0, 5.0).as_cmems()["minimum_longitude"] == -10.0
