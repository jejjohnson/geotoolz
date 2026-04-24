"""AEMET catalog + top-level catalog integration."""

from __future__ import annotations

from xr_toolz.data import CATALOG, describe
from xr_toolz.data._src.aemet.catalog import AEMET_DATASETS
from xr_toolz.data._src.base import DatasetKind


def test_all_presets_present():
    expected = {
        "aemet_stations",
        "aemet_daily",
        "aemet_hourly",
        "aemet_monthly",
        "aemet_normals",
        "aemet_extremes",
        "aemet_pollution",
    }
    assert expected <= set(AEMET_DATASETS)


def test_presets_are_stations_kind():
    for info in AEMET_DATASETS.values():
        assert info.kind == DatasetKind.STATIONS


def test_presets_registered_in_top_level_catalog():
    names = {k for k in CATALOG if k.startswith("aemet.")}
    assert {
        "aemet.stations",
        "aemet.daily",
        "aemet.hourly",
        "aemet.monthly",
        "aemet.normals",
        "aemet.extremes",
        "aemet.pollution",
    } <= names


def test_describe_aemet_short_name():
    info = describe("aemet.daily")
    assert info.dataset_id == "aemet_daily"
    assert info.source == "aemet"


def test_aemet_kind_extras_dispatch():
    for key, info in AEMET_DATASETS.items():
        assert info.extras.get("aemet_kind") == key


def test_daily_preset_has_expected_variables():
    info = AEMET_DATASETS["aemet_daily"]
    var_names = {v.name for v in info.variables}
    assert {
        "air_temperature_daily_mean",
        "precipitation_amount",
        "wind_speed_daily_mean",
    } <= var_names
