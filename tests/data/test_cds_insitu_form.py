"""CDS in-situ form construction tests.

Exercise the profile-driven ``_build_form`` path:

- In-situ presets emit ``format=zip``, no ``product_type``, no ``area``.
- Missing ``time_aggregation`` raises a clear ``ValueError``.
- Caller-supplied ``time_aggregation`` / ``usage_restrictions`` /
  ``data_quality`` flow through unchanged.
- ERA5 regression: reanalysis presets still emit ``format=netcdf`` +
  ``product_type`` + ``area``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from xr_toolz.data import (
    INSITU,
    REANALYSIS,
    CDSCredentials,
    CDSFormProfile,
    CDSSource,
)
from xr_toolz.data._src.cds.profiles import resolve_profile
from xr_toolz.types import BBox, TimeRange


class FakeCdsClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], str]] = []

    def retrieve(self, dataset_id: str, form: dict[str, Any], target: str) -> None:
        self.calls.append((dataset_id, form, target))
        Path(target).write_text("stub")


@pytest.fixture
def cds_source() -> tuple[CDSSource, FakeCdsClient]:
    fake = FakeCdsClient()
    src = CDSSource(
        credentials=CDSCredentials(url="https://example", key="abc"),
        client=fake,
    )
    return src, fake


# ---- profile lookup ------------------------------------------------------


def test_resolve_profile_insitu_land():
    assert resolve_profile("insitu-observations-surface-land").family == "insitu-land"


def test_resolve_profile_insitu_marine():
    assert (
        resolve_profile("insitu-observations-surface-marine").family == "insitu-marine"
    )


def test_resolve_profile_reanalysis_default():
    assert resolve_profile("reanalysis-era5-single-levels").family == "reanalysis"


def test_resolve_profile_unknown_falls_back_to_reanalysis():
    assert resolve_profile("totally-made-up-dataset").family == "reanalysis"


# ---- form: in-situ -------------------------------------------------------


def test_insitu_land_form_shape(cds_source, tmp_path):
    src, fake = cds_source
    src.download(
        "insitu-observations-surface-land",
        tmp_path / "obs.zip",
        variables=["air_temperature"],
        time=TimeRange.parse("2020-06-01", "2020-06-30"),
        time_aggregation="daily",
    )
    _, form, _ = fake.calls[0]
    # In-situ uses ``data_format``, not ``format``.
    assert form["data_format"] == "csv"
    assert "format" not in form
    # Fixed profile key.
    assert form["version"] == "2_0_0"
    # No product_type on in-situ.
    assert "product_type" not in form
    assert form["time_aggregation"] == "daily"
    assert form["variable"] == ["air_temperature"]
    # ``year`` is a single string on CDS in-situ, not an array.
    assert form["year"] == "2020"
    assert form["month"] == ["06"]


def test_insitu_marine_form_no_time_aggregation(cds_source, tmp_path):
    """Marine has no ``time_aggregation`` form key."""
    src, fake = cds_source
    src.download(
        "insitu-observations-surface-marine",
        tmp_path / "obs.zip",
        variables=["air_temperature"],
        time=TimeRange.parse("2020-06-01", "2020-06-30"),
    )
    _, form, _ = fake.calls[0]
    assert form["data_format"] == "csv"
    assert form["version"] == "2_0_0"
    assert "time_aggregation" not in form
    assert "product_type" not in form


def test_insitu_bbox_forwards_to_area(cds_source, tmp_path):
    """In-situ accepts server-side ``area`` — bbox must forward."""
    src, fake = cds_source
    src.download(
        "insitu-observations-surface-land",
        tmp_path / "obs.zip",
        variables=["air_temperature"],
        bbox=BBox(-10.0, 40.0, 30.0, 60.0),
        time=TimeRange.parse("2020-01-01", "2020-01-01"),
        time_aggregation="daily",
    )
    _, form, _ = fake.calls[0]
    # [N, W, S, E]
    assert form["area"] == [60.0, -10.0, 30.0, 40.0]


def test_insitu_land_missing_time_aggregation_raises(cds_source, tmp_path):
    """Clear error when the land caller forgets ``time_aggregation``."""
    src, _ = cds_source
    with pytest.raises(ValueError, match="time_aggregation"):
        src.download(
            "insitu-observations-surface-land",
            tmp_path / "obs.zip",
            variables=["air_temperature"],
            time=TimeRange.parse("2020-01-01", "2020-01-01"),
        )


def test_insitu_multi_year_window_raises(cds_source, tmp_path):
    """CDS in-situ rejects multi-year requests — we catch this early."""
    src, _ = cds_source
    with pytest.raises(ValueError, match="one year per request"):
        src.download(
            "insitu-observations-surface-land",
            tmp_path / "obs.zip",
            variables=["air_temperature"],
            time=TimeRange.parse("2019-06-01", "2020-06-30"),
            time_aggregation="daily",
        )


@pytest.mark.parametrize("agg", ["sub_daily", "daily", "monthly"])
def test_insitu_land_all_time_aggregations_supported(cds_source, tmp_path, agg):
    src, fake = cds_source
    src.download(
        "insitu-observations-surface-land",
        tmp_path / "obs.zip",
        variables=["air_temperature"],
        time=TimeRange.parse("2020-01-01", "2020-01-01"),
        time_aggregation=agg,
    )
    _, form, _ = fake.calls[0]
    assert form["time_aggregation"] == agg


def test_insitu_variable_alias_resolution(cds_source, tmp_path):
    """Registered Variable names translate to CDS aliases on the way out."""
    src, fake = cds_source
    src.download(
        "insitu-observations-surface-land",
        tmp_path / "obs.zip",
        variables=["wind_speed", "precipitation_amount"],
        time=TimeRange.parse("2020-01-01", "2020-01-01"),
        time_aggregation="daily",
    )
    _, form, _ = fake.calls[0]
    assert form["variable"] == ["wind_speed", "accumulated_precipitation"]


# ---- form: reanalysis regression ----------------------------------------


def test_reanalysis_regression_single_levels(cds_source, tmp_path):
    """ERA5 single-levels form is unchanged by the profile refactor."""
    src, fake = cds_source
    src.download(
        "reanalysis-era5-single-levels",
        tmp_path / "era5.nc",
        variables=["t2m", "u10"],
        bbox=BBox(-10.0, 40.0, 30.0, 60.0),
        time=TimeRange.parse("2020-01-29", "2020-02-02"),
    )
    _, form, _ = fake.calls[0]
    assert form["variable"] == ["2m_temperature", "10m_u_component_of_wind"]
    assert form["area"] == [60.0, -10.0, 30.0, 40.0]
    assert form["year"] == ["2020"]
    assert set(form["month"]) == {"01", "02"}
    assert form["format"] == "netcdf"
    assert form["product_type"] == "reanalysis"


def test_source_format_override_wins_over_profile(tmp_path):
    """Explicit ``format`` on CDSSource beats the profile default."""
    fake = FakeCdsClient()
    src = CDSSource(
        credentials=CDSCredentials(url="u", key="k"),
        client=fake,
        format="grib",
    )
    src.download(
        "reanalysis-era5-single-levels",
        tmp_path / "x.grib",
        variables=["t2m"],
        time=TimeRange.parse("2020-01-01", "2020-01-01"),
    )
    _, form, _ = fake.calls[0]
    assert form["format"] == "grib"


def test_extras_format_wins_over_source_and_profile(cds_source, tmp_path):
    """Per-call ``format`` in extras beats both source and profile."""
    src, fake = cds_source
    src.download(
        "reanalysis-era5-single-levels",
        tmp_path / "x.nc",
        variables=["t2m"],
        time=TimeRange.parse("2020-01-01", "2020-01-01"),
        format="grib",
    )
    _, form, _ = fake.calls[0]
    assert form["format"] == "grib"


def test_insitu_format_is_csv_by_default(cds_source, tmp_path):
    """Profile default (``data_format=csv``) applies when no override is set."""
    src, fake = cds_source
    assert src.format is None  # default construction
    src.download(
        "insitu-observations-surface-marine",
        tmp_path / "m.zip",
        variables=["air_temperature"],
        time=TimeRange.parse("2020-01-01", "2020-01-01"),
    )
    _, form, _ = fake.calls[0]
    assert form["data_format"] == "csv"


# ---- profile objects -----------------------------------------------------


def test_profile_identity():
    """Sanity check that the module constants are ``CDSFormProfile`` instances."""
    assert isinstance(INSITU, CDSFormProfile)
    assert isinstance(REANALYSIS, CDSFormProfile)
    assert INSITU.format_default == "csv"
    assert INSITU.format_key == "data_format"
    assert REANALYSIS.format_default == "netcdf"
    assert REANALYSIS.format_key == "format"
    # Land needs ``time_aggregation``; marine does not.
    assert "time_aggregation" in INSITU.required_extras
    assert REANALYSIS.includes_product_type is True
    assert INSITU.uses_area is True


def test_open_csv_format_raises_clear_error(cds_source):
    """CSV/zip in-situ outputs aren't xarray-readable; ``open()`` must say so."""
    src, _ = cds_source
    with pytest.raises(ValueError, match="bundle"):
        src.open(
            "insitu-observations-surface-land",
            variables=["air_temperature"],
            time=TimeRange.parse("2020-01-01", "2020-01-01"),
            time_aggregation="daily",
        )


# ---- review follow-ups --------------------------------------------------


def test_open_bails_before_download_for_zip_csv(cds_source):
    """``open()`` must raise *before* hitting ``retrieve`` for zip/csv formats."""
    src, fake = cds_source
    with pytest.raises(ValueError, match="bundle"):
        src.open(
            "insitu-observations-surface-marine",
            variables=["air_temperature"],
            time=TimeRange.parse("2020-01-01", "2020-01-01"),
        )
    # No CDS call should have been made — otherwise we'd be wasting
    # queue time on a guaranteed-failing code path.
    assert fake.calls == []


def test_product_type_rejected_on_profile_without_it(cds_source, tmp_path):
    """In-situ profiles don't have ``product_type``; caller's arg must raise."""
    src, _ = cds_source
    with pytest.raises(ValueError, match="product_type"):
        src.download(
            "insitu-observations-surface-land",
            tmp_path / "obs.zip",
            variables=["air_temperature"],
            time=TimeRange.parse("2020-01-01", "2020-01-01"),
            time_aggregation="daily",
            product_type="reanalysis",
        )


def test_format_extras_maps_to_profile_data_format_key(cds_source, tmp_path):
    """``format=netcdf`` on an in-situ download must rewrite to ``data_format``."""
    src, fake = cds_source
    src.download(
        "insitu-observations-surface-marine",
        tmp_path / "obs.nc",
        variables=["air_temperature"],
        time=TimeRange.parse("2020-01-01", "2020-01-01"),
        format="netcdf",
    )
    _, form, _ = fake.calls[0]
    # The profile's format_key is ``data_format`` — that's what the CDS
    # API actually reads. The alias ``format`` must be translated, not
    # forwarded alongside, otherwise CDS gets two conflicting keys.
    assert form["data_format"] == "netcdf"
    assert "format" not in form


def test_format_none_in_extras_is_ignored(cds_source, tmp_path):
    """``format=None`` must not blank out the form's format key."""
    src, fake = cds_source
    src.download(
        "reanalysis-era5-single-levels",
        tmp_path / "x.nc",
        variables=["t2m"],
        time=TimeRange.parse("2020-01-01", "2020-01-01"),
        format=None,
    )
    _, form, _ = fake.calls[0]
    # Should fall through to the profile default (netcdf) rather than
    # emitting ``form["format"] = None``.
    assert form["format"] == "netcdf"
