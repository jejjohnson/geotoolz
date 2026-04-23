"""CMEMS + CDS adapters tested against in-memory client stand-ins."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from xr_toolz.data import (
    CDSCredentials,
    CDSSource,
    CMEMSCredentials,
    CMEMSSource,
)
from xr_toolz.types import BBox, PressureLevels, TimeRange


class FakeCmemsClient:
    """Test double for the ``copernicusmarine`` module."""

    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None

    def subset(self, **kwargs: Any) -> None:
        self.last_kwargs = kwargs
        output = Path(kwargs["output_directory"]) / kwargs["output_filename"]
        output.write_text("stub")

    def open_dataset(self, **kwargs: Any) -> dict[str, Any]:
        self.last_kwargs = kwargs
        return {"opened": True, **kwargs}


class FakeCdsClient:
    """Test double for a ``cdsapi.Client`` instance."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], str]] = []

    def retrieve(self, dataset_id: str, form: dict[str, Any], target: str) -> None:
        self.calls.append((dataset_id, form, target))
        Path(target).write_text("stub")


# ---- CMEMS ---------------------------------------------------------------


@pytest.fixture
def cmems_source() -> tuple[CMEMSSource, FakeCmemsClient]:
    fake = FakeCmemsClient()
    src = CMEMSSource(
        credentials=CMEMSCredentials(username="u", password="p"),
        client=fake,
    )
    return src, fake


def test_cmems_list_and_describe(cmems_source):
    src, _ = cmems_source
    items = src.list_datasets()
    assert items, "expected at least one curated CMEMS dataset"
    info = src.describe(items[0].dataset_id)
    assert info.source == "cmems"


def test_cmems_describe_unknown_id_builds_minimal_info(cmems_source):
    src, _ = cmems_source
    info = src.describe("totally_made_up_product")
    assert info.dataset_id == "totally_made_up_product"
    assert info.source == "cmems"


def test_cmems_download_builds_correct_payload(cmems_source, tmp_path):
    src, fake = cmems_source
    out = src.download(
        "cmems_mod_glo_phy_my_0.083deg_P1D-m",
        tmp_path / "slice.nc",
        variables=["sst", "ssh"],
        bbox=BBox(-10.0, 40.0, 30.0, 60.0),
        time=TimeRange.parse("2020-01-01", "2020-01-31"),
    )
    assert out.exists()
    kw = fake.last_kwargs
    assert kw is not None
    assert kw["dataset_id"] == "cmems_mod_glo_phy_my_0.083deg_P1D-m"
    # Variables must be translated to CMEMS aliases.
    assert kw["variables"] == ["thetao", "zos"]
    # BBox + time serialized into CMEMS keys.
    assert kw["minimum_longitude"] == -10.0
    assert kw["maximum_latitude"] == 60.0
    assert kw["start_datetime"].startswith("2020-01-01")
    # Credentials threaded through.
    assert kw["username"] == "u"
    assert kw["password"] == "p"


def test_cmems_open_returns_client_payload(cmems_source):
    src, _ = cmems_source
    ds = src.open("foo", variables=["sst"])
    assert ds["opened"] is True
    assert ds["variables"] == ["thetao"]


# ---- CDS -----------------------------------------------------------------


@pytest.fixture
def cds_source() -> tuple[CDSSource, FakeCdsClient]:
    fake = FakeCdsClient()
    src = CDSSource(
        credentials=CDSCredentials(url="https://example", key="abc"),
        client=fake,
    )
    return src, fake


def test_cds_list_and_describe(cds_source):
    src, _ = cds_source
    items = src.list_datasets()
    assert any(i.dataset_id == "reanalysis-era5-single-levels" for i in items)


def test_cds_download_form_shape(cds_source, tmp_path):
    src, fake = cds_source
    out = src.download(
        "reanalysis-era5-single-levels",
        tmp_path / "era5.nc",
        variables=["t2m", "u10"],
        bbox=BBox(-10.0, 40.0, 30.0, 60.0),
        time=TimeRange.parse("2020-01-29", "2020-02-02"),
    )
    assert out.exists()
    (dataset_id, form, target) = fake.calls[0]
    assert dataset_id == "reanalysis-era5-single-levels"
    assert target == str(out)
    # Variables should be translated.
    assert form["variable"] == ["2m_temperature", "10m_u_component_of_wind"]
    # BBox in CDS [N,W,S,E] order.
    assert form["area"] == [60.0, -10.0, 30.0, 40.0]
    # Time exploded to year/month/day lists.
    assert form["year"] == ["2020"]
    assert set(form["month"]) == {"01", "02"}
    # Defaults applied.
    assert form["format"] == "netcdf"
    assert form["product_type"] == "reanalysis"


def test_cds_pressure_levels_round_trip(cds_source, tmp_path):
    src, fake = cds_source
    src.download(
        "reanalysis-era5-pressure-levels",
        tmp_path / "plev.nc",
        variables=["t2m"],
        time=TimeRange.parse("2020-01-01", "2020-01-01"),
        levels=PressureLevels((500, 850, 1000)),
    )
    _, form, _ = fake.calls[0]
    assert form["pressure_level"] == ["500", "850", "1000"]


def test_cds_extras_passthrough(cds_source, tmp_path):
    src, fake = cds_source
    src.download(
        "reanalysis-era5-single-levels",
        tmp_path / "x.nc",
        variables=["t2m"],
        time=TimeRange.parse("2020-01-01", "2020-01-01"),
        grid=[0.25, 0.25],
        product_type="ensemble_mean",
    )
    _, form, _ = fake.calls[0]
    assert form["grid"] == [0.25, 0.25]
    assert form["product_type"] == "ensemble_mean"
