"""End-to-end flows: short-name -> adapter -> fake client, and CDS caching."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import xarray as xr

from xr_toolz.data import (
    CATALOG,
    CDSCredentials,
    CDSSource,
    CMEMSCredentials,
    CMEMSSource,
    DataSource,
    describe,
)
from xr_toolz.types import BBox, TimeRange


# ---- Fakes shared across the end-to-end suite ---------------------------


class FakeCmemsClient:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None

    def subset(self, **kwargs: Any) -> None:
        self.last_kwargs = kwargs
        (Path(kwargs["output_directory"]) / kwargs["output_filename"]).write_text(
            "stub"
        )

    def open_dataset(self, **kwargs: Any) -> xr.Dataset:
        self.last_kwargs = kwargs
        return xr.Dataset({"marker": ([], 1)})


class FakeCdsClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], str]] = []

    def retrieve(self, dataset_id: str, form: dict[str, Any], target: str) -> None:
        self.calls.append((dataset_id, form, target))
        # Write a minimal NetCDF file so xarray.open_dataset works downstream.
        xr.Dataset({"marker": ([], 1)}).to_netcdf(target)


def _source_for(entry_source: str) -> DataSource:
    if entry_source == "cmems":
        return CMEMSSource(
            credentials=CMEMSCredentials(username="u", password="p"),
            client=FakeCmemsClient(),
        )
    if entry_source == "cds":
        return CDSSource(
            credentials=CDSCredentials(url="https://x", key="k"),
            client=FakeCdsClient(),
        )
    raise AssertionError(f"unknown source {entry_source!r}")


# ---- Short-name -> adapter dispatch -------------------------------------


@pytest.mark.parametrize(
    "short_name",
    [
        "glorys12.daily",
        "duacs.sla",
        "ostia.sst",
        "odyssea.sst",
        "multiobs.sss.daily",
        "globcolour.chl.monthly",
        "era5.single_levels",
    ],
)
def test_short_name_resolves_and_describe_agrees(short_name):
    entry = CATALOG[short_name]
    info = describe(short_name)
    assert info.dataset_id == entry.dataset_id
    assert info.source == entry.source


def test_download_via_short_name_for_cmems(tmp_path):
    entry = CATALOG["glorys12.daily"]
    src = _source_for(entry.source)
    assert isinstance(src, CMEMSSource)
    out = src.download(
        entry.dataset_id,
        tmp_path / "x.nc",
        variables=["sst", "ssh"],
        bbox=BBox(-10.0, 40.0, 30.0, 60.0),
        time=TimeRange.parse("2020-01-01", "2020-01-10"),
    )
    assert out.exists()
    # Variables were resolved via the registry and translated to CMEMS aliases.
    kw = src._client.last_kwargs  # type: ignore[union-attr]
    assert kw is not None
    assert kw["variables"] == ["thetao", "zos"]


def test_download_via_short_name_for_cds(tmp_path):
    entry = CATALOG["era5.single_levels"]
    src = _source_for(entry.source)
    assert isinstance(src, CDSSource)
    out = src.download(
        entry.dataset_id,
        tmp_path / "era5.nc",
        variables=["t2m"],
        bbox=BBox(-10.0, 40.0, 30.0, 60.0),
        time=TimeRange.parse("2020-01-01", "2020-01-02"),
    )
    assert out.exists()


# ---- CDS .open() caches results ----------------------------------------


def test_cds_open_populates_cache_on_miss(monkeypatch, tmp_path):
    monkeypatch.setenv("XR_TOOLZ_CACHE", str(tmp_path))
    fake = FakeCdsClient()
    src = CDSSource(
        credentials=CDSCredentials(url="https://x", key="k"),
        client=fake,
    )
    ds = src.open(
        "reanalysis-era5-single-levels",
        variables=["t2m"],
        bbox=BBox(-10.0, 40.0, 30.0, 60.0),
        time=TimeRange.parse("2020-01-01", "2020-01-01"),
    )
    assert "marker" in ds.data_vars
    # One retrieve call populated the cache.
    assert len(fake.calls) == 1


def test_cds_open_reuses_cache_on_second_call(monkeypatch, tmp_path):
    monkeypatch.setenv("XR_TOOLZ_CACHE", str(tmp_path))
    fake = FakeCdsClient()
    src = CDSSource(
        credentials=CDSCredentials(url="https://x", key="k"),
        client=fake,
    )
    kwargs = {
        "variables": ["t2m"],
        "bbox": BBox(-10.0, 40.0, 30.0, 60.0),
        "time": TimeRange.parse("2020-01-01", "2020-01-01"),
    }
    ds1 = src.open("reanalysis-era5-single-levels", **kwargs)
    ds2 = src.open("reanalysis-era5-single-levels", **kwargs)
    assert "marker" in ds1.data_vars
    assert "marker" in ds2.data_vars
    # Second call must be served from disk — retrieve fires exactly once.
    assert len(fake.calls) == 1


def test_cmems_open_passes_credentials_when_absent(tmp_path):
    # No explicit credentials, no env vars, no ~/.cmems file in a sandboxed HOME.
    fake = FakeCmemsClient()
    src = CMEMSSource(credentials=None, client=fake)
    src.open("cmems_mod_glo_phy_my_0.083deg_P1D-m", variables=["sst"])
    kw = fake.last_kwargs
    assert kw is not None
    # _auth_kwargs returns {} when credentials is None → no username/password keys.
    assert "username" not in kw
    assert "password" not in kw
