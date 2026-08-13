"""Regression tests for the review fixes folded into the migration.

Each test pins one of the seven behaviours flagged on the migration PR so
the fixes don't silently regress.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xrreader import CDSCredentials, CDSSource
from xrreader._src.aemet.archive import _geodataframe_to_dataset
from xrreader._src.aemet.source import _subset_variables
from xrreader._src.credentials import load_cmems
from xrreader.types import BBox, TimeRange


class _FakeCds:
    """Records retrieve() calls and writes a NetCDF stub (or a given ds)."""

    def __init__(self, ds: xr.Dataset | None = None) -> None:
        self.calls: list[tuple[str, dict, str]] = []
        self._ds = ds

    def retrieve(self, dataset_id: str, form: dict, target: str) -> None:
        self.calls.append((dataset_id, dict(form), target))
        ds = self._ds if self._ds is not None else xr.Dataset({"marker": ([], 1)})
        ds.to_netcdf(target)


def _cds(ds: xr.Dataset | None = None) -> CDSSource:
    return CDSSource(
        credentials=CDSCredentials(url="https://x", key="k"), client=_FakeCds(ds)
    )


# ---- 1 + 2. ERA5 hourly `time` field -----------------------------------------


def test_era5_form_includes_time_hours():
    form = _cds()._build_form(
        dataset_id="reanalysis-era5-single-levels",
        variables=["t2m"],
        bbox=BBox(-10.0, 40.0, 30.0, 60.0),
        time=TimeRange.parse("2020-01-01", "2020-01-02"),
        levels=None,
        extras={},
    )
    # Hourly product: a request without `time` is rejected by CDS.
    assert form["time"] == ["00:00"]


def test_era5_form_time_follows_freq():
    form = _cds()._build_form(
        dataset_id="reanalysis-era5-single-levels",
        variables=["t2m"],
        bbox=None,
        time=TimeRange.parse("2020-01-01", "2020-01-01T18:00", freq="6h"),
        levels=None,
        extras={},
    )
    assert form["time"] == ["00:00", "06:00", "12:00", "18:00"]


def test_time_range_cds_times():
    assert TimeRange.parse("2020-01-01", "2020-01-03").cds_times() == ["00:00"]
    assert TimeRange.parse("2020-01-01", "2020-01-01T12:00", freq="6h").cds_times() == [
        "00:00",
        "06:00",
        "12:00",
    ]


# ---- cartesian over-fetch: warn + trim ---------------------------------------


def test_partial_range_warns_overfetch(tmp_path, monkeypatch):
    monkeypatch.setenv("XRREADER_CACHE", str(tmp_path))
    with pytest.warns(UserWarning, match="cartesian superset"):
        _cds().download(
            "reanalysis-era5-single-levels",
            tmp_path / "x.nc",
            variables=["t2m"],
            time=TimeRange.parse("2020-01-29", "2020-02-02"),
        )


def test_full_month_does_not_warn(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _cds().download(
            "reanalysis-era5-single-levels",
            tmp_path / "x.nc",
            variables=["t2m"],
            time=TimeRange.parse("2020-01-01", "2020-01-31"),
        )


def test_open_trims_to_requested_window(tmp_path, monkeypatch):
    monkeypatch.setenv("XRREADER_CACHE", str(tmp_path))
    times = pd.date_range("2020-01-01", "2020-01-05", freq="1D")
    full = xr.Dataset({"t2m": ("time", np.arange(len(times)))}, coords={"time": times})
    out = _cds(full).open(
        "reanalysis-era5-single-levels",
        variables=["t2m"],
        time=TimeRange.parse("2020-01-02", "2020-01-04"),
    )
    # CDS over-fetched all 5 days; open() trims back to the 3-day window.
    assert list(pd.to_datetime(out["time"].values)) == list(times[1:4])


# ---- 3 covered in tests/data/test_cds_insitu_archive.py (needs the fixture) --


# ---- 4. CMEMS reads the documented .env --------------------------------------


def test_cmems_reads_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("COPERNICUSMARINE_SERVICE_USERNAME", raising=False)
    monkeypatch.delenv("COPERNICUSMARINE_SERVICE_PASSWORD", raising=False)
    (tmp_path / ".env").write_text(
        "COPERNICUSMARINE_SERVICE_USERNAME=alice\n"
        "COPERNICUSMARINE_SERVICE_PASSWORD=secret\n"
    )
    monkeypatch.chdir(tmp_path)
    creds = load_cmems()
    assert creds is not None
    assert (creds.username, creds.password) == ("alice", "secret")


# ---- 5. AEMET variable filter rejects unmatched ------------------------------


def test_aemet_subset_unmatched_variable_raises():
    ds = xr.Dataset({"tmed": ("time", [1.0, 2.0])}, coords={"time": [0, 1]})
    with pytest.raises(ValueError, match="none of the requested"):
        _subset_variables(ds, ["does_not_exist"])


def test_aemet_subset_keeps_matching_variables():
    ds = xr.Dataset(
        {"tmed": ("time", [1.0]), "prec": ("time", [2.0])}, coords={"time": [0]}
    )
    out = _subset_variables(ds, ["tmed"])
    assert list(out.data_vars) == ["tmed"]


# ---- 6 + 7. archive reload keeps all-NaN gaps and station coords -------------


def test_archive_reload_preserves_gaps_and_station_coords():
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    rows = [
        {
            "station_id": "s1",
            "time": "2020-01-01",
            "lon": 0.0,
            "lat": 40.0,
            "tmed": 10.0,
        },
        {
            "station_id": "s2",
            "time": "2020-01-01",
            "lon": 1.0,
            "lat": 41.0,
            "tmed": 12.0,
        },
        # 2020-01-03: every station NaN -> an all-gap timestamp the writer keeps.
        {
            "station_id": "s1",
            "time": "2020-01-03",
            "lon": 0.0,
            "lat": 40.0,
            "tmed": np.nan,
        },
        {
            "station_id": "s2",
            "time": "2020-01-03",
            "lon": 1.0,
            "lat": 41.0,
            "tmed": np.nan,
        },
    ]
    df = pd.DataFrame(rows)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(x, y) for x, y in zip(df["lon"], df["lat"], strict=True)],
        crs="EPSG:4326",
    )
    ds = _geodataframe_to_dataset(gdf)

    # The all-NaN timestamp survives (dropna=False) -> gap structure intact.
    assert ds.sizes["time"] == 2
    assert ds.sizes["station"] == 2
    # Per-station coordinates are restored from the GeoParquet lon/lat.
    assert ds["lon"].dims == ("station",)
    assert ds["lat"].dims == ("station",)
    by_station = dict(zip(ds["station"].values, ds["lon"].values, strict=True))
    assert by_station["s1"] == 0.0
    assert by_station["s2"] == 1.0
