"""Regression tests for the third review round on the migration PR."""

from __future__ import annotations

import pandas as pd
import pytest

from xrreader import AEMETCredentials, AemetSource
from xrreader._src.aemet.archive import AemetArchive
from xrreader._src.aemet.source import _normals_to_dataset, _subset_variables
from xrreader.types import AIR_TEMPERATURE_DAILY_MEAN


def _archive(tmp_path):
    return AemetArchive(
        root=tmp_path, source=AemetSource(credentials=AEMETCredentials("k"))
    )


# ---- pollution / sub-daily auto-resume ---------------------------------------


def test_pollution_resume_uses_last_timestamp(tmp_path, monkeypatch):
    arch = _archive(tmp_path)
    df = pd.DataFrame(
        {"time": pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T06:00:00Z"])}
    )
    monkeypatch.setattr(arch, "load", lambda preset: df)
    # Sub-daily preset resumes from the last timestamp itself (not +1 day),
    # so a later same-day sample is not permanently skipped.
    start = pd.Timestamp(arch._resolve_start("aemet_pollution", None))
    assert start == pd.Timestamp("2024-01-01T06:00:00Z")


def test_daily_resume_increments_one_day(tmp_path, monkeypatch):
    arch = _archive(tmp_path)
    df = pd.DataFrame({"time": pd.to_datetime(["2024-01-01", "2024-01-02"])})
    monkeypatch.setattr(arch, "load", lambda preset: df)
    start = pd.Timestamp(arch._resolve_start("aemet_daily", None))
    assert start == pd.Timestamp("2024-01-03")


# ---- normals emit canonical variable names -----------------------------------


def test_normals_emits_canonical_names_and_is_filterable():
    rows = {
        "s1": [
            {
                "mes": "1",
                "tm_mes": "5.0",
                "ta_min": "1.0",
                "ta_max": "9.0",
                "p_mes": "0",
                "inso": "3.0",
            }
        ]
    }
    ds = _normals_to_dataset(rows, ("s1",))
    assert "air_temperature_daily_mean" in ds.data_vars
    assert "tm_mes" not in ds.data_vars
    # Filterable by a catalog Variable instance (canonical name).
    out = _subset_variables(ds, [AIR_TEMPERATURE_DAILY_MEAN])
    assert list(out.data_vars) == ["air_temperature_daily_mean"]


# ---- CDS in-situ QC/metadata columns survive reload --------------------------


def test_cds_long_reload_keeps_qc_metadata():
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    from xrreader._src.cds.archive import _long_to_dataset

    rows = [
        {
            "station_id": "a",
            "time": "2020-01-01",
            "lon": 0.0,
            "lat": 40.0,
            "variable": "ta",
            "value": 1.0,
            "units": "degC",
            "quality_flag": "0",
        },
        {
            "station_id": "a",
            "time": "2020-01-02",
            "lon": 0.0,
            "lat": 40.0,
            "variable": "ta",
            "value": 2.0,
            "units": "degC",
            "quality_flag": "1",
        },
    ]
    df = pd.DataFrame(rows)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(x, y) for x, y in zip(df["lon"], df["lat"], strict=True)],
        crs="EPSG:4326",
    )
    ds = _long_to_dataset(gdf, "cds")
    assert "ta" in ds.data_vars
    assert "ta__units" in ds.data_vars
    assert "ta__quality_flag" in ds.data_vars
    # QC flag round-trips per (station, time).
    assert list(ds["ta__quality_flag"].sel(station="a").values) == ["0", "1"]
