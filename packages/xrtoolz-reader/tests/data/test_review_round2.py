"""Regression tests for the second review round on the migration PR."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
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
    Station,
    TimeRange,
)
from xrreader._src.aemet.source import _chunk_days_range


class _FakeCds:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, str]] = []

    def retrieve(self, dataset_id: str, form: dict, target: str) -> None:
        self.calls.append((dataset_id, dict(form), target))
        xr.Dataset({"marker": ([], 1)}).to_netcdf(target)


# ---- geometry: full-globe normalization --------------------------------------


def test_to_360_preserves_full_globe():
    assert BBox.global_().to_360() == BBox(0.0, 360.0, -90.0, 90.0)
    assert BBox.global_().to_180() == BBox(-180.0, 180.0, -90.0, 90.0)
    # A shifted full-globe box collapses under naive modulo too.
    assert BBox(10.0, 370.0, -10.0, 10.0).to_360() == BBox(0.0, 360.0, -10.0, 10.0)


# ---- station: 0-360 bbox membership ------------------------------------------


def test_station_in_bbox_accepts_0_360_box():
    s = Station(id="x", name="X", lon=-20.0, lat=40.0, source="aemet")
    assert s.in_bbox(BBox(330.0, 350.0, 30.0, 50.0))  # -20 == 340 in [0, 360]
    assert not s.in_bbox(BBox(10.0, 20.0, 30.0, 50.0))


# ---- base: native variable names pass through --------------------------------


def test_encode_variables_passes_through_native_names():
    src = CMEMSSource(credentials=CMEMSCredentials("u", "p"), client=object())
    # Not a registry canonical name -> already-native, pass through.
    assert src._encode_variables(["my_native_var"]) == ["my_native_var"]
    # Canonical name still maps to the CMEMS alias.
    assert src._encode_variables(["sst"]) == ["thetao"]


# ---- AEMET: chunk cap + single-station string --------------------------------


def test_chunk_days_range_stays_within_cap():
    chunks = _chunk_days_range("s", datetime(2024, 1, 1), datetime(2024, 6, 29), 180)
    assert len(chunks) == 2
    _, a, b = chunks[0]
    # Inclusive window must be exactly the cap, not cap + 1.
    assert (b - a).days + 1 == 180


def test_resolve_single_station_string():
    src = AemetSource(credentials=AEMETCredentials("k"))
    assert src._resolve_station_ids("3195", None) == ("3195",)
    assert src._resolve_station_ids(["3195", "3196"], None) == ("3195", "3196")


def test_open_forwards_variables_to_normals_and_pollution(monkeypatch):
    src = AemetSource(credentials=AEMETCredentials("k"))
    seen: dict[str, object] = {}
    monkeypatch.setattr(src, "_resolve_station_ids", lambda s, b: ("1",))
    monkeypatch.setattr(
        src,
        "get_normals",
        lambda ids, *, variables=None: seen.update(normals=variables) or xr.Dataset(),
    )
    monkeypatch.setattr(
        src,
        "get_pollution",
        lambda ids, *, time=None, variables=None: (
            seen.update(poll=variables) or xr.Dataset()
        ),
    )
    src.open("aemet_normals", stations=["1"], variables=["tm_mes"])
    src.open("aemet_pollution", stations=["1"], variables=["no2"])
    assert seen["normals"] == ["tm_mes"]
    assert seen["poll"] == ["no2"]


# ---- CDS: cache key includes product_type ------------------------------------


def test_cds_cache_key_varies_with_product_type(tmp_path, monkeypatch):
    monkeypatch.setenv("XRREADER_CACHE", str(tmp_path))
    f1, f2 = _FakeCds(), _FakeCds()
    s1 = CDSSource(
        credentials=CDSCredentials("u", "k"), client=f1, product_type="reanalysis"
    )
    s2 = CDSSource(
        credentials=CDSCredentials("u", "k"), client=f2, product_type="ensemble_mean"
    )
    sel = dict(variables=["t2m"], time=TimeRange.parse("2020-01-01", "2020-01-01"))
    s1.open("reanalysis-era5-single-levels", **sel)
    s2.open("reanalysis-era5-single-levels", **sel)
    # Distinct product_type -> distinct cache path -> both must fetch.
    assert len(f1.calls) == 1
    assert len(f2.calls) == 1


# ---- CDS archive: long-layout reload keeps gaps + station coords -------------


def test_cds_long_reconstruction_keeps_gaps_and_coords():
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
        },
        {
            "station_id": "b",
            "time": "2020-01-01",
            "lon": 1.0,
            "lat": 41.0,
            "variable": "ta",
            "value": 2.0,
        },
        # 2020-01-03: all stations NaN -> an all-gap timestamp.
        {
            "station_id": "a",
            "time": "2020-01-03",
            "lon": 0.0,
            "lat": 40.0,
            "variable": "ta",
            "value": np.nan,
        },
        {
            "station_id": "b",
            "time": "2020-01-03",
            "lon": 1.0,
            "lat": 41.0,
            "variable": "ta",
            "value": np.nan,
        },
    ]
    df = pd.DataFrame(rows)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(x, y) for x, y in zip(df["lon"], df["lat"], strict=True)],
        crs="EPSG:4326",
    )
    ds = _long_to_dataset(gdf, "cds")
    assert ds.sizes["time"] == 2  # all-NaN timestamp retained
    assert ds["lon"].dims == ("station",)
    assert list(ds["lon"].values) == [0.0, 1.0]
