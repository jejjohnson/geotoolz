"""Regressions for the review round on the xrtoolz-reader migration PR.

Covers, in order:

- ``time_aggregation`` participates in the land archive's scope
  fingerprint (and is ignored for marine, which doesn't send it).
- ``CDSInsituArchive`` forces ``data_format="csv"`` even when the
  supplied ``CDSSource`` was configured for NetCDF.
- ``validate_variable`` works on dask-backed arrays.
- ``BBox.as_xarray_sel`` handles a descending latitude index.
- ``TimeRange`` is UTC-aware however it is constructed.
- ``REGISTRY`` is importable from the flat top-level namespace.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xrreader import BBox, CDSCredentials, CDSInsituArchive, CDSSource, TimeRange
from xrreader.types import resolve, validate_variable


# ---- helpers -------------------------------------------------------------


def _csv_zip(path: Path, year: str) -> None:
    rows = [
        {
            "station_id": "STN-A",
            "date_time": f"{year}-01-15T00:00:00Z",
            "longitude": 5.0,
            "latitude": 45.0,
            "observed_variable": "air_temperature",
            "observation_value": 15.0,
            "units": "degC",
            "quality_flag": "0",
        }
    ]
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data.csv", buf.getvalue())


class _RecordingClient:
    """Minimal cdsapi stand-in that records each submitted form."""

    def __init__(self) -> None:
        self.forms: list[dict[str, Any]] = []

    def retrieve(self, dataset_id: str, form: dict[str, Any], target: str) -> None:
        self.forms.append(form)
        _csv_zip(Path(target), form["year"])


def _archive(tmp_path, preset, aggregation, *, source_format=None):
    client = _RecordingClient()
    source = CDSSource(
        credentials=CDSCredentials(url="u", key="k"),
        client=client,
        **({"format": source_format} if source_format else {}),
    )
    archive = CDSInsituArchive(
        root=tmp_path,
        preset=preset,
        source=source,
        time_aggregation=aggregation,
    )
    return archive, client


# ---- time_aggregation in the archive identity ----------------------------


def test_land_archive_refuses_reuse_under_a_different_aggregation(tmp_path):
    # Same root + preset, daily then monthly: the second archive must not
    # inherit the first's completed_chunks and serve daily rows as monthly.
    daily, _ = _archive(tmp_path, "cds_insitu_land", "daily")
    daily.sync("2020-01-01", "2020-12-31")

    monthly, _ = _archive(tmp_path, "cds_insitu_land", "monthly")
    with pytest.raises(ValueError, match="scope"):
        monthly.sync("2020-01-01", "2020-12-31")


def test_marine_archive_ignores_aggregation_in_scope(tmp_path):
    # Marine never sends time_aggregation, so it must not fragment the
    # archive identity — otherwise identical archives stop being reusable.
    first, _ = _archive(tmp_path, "cds_insitu_marine", "daily")
    first.sync("2020-01-01", "2020-12-31")

    second, client = _archive(tmp_path, "cds_insitu_marine", "monthly")
    second.sync("2020-01-01", "2020-12-31")
    # Year already complete under the shared scope — nothing re-fetched.
    assert client.forms == []


# ---- in-situ archive forces CSV ------------------------------------------


def test_insitu_archive_forces_csv_over_a_netcdf_source(tmp_path):
    # A source-level format override outranks the profile default, which
    # would write NetCDF bytes to the temp .zip and raise BadZipFile.
    archive, client = _archive(
        tmp_path, "cds_insitu_land", "daily", source_format="netcdf"
    )
    archive.sync("2020-01-01", "2020-12-31")

    assert client.forms, "expected at least one submitted form"
    assert all(f["data_format"] == "csv" for f in client.forms)


# ---- dask-backed validation ----------------------------------------------


def test_validate_variable_accepts_dask_backed_arrays():
    dask_array = pytest.importorskip("dask.array")
    var = resolve("ssh")
    values = np.full((4, 5), 0.1)
    values[0, 0] = np.nan

    eager = validate_variable(xr.DataArray(values, dims=("y", "x")), var)
    lazy = validate_variable(
        xr.DataArray(dask_array.from_array(values, chunks=2), dims=("y", "x")), var
    )
    assert [i.code for i in lazy.issues] == [i.code for i in eager.issues]


def test_validate_variable_flags_out_of_range_when_dask_backed():
    dask_array = pytest.importorskip("dask.array")
    var = resolve("ssh")  # valid_range (-5.0, 5.0)
    out_of_range = xr.DataArray(
        dask_array.from_array(np.full((3, 3), 999.0), chunks=2), dims=("y", "x")
    )
    report = validate_variable(out_of_range, var)
    assert "out_of_range" in {i.code for i in report.issues}


# ---- descending latitude -------------------------------------------------


def test_as_xarray_sel_handles_descending_latitude():
    # ERA5 and most CDS reanalysis output run north-to-south.
    lat = np.arange(60.0, 20.0, -1.0)
    lon = np.arange(-80.0, -40.0, 1.0)
    ds = xr.Dataset(
        {"v": (("latitude", "longitude"), np.zeros((lat.size, lon.size)))},
        coords={"latitude": lat, "longitude": lon},
    )
    bbox = BBox(lon_min=-70, lon_max=-50, lat_min=30, lat_max=45)

    ascending_sel = bbox.as_xarray_sel(lat="latitude", lon="longitude")
    assert ds.sel(**ascending_sel).sizes["latitude"] == 0

    sel = bbox.as_xarray_sel(lat="latitude", lon="longitude", lat_descending=True)
    picked = ds.sel(**sel)
    assert picked.sizes["latitude"] > 0
    assert float(picked["latitude"].min()) >= 30
    assert float(picked["latitude"].max()) <= 45


# ---- TimeRange UTC invariant ---------------------------------------------


def test_timerange_is_utc_aware_however_constructed():
    direct = TimeRange(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-31"))
    parsed = TimeRange.parse("2020-01-01", "2020-01-31")
    assert direct.start.tz is not None
    assert direct == parsed
    # The paths that broke on naive input (AEMET hourly) now work.
    assert direct.start.tz_convert("Europe/Madrid") is not None


def test_timerange_normalizes_a_non_utc_aware_input():
    madrid = TimeRange(
        pd.Timestamp("2020-01-01", tz="Europe/Madrid"),
        pd.Timestamp("2020-01-31", tz="Europe/Madrid"),
    )
    assert str(madrid.start.tz) == "UTC"


def test_timerange_still_rejects_inverted_windows():
    with pytest.raises(ValueError, match="must be <="):
        TimeRange(pd.Timestamp("2020-02-01"), pd.Timestamp("2020-01-01"))


# ---- flat namespace ------------------------------------------------------


def test_registry_is_importable_from_the_top_level():
    from xrreader import REGISTRY

    assert "ssh" in REGISTRY
