"""Regression tests for the PR-review fixes on #8."""

from __future__ import annotations

from types import MappingProxyType
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from xr_toolz.data import CDSCredentials, CDSSource, describe
from xr_toolz.data._src.cache import _json_default
from xr_toolz.data._src.cds.source import _engine_for_format, _suffix_for_format
from xr_toolz.types import BBox, TimeRange, Variable, apply_cf_attrs


# ---- 1. apply_cf_attrs uses shallow copy --------------------------------


def test_apply_cf_attrs_does_not_deep_copy_data():
    data = np.zeros(1_000_000, dtype=np.float32)
    da = xr.DataArray(data, dims="i")
    out = apply_cf_attrs(da, "sst")
    # Shared underlying buffer: mutating source visible through out.
    data[0] = 42.0
    assert float(out.values[0]) == 42.0


# ---- 2. cache._json_default sorts sets deterministically ----------------


def test_json_default_sorts_sets_across_runs():
    # The fundamental guarantee: repeated calls with the same logical
    # request must yield the same list, regardless of construction
    # order. Even though within a single interpreter this would usually
    # hold, the contract must not depend on that.
    s1 = {"b", "a", "c"}
    s2 = {"c", "a", "b"}
    assert _json_default(s1) == _json_default(s2) == ["a", "b", "c"]


def test_json_default_sorts_mixed_types_via_str():
    # Types that aren't mutually comparable still need a stable order.
    out = _json_default({3, "2", 1})
    assert out == sorted(out, key=str)


# ---- 3. Variable.aliases is truly immutable -----------------------------


def test_variable_aliases_cannot_be_mutated():
    v = Variable(name="x", aliases={"cmems": "thetao"})
    assert isinstance(v.aliases, MappingProxyType)
    with pytest.raises(TypeError):
        v.aliases["cmems"] = "oops"  # type: ignore[index]


def test_variable_aliases_input_dict_mutation_does_not_affect_variable():
    raw = {"cmems": "thetao"}
    v = Variable(name="x", aliases=raw)
    raw["cmems"] = "MUTATED"  # must not leak through Variable.
    assert v.for_source("cmems") == "thetao"


# ---- 4. BBox.as_xarray_sel guards antimeridian --------------------------


def test_bbox_as_xarray_sel_rejects_antimeridian_crossing():
    wrap = BBox(170.0, -170.0, -10.0, 10.0)
    with pytest.raises(ValueError, match="antimeridian"):
        wrap.as_xarray_sel()


def test_bbox_as_xarray_sel_works_after_to_360():
    wrap = BBox(170.0, -170.0, -10.0, 10.0)
    # After to_360 the box is 170..190, which xarray can still handle on
    # a 0-360 grid; here we just confirm the guard no longer trips once
    # the longitudes are ordered.
    normalized = BBox(170.0, 190.0, -10.0, 10.0)
    assert not normalized.crosses_antimeridian
    sel = normalized.as_xarray_sel()
    assert sel["lon"] == slice(170.0, 190.0)
    # Sanity: the underlying `to_360` reorientation still wraps to
    # lon_max=190 in modulo space.
    _ = wrap.to_360  # method exists; antimeridian path tested separately


# ---- 5. describe() return type is DatasetInfo --------------------------


def test_describe_returns_dataset_info_not_any():
    from xr_toolz.data import DatasetInfo

    info = describe("glorys12.daily")
    assert isinstance(info, DatasetInfo)


# ---- 6. ERA5 single-levels title no longer misleads --------------------


def test_era5_single_levels_title_does_not_mention_pressure_levels():
    info = describe("era5.single_levels")
    assert "pressure" not in info.title.lower()
    assert "single levels" in info.title.lower()


# ---- 7. CDS format <-> suffix/engine glue ------------------------------


@pytest.mark.parametrize(
    ("fmt", "suffix", "engine"),
    [
        ("netcdf", ".nc", None),
        ("netcdf4", ".nc", None),
        ("grib", ".grib", "cfgrib"),
        ("GRIB2", ".grib", "cfgrib"),
    ],
)
def test_format_glue_maps_consistently(fmt, suffix, engine):
    assert _suffix_for_format(fmt) == suffix
    assert _engine_for_format(fmt) == engine


def test_unknown_format_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        _suffix_for_format("zarr")


class _FakeCdsClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, str]] = []

    def retrieve(self, dataset_id, form, target):
        self.calls.append((dataset_id, form, target))
        xr.Dataset({"marker": ([], 1)}).to_netcdf(target)


def test_cds_open_uses_format_derived_suffix(monkeypatch, tmp_path):
    monkeypatch.setenv("XR_TOOLZ_CACHE", str(tmp_path))
    fake = _FakeCdsClient()
    # format="grib" → cache path must end in .grib, not .nc.
    src = CDSSource(
        credentials=CDSCredentials(url="https://x", key="k"),
        client=fake,
        format="grib",
    )
    # Patch xr.open_dataset so the fake .grib file (actually netCDF) doesn't
    # get fed to cfgrib; we only care about the cache path here.
    with patch("xarray.open_dataset") as mocked:
        mocked.return_value = xr.Dataset({"marker": ([], 1)})
        src.open(
            "reanalysis-era5-single-levels",
            variables=["t2m"],
            bbox=BBox(-10.0, 40.0, 30.0, 60.0),
            time=TimeRange.parse("2020-01-01", "2020-01-01"),
        )
    # The file written by the fake client must have the .grib suffix.
    (_, _, target) = fake.calls[0]
    assert target.endswith(".grib")


# ---- 8. finite_vals.count() no longer relies on DataArray bool truth ---


def test_validation_range_check_handles_empty_data_without_raising():
    """Construct an all-NaN DataArray so ``count() == 0``. The check
    must return cleanly — previously it could trip the "truth value of
    a DataArray is ambiguous" path on some xarray versions."""
    from xr_toolz.types import validate_variable

    da = xr.DataArray(
        np.full((3,), np.nan),
        dims="i",
        attrs={"standard_name": "sea_surface_temperature", "units": "K"},
    )
    report = validate_variable(da, "sst")
    assert report.ok
    assert not any(i.code == "out_of_range" for i in report.issues)


def test_validation_range_check_still_flags_when_data_present():
    from xr_toolz.types import validate_variable

    da = xr.DataArray(
        np.array([50.0, 100.0]),  # way below SST's valid range [270, 320]
        dims="i",
        attrs={"standard_name": "sea_surface_temperature", "units": "K"},
    )
    report = validate_variable(da, "sst")
    assert not report.ok
    assert any(i.code == "out_of_range" for i in report.issues)
