"""CF validators."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xr_toolz.types import (
    Severity,
    Variable,
    apply_cf_attrs,
    validate_dataset,
    validate_variable,
)


def _sst_da(values, attrs=None):
    da = xr.DataArray(np.asarray(values, dtype=float), dims="i")
    if attrs:
        da.attrs.update(attrs)
    return da


def test_validate_variable_all_green():
    da = _sst_da(
        [280.0, 290.0, 300.0],
        attrs={"standard_name": "sea_surface_temperature", "units": "K"},
    )
    r = validate_variable(da, "sst")
    assert r.ok
    assert r.issues == []


def test_validate_variable_missing_attrs_are_warnings():
    da = _sst_da([280.0, 290.0])
    r = validate_variable(da, "sst")
    assert r.ok
    codes = {i.code for i in r.warnings()}
    assert "missing_standard_name" in codes
    assert "missing_units" in codes


def test_validate_variable_wrong_standard_name_is_error():
    da = _sst_da(
        [280.0, 290.0], attrs={"standard_name": "air_temperature", "units": "K"}
    )
    r = validate_variable(da, "sst")
    assert not r.ok
    codes = {i.code for i in r.errors()}
    assert "wrong_standard_name" in codes


def test_validate_variable_wrong_units_is_error():
    da = _sst_da(
        [280.0, 290.0],
        attrs={"standard_name": "sea_surface_temperature", "units": "degC"},
    )
    r = validate_variable(da, "sst")
    assert not r.ok
    codes = {i.code for i in r.errors()}
    assert "wrong_units" in codes


def test_validate_variable_out_of_range_is_error():
    da = _sst_da(
        [50.0, 100.0],
        attrs={"standard_name": "sea_surface_temperature", "units": "K"},
    )
    r = validate_variable(da, "sst")
    assert not r.ok
    codes = {i.code for i in r.errors()}
    assert "out_of_range" in codes


def test_validate_variable_ignores_range_when_disabled():
    da = _sst_da(
        [50.0, 100.0],
        attrs={"standard_name": "sea_surface_temperature", "units": "K"},
    )
    r = validate_variable(da, "sst", check_range=False)
    assert r.ok


def test_validate_dataset_missing_variable():
    ds = xr.Dataset({"sst": _sst_da([280.0], attrs={"units": "K"})})
    r = validate_dataset(ds, ["sst", "t2m"], check_range=False)
    codes = {(i.variable, i.code) for i in r.errors()}
    assert ("t2m", "missing_variable") in codes


def test_raise_if_errors_raises_only_on_errors():
    da = _sst_da([280.0])  # missing attrs → warnings only
    r = validate_variable(da, "sst")
    r.raise_if_errors()  # should not raise

    bad_attrs = {"standard_name": "sea_surface_temperature", "units": "K"}
    bad = _sst_da([50.0], attrs=bad_attrs)
    r = validate_variable(bad, "sst")
    with pytest.raises(ValueError, match="Validation failed"):
        r.raise_if_errors()


def test_apply_cf_attrs_no_overwrite_by_default():
    da = _sst_da([280.0], attrs={"units": "degC"})
    out = apply_cf_attrs(da, "sst")
    assert out.attrs["units"] == "degC"  # preserved
    assert out.attrs["standard_name"] == "sea_surface_temperature"


def test_apply_cf_attrs_overwrite():
    da = _sst_da([280.0], attrs={"units": "degC"})
    out = apply_cf_attrs(da, "sst", overwrite=True)
    assert out.attrs["units"] == "K"


def test_check_dtype_opt_in():
    ok_attrs = {"units": "K", "standard_name": "sea_surface_temperature"}
    da = _sst_da([280.0], attrs=ok_attrs)
    var = Variable(name="sst_i32", units="K", dtype="int32")
    r = validate_variable(da, var, check_dtype=True, check_range=False)
    assert any(
        i.code == "wrong_dtype" and i.severity is Severity.ERROR for i in r.issues
    )
