"""open_ch4_inventory on synthetic EDGAR- and GFEI-shaped files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from xrproducts import open_ch4_inventory
from xrproducts.types import CH4_EMISSION_FLUX


def _write(
    path: Path,
    values: np.ndarray,
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    name: str = "fluxes",
    units: str | None = "kg m-2 s-1",
    long_names: bool = False,
    extra: dict[str, xr.DataArray] | None = None,
) -> Path:
    dims = ("latitude", "longitude") if long_names else ("lat", "lon")
    da = xr.DataArray(values, dims=dims, coords={dims[0]: lat, dims[1]: lon})
    if units is not None:
        da.attrs["units"] = units
    ds = xr.Dataset({name: da, **(extra or {})})
    ds.to_netcdf(path)
    return path


def test_kg_m2_s_passes_through_with_cf_attrs(tmp_path: Path) -> None:
    lat, lon = np.array([10.0, 20.0]), np.array([-5.0, 5.0, 15.0])
    values = np.arange(6.0).reshape(2, 3) * 1e-10
    ds = open_ch4_inventory(_write(tmp_path / "f.nc", values, lat=lat, lon=lon))
    flux = ds[CH4_EMISSION_FLUX.name]
    np.testing.assert_array_equal(flux.values, values)
    assert flux.attrs["units"] == "kg m-2 s-1"
    assert flux.attrs["standard_name"] == CH4_EMISSION_FLUX.standard_name
    assert flux.attrs["conversion_factor"] == 1.0


def test_gfei_units_convert_to_kg_m2_s(tmp_path: Path) -> None:
    # 1 Mg km⁻² a⁻¹ = 1e3 kg / 1e6 m² / (365.25 · 86400 s)
    lat, lon = np.array([0.05]), np.array([0.05])
    path = _write(
        tmp_path / "gfei.nc",
        np.array([[1.0]]),
        lat=lat,
        lon=lon,
        name="emis_ch4",
        units="Mg km-2 a-1",
    )
    flux = open_ch4_inventory(path, source="GFEI v2")["ch4_emission_flux"]
    expected = 1.0e3 / 1.0e6 / (365.25 * 86_400.0)
    np.testing.assert_allclose(float(flux.squeeze()), expected, rtol=1e-12)
    assert flux.attrs["units_in_file"] == "Mg km-2 a-1"


def test_longitudes_wrap_and_axes_sort(tmp_path: Path) -> None:
    lat = np.array([30.0, 10.0])  # descending, as some inventories store it
    lon = np.array([10.0, 190.0, 350.0])
    values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    ds = open_ch4_inventory(
        _write(tmp_path / "w.nc", values, lat=lat, lon=lon, long_names=True)
    )
    flux = ds["ch4_emission_flux"]
    np.testing.assert_array_equal(flux["lat"].values, [10.0, 30.0])
    np.testing.assert_allclose(flux["lon"].values, [-170.0, -10.0, 10.0])
    # (lat=10, lon=190→-170) was values[1, 1]
    assert float(flux.sel(lat=10.0, lon=-170.0)) == 5.0
    assert float(flux.sel(lat=30.0, lon=10.0)) == 1.0


def test_units_override_and_missing_units(tmp_path: Path) -> None:
    lat, lon = np.array([0.0]), np.array([0.0])
    path = _write(tmp_path / "u.nc", np.array([[2.0]]), lat=lat, lon=lon, units=None)
    with pytest.raises(ValueError, match="no `units`"):
        open_ch4_inventory(path)
    flux = open_ch4_inventory(path, units="g m-2 s-1")["ch4_emission_flux"]
    assert float(flux.squeeze()) == pytest.approx(2.0e-3)


def test_per_cell_totals_are_rejected(tmp_path: Path) -> None:
    lat, lon = np.array([0.0]), np.array([0.0])
    path = _write(
        tmp_path / "e.nc", np.array([[1.0]]), lat=lat, lon=lon, units="Tonnes"
    )
    with pytest.raises(ValueError, match="Per-cell totals"):
        open_ch4_inventory(path)


def test_variable_inference_needs_exactly_one_grid(tmp_path: Path) -> None:
    lat, lon = np.array([0.0]), np.array([0.0])
    other = xr.DataArray([[1.0]], dims=("lat", "lon"), attrs={"units": "kg m-2 s-1"})
    path = _write(
        tmp_path / "two.nc", np.array([[1.0]]), lat=lat, lon=lon, extra={"b": other}
    )
    with pytest.raises(KeyError, match="pass `variable=`"):
        open_ch4_inventory(path)
    assert "ch4_emission_flux" in open_ch4_inventory(path, variable="b")
    with pytest.raises(KeyError, match="no variable"):
        open_ch4_inventory(path, variable="missing")


def test_units_are_relabelled_under_keep_attrs(tmp_path: Path) -> None:
    lat, lon = np.array([0.0]), np.array([0.0])
    path = _write(
        tmp_path / "g.nc", np.array([[2.0]]), lat=lat, lon=lon, units="g m-2 s-1"
    )
    with xr.set_options(keep_attrs=True):
        flux = open_ch4_inventory(path)["ch4_emission_flux"]
    assert float(flux.squeeze()) == pytest.approx(2.0e-3)
    assert flux.attrs["units"] == "kg m-2 s-1"
    assert flux.attrs["units_in_file"] == "g m-2 s-1"
