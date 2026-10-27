"""Gridded methane emission inventories (EDGAR, GFEI, EPA) as one CF flux.

Inventories ship as lat/lon rasters in different units: EDGAR's ``flx``
files in kg m⁻² s⁻¹, GFEI's grids in Mg km⁻² a⁻¹. :func:`open_ch4_inventory`
returns every one of them as the same CF variable, ``ch4_emission_flux`` in
kg m⁻² s⁻¹, on ascending ``lat`` and ``lon`` in [-180, 180).

Only area fluxes are accepted. Per-cell totals (EDGAR's ``emi`` files, in
tonnes per cell per year or per month) need the cell area and the period,
and the period is not in the file, so they are rejected rather than
converted with a guessed period: use the ``flx`` files instead.
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from xrreader._src.local.openers._common import promote_lonlat
from xrreader.types import CH4_EMISSION_FLUX, apply_cf_attrs


#: Seconds in a Julian year (365.25 d), the year in the ``a⁻¹`` conversions.
SECONDS_PER_YEAR: float = 365.25 * 86_400.0

#: Accepted flux units → factor to kg m⁻² s⁻¹ (spelling variants included).
_PER_YEAR_MG_KM2 = 1.0e3 / 1.0e6 / SECONDS_PER_YEAR
UNIT_FACTORS: dict[str, float] = {
    "kg m-2 s-1": 1.0,
    "kg m**-2 s**-1": 1.0,
    "kg/m2/s": 1.0,
    "kg m^-2 s^-1": 1.0,
    "g m-2 s-1": 1.0e-3,
    "g/m2/s": 1.0e-3,
    "Mg km-2 a-1": _PER_YEAR_MG_KM2,
    "Mg km-2 yr-1": _PER_YEAR_MG_KM2,
    "Mg/km2/yr": _PER_YEAR_MG_KM2,
    "Mg/km2/a": _PER_YEAR_MG_KM2,
    "t km-2 a-1": _PER_YEAR_MG_KM2,
    "t km-2 yr-1": _PER_YEAR_MG_KM2,
}


def open_ch4_inventory(
    path: str | Path,
    *,
    variable: str | None = None,
    units: str | None = None,
    source: str | None = None,
) -> xr.Dataset:
    """Open a gridded CH₄ emission inventory as a flux in kg m⁻² s⁻¹.

    The flux variable is ``variable`` or, when omitted, the one data
    variable on both ``lat`` and ``lon`` (``latitude`` / ``longitude`` are
    renamed first). Its units come from ``units`` or its ``units``
    attribute and must be an area flux in :data:`UNIT_FACTORS`; the values
    are multiplied by that factor, so with ``f`` the factor,
    ``F [kg m⁻² s⁻¹] = f · F_file``. Longitudes in [0, 360) are wrapped to
    [-180, 180), and both axes are sorted ascending.

    Args:
        path: NetCDF file of the inventory (one sector or a total).
        variable: Name of the flux variable; inferred when ``None``.
        units: Units of the flux, overriding the file's ``units``
            attribute (for files that omit or misspell it).
        source: Free-text provenance (e.g. ``"EDGAR v8.0"``), stored in
            the ``source`` attribute.

    Returns:
        Dataset with ``ch4_emission_flux`` on ``([time,] lat, lon)`` in
        kg m⁻² s⁻¹, with ``units_in_file`` and ``conversion_factor``
        attributes recording what was converted.

    Raises:
        KeyError: If ``variable`` is not in the file, or no variable (or
            more than one) lies on ``lat`` and ``lon`` when inferring it.
        ValueError: If the units are missing or not an accepted area flux.
    """
    ds = promote_lonlat(xr.open_dataset(path))
    name = variable if variable is not None else _infer_flux_variable(ds, path)
    if name not in ds.data_vars:
        raise KeyError(f"{path!s} has no variable {name!r}")
    flux = ds[name]
    units_in_file = units if units is not None else flux.attrs.get("units")
    factor = _factor(units_in_file, name)

    flux = _wrap_and_sort(flux * factor)
    # Overwrite: under `xr.set_options(keep_attrs=True)` the product above
    # keeps the file's `units`, which would mislabel the converted values.
    flux = apply_cf_attrs(
        flux.rename(CH4_EMISSION_FLUX.name), CH4_EMISSION_FLUX, overwrite=True
    )
    flux.attrs["units_in_file"] = str(units_in_file)
    flux.attrs["conversion_factor"] = factor
    out = flux.to_dataset()
    out.attrs = dict(ds.attrs)
    if source is not None:
        out.attrs["source"] = source
    return out


def _infer_flux_variable(ds: xr.Dataset, path: str | Path) -> str:
    gridded = [str(n) for n, v in ds.data_vars.items() if {"lat", "lon"} <= set(v.dims)]
    if len(gridded) != 1:
        raise KeyError(
            f"{path!s}: expected one variable on (lat, lon) to infer the flux, "
            f"found {gridded!r}; pass `variable=`"
        )
    return gridded[0]


def _factor(units: object, name: str) -> float:
    if units is None:
        raise ValueError(
            f"{name!r} has no `units` attribute; pass `units=` (one of "
            f"{sorted(UNIT_FACTORS)})"
        )
    key = " ".join(str(units).split())
    if key not in UNIT_FACTORS:
        raise ValueError(
            f"{name!r} is in {units!r}, which is not an accepted area flux "
            f"{sorted(UNIT_FACTORS)}. Per-cell totals (tonnes per cell) need a "
            "period the file does not state; use the inventory's flux files"
        )
    return UNIT_FACTORS[key]


def _wrap_and_sort(da: xr.DataArray) -> xr.DataArray:
    lon = da["lon"]
    if float(lon.max()) >= 180.0:
        da = da.assign_coords(lon=((lon + 180.0) % 360.0) - 180.0)
    return da.sortby("lat").sortby("lon")


__all__ = ["SECONDS_PER_YEAR", "UNIT_FACTORS", "open_ch4_inventory"]
