"""WRF-ARW ``wrfout`` files as CF Datasets.

``wrfout_d0X_*.nc`` files are not CF: time is a ``Times`` char array (or
``XTIME`` minutes), ``U`` / ``V`` / ``W`` sit on staggered faces,
and are grid-relative (rotated to earth-relative via ``COSALPHA`` /
``SINALPHA`` when the file carries them), temperature is a perturbation
potential temperature ``T`` around 300 K, pressure and geopotential are
split into base + perturbation (``P + PB``, ``PH + PHB``), and height above
ground has to be reconstructed per column from the geopotential and
``HGT``. :func:`open_wrfout` does all of that lazily (no data are
materialised, so ``chunks=`` keeps everything dask backed) and returns a
Dataset on ``(time, level, y, x)``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import xarray as xr
from xrreader.types import Variable, apply_cf_attrs

from xrtoolz.atm._src.constants import C_P, R_D


#: Gravitational acceleration WRF uses for the geopotential [m s⁻²].
GRAVITY: float = 9.81
#: Reference pressure for potential temperature [Pa].
P_REFERENCE: float = 1.0e5
#: Base-state potential temperature added to the perturbation ``T`` [K].
THETA_BASE: float = 300.0
#: Poisson constant ``R_d / c_p = 287.04 / 1004.5``.
KAPPA: float = R_D / C_P

#: WRF dimension → CF-style dimension.
_DIMS: dict[str, str] = {
    "Time": "time",
    "bottom_top": "level",
    "bottom_top_stag": "level_stag",
    "south_north": "y",
    "south_north_stag": "y_stag",
    "west_east": "x",
    "west_east_stag": "x_stag",
}
#: Global attributes copied through for CRS work.
_PROJECTION_ATTRS: tuple[str, ...] = (
    "DX",
    "DY",
    "MAP_PROJ",
    "CEN_LAT",
    "CEN_LON",
    "TRUELAT1",
    "TRUELAT2",
    "STAND_LON",
)

_W = Variable(
    name="w",
    standard_name="upward_air_velocity",
    long_name="Vertical velocity",
    units="m s-1",
)
_U_GRID = Variable(name="u", long_name="grid-relative U wind component", units="m s-1")
_V_GRID = Variable(name="v", long_name="grid-relative V wind component", units="m s-1")
_PRESSURE = Variable(
    name="pressure", standard_name="air_pressure", long_name="Pressure", units="Pa"
)
_TERRAIN = Variable(
    name="terrain",
    standard_name="surface_altitude",
    long_name="Terrain height",
    units="m",
)
_Z_AGL = Variable(
    name="z_agl", standard_name="height", long_name="Height above ground", units="m"
)
_LON = Variable(
    name="lon", standard_name="longitude", long_name="Longitude", units="degrees_east"
)
_LAT = Variable(
    name="lat", standard_name="latitude", long_name="Latitude", units="degrees_north"
)
_X = Variable(
    name="x", long_name="Distance east of the south-west mass point", units="m"
)
_Y = Variable(
    name="y", long_name="Distance north of the south-west mass point", units="m"
)


def _renamed(var: xr.Variable, dim: str, new_dim: str) -> xr.Variable:
    dims = tuple(new_dim if d == dim else d for d in var.dims)
    return xr.Variable(dims, var.data, var.attrs, var.encoding)


def destagger(
    da: xr.DataArray, dim: str, *, new_dim: str | None = None
) -> xr.DataArray:
    """Average a face-staggered variable onto cell centres along ``dim``.

    ``out[..., i, ...] = 0.5 * (da[..., i, ...] + da[..., i + 1, ...])``, so
    the output has ``da.sizes[dim] - 1`` entries along ``new_dim``. Numeric
    coordinates carrying ``dim`` are averaged the same way (non-numeric ones
    are dropped); other coordinates and ``attrs`` are preserved. Exact for
    fields linear in the staggered coordinate.

    Args:
        da: Staggered variable (eager or dask backed; stays lazy).
        dim: Staggered dimension to average over.
        new_dim: Name of the output dimension; defaults to ``dim`` with a
            trailing ``_stag`` stripped.

    Returns:
        The cell-centred variable.

    Raises:
        ValueError: If ``dim`` is not a dimension of ``da``.
    """
    if dim not in da.dims:
        raise ValueError(f"dim {dim!r} not in DataArray dims {tuple(da.dims)}")
    if new_dim is None:
        new_dim = dim.removesuffix("_stag")
    lo, hi = {dim: slice(None, -1)}, {dim: slice(1, None)}

    def _centre(var: xr.Variable) -> xr.Variable:
        return _renamed(0.5 * (var.isel(lo) + var.isel(hi)), dim, new_dim)

    coords: dict[str, xr.Variable] = {}
    for name, coord in da.coords.items():
        if dim not in coord.dims:
            coords[str(name)] = coord.variable
        elif np.issubdtype(coord.dtype, np.number):
            coords[str(name) if name != dim else new_dim] = _centre(coord.variable)
    return xr.DataArray(
        _centre(da.variable), coords=coords, name=da.name, attrs=da.attrs
    )


def _decode_times(rows: np.ndarray) -> np.ndarray:
    out = []
    for row in rows:
        raw = row.tobytes() if isinstance(row, np.ndarray) else row
        text = raw.decode() if isinstance(raw, bytes) else str(raw)
        out.append(np.datetime64(text.strip("\x00 ").replace("_", "T"), "s"))
    return np.asarray(out, dtype="datetime64[s]")


def wrf_time(ds: xr.Dataset) -> xr.DataArray:
    """Time axis of a ``wrfout`` Dataset as ``datetime64[s]``.

    Decodes the ``Times`` char rows (``YYYY-MM-DD_HH:MM:SS``) when present;
    otherwise uses ``XTIME`` — raw minutes since the ``SIMULATION_START_DATE``
    global attribute, or datetimes if a caller already decoded it.

    Args:
        ds: Raw ``wrfout`` Dataset (dimension names may be renamed already).

    Returns:
        1-D ``datetime64[s]`` DataArray named ``"time"`` on the source's
        time dimension.

    Raises:
        ValueError: If neither ``Times`` nor ``XTIME`` is present, or
            ``XTIME`` holds minutes and ``SIMULATION_START_DATE`` is missing.
    """
    if "Times" in ds:
        source = ds["Times"]
        stamps = _decode_times(np.asarray(source.values))
    elif "XTIME" in ds:
        source = ds["XTIME"]
        xtime = np.asarray(source.values)
        if np.issubdtype(xtime.dtype, np.datetime64):
            stamps = xtime.astype("datetime64[s]")
        else:
            start = ds.attrs.get("SIMULATION_START_DATE")
            if start is None:
                raise ValueError(
                    "`XTIME` holds minutes but the `SIMULATION_START_DATE` "
                    "global attribute is missing"
                )
            origin = np.datetime64(str(start).strip().replace("_", "T"), "s")
            seconds = np.round(np.asarray(xtime, dtype=np.float64) * 60.0)
            stamps = origin + seconds.astype("timedelta64[s]")
    else:
        raise ValueError(
            "neither `Times` nor `XTIME` present; cannot build a time axis"
        )
    return xr.DataArray(stamps, dims=(source.dims[0],), name="time")


def _stag_dim(da: xr.DataArray) -> str:
    stag = [str(d) for d in da.dims if str(d).endswith("_stag")]
    if len(stag) != 1:
        raise ValueError(f"expected exactly one staggered dim, got {tuple(da.dims)}")
    return stag[0]


def wrf_height_agl(ds: xr.Dataset) -> xr.DataArray:
    """Height above ground on mass levels.

    ``z_stag = (PH + PHB) / GRAVITY`` on the staggered levels, averaged to
    mass levels and reduced by the terrain height:
    ``z_agl = 0.5 * (z_stag[k] + z_stag[k + 1]) - HGT``. ``HGT`` may carry a
    time axis or not.

    Args:
        ds: Raw ``wrfout`` Dataset with ``PH``, ``PHB`` and ``HGT``.

    Returns:
        ``(time, level, y, x)`` height above ground [m].
    """
    z_stag = (ds["PH"] + ds["PHB"]) / GRAVITY
    return destagger(z_stag, _stag_dim(z_stag)) - ds["HGT"]


def wrf_temperature(ds: xr.Dataset) -> xr.DataArray:
    """Air temperature from WRF's perturbation potential temperature.

    ``T = (T' + THETA_BASE) * ((P + PB) / P_REFERENCE) ** KAPPA`` [K].

    Args:
        ds: Raw ``wrfout`` Dataset with ``T``, ``P`` and ``PB``.

    Returns:
        Air temperature on mass points [K].
    """
    pressure = ds["P"] + ds["PB"]
    return (ds["T"] + THETA_BASE) * (pressure / P_REFERENCE) ** KAPPA


def _clean(da: xr.DataArray) -> xr.DataArray:
    """Strip WRF's raw ``XLONG`` / ``XLAT`` / ``XTIME`` coords and attrs."""
    out = da.reset_coords(drop=True)
    out.attrs = {}
    return out


def _static(da: xr.DataArray) -> xr.DataArray:
    """Drop a ``time`` axis that WRF stamps on a field that never changes.

    ``HGT`` / ``XLONG`` / ``XLAT`` / ``COSALPHA`` / ``SINALPHA`` carry a
    ``Time`` axis in every ``wrfout`` but only vary on moving nests, so the
    axis is collapsed only when every step equals the first. The comparison
    is a coordinate-sized computation, so it is evaluated even for dask.
    """
    if "time" not in da.dims:
        return da
    first = da.isel(time=0)
    if da.sizes["time"] <= 1 or bool((da == first).all().values):
        return first
    return da


def wrf_wind(ds: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    """Horizontal wind on mass points, earth-relative where possible.

    ``U`` / ``V`` are destaggered onto mass points. When the file carries
    the map-rotation fields ``COSALPHA`` / ``SINALPHA`` the components are
    rotated from grid-relative to earth-relative,
    ``u_e = u cos α − v sin α``, ``v_e = v cos α + u sin α``, and stamped
    with the CF ``eastward_wind`` / ``northward_wind`` names. Otherwise the
    grid-relative components are returned with no ``standard_name`` and the
    attr ``grid_relative = "true"``.

    Args:
        ds: Raw ``wrfout`` Dataset (dims renamed) with ``U`` and ``V``.

    Returns:
        ``(u, v)`` on mass points [m s⁻¹].
    """
    u = destagger(_clean(ds["U"]), "x_stag")
    v = destagger(_clean(ds["V"]), "y_stag")
    if "COSALPHA" in ds and "SINALPHA" in ds:
        cos_a = _static(_clean(ds["COSALPHA"]))
        sin_a = _static(_clean(ds["SINALPHA"]))
        u, v = u * cos_a - v * sin_a, v * cos_a + u * sin_a
        return (
            apply_cf_attrs(u, "u", overwrite=True),
            apply_cf_attrs(v, "v", overwrite=True),
        )
    u = apply_cf_attrs(u, _U_GRID, overwrite=True)
    v = apply_cf_attrs(v, _V_GRID, overwrite=True)
    u.attrs["grid_relative"] = "true"
    v.attrs["grid_relative"] = "true"
    return u, v


def open_wrfout(
    path: str | Path,
    *,
    times: slice | None = None,
    variables: str | Sequence[str] | None = None,
    chunks: Mapping[str, int] | None = None,
) -> xr.Dataset:
    """Open a ``wrfout`` file as a CF-conformant Dataset.

    Returns dims ``(time, level, y, x)`` with:

    - coords: ``time`` (``datetime64[s]``), ``x`` / ``y`` (metres from the
      south-west mass point, from ``DX`` / ``DY``), ``level`` (int index),
      ``z_agl(time, level, y, x)`` (m above ground), and 2-D ``lon`` /
      ``lat`` from ``XLONG`` / ``XLAT`` when present.
    - data_vars (all on mass points): ``u``, ``v`` (m s⁻¹, destaggered and
      rotated to earth-relative when ``COSALPHA`` / ``SINALPHA`` are in the
      file — see :func:`wrf_wind`), ``w`` (m s⁻¹, destaggered),
      ``temperature`` (K), ``pressure`` (Pa), ``pbl_height`` (m, from
      ``PBLH``), ``terrain`` (m, from ``HGT``), plus any raw variable named
      in ``variables`` (dims renamed, otherwise untouched).
    - attrs: ``DX``, ``DY``, ``MAP_PROJ``, ``CEN_LAT``, ``CEN_LON``,
      ``TRUELAT1`` / ``2``, ``STAND_LON`` copied through for CRS work.

    ``HGT``, ``XLONG`` and ``XLAT`` lose their ``time`` axis only when they
    are the same at every step; on a moving nest they keep it. CF
    ``standard_name`` / ``units`` are stamped via
    :func:`xrreader.apply_cf_attrs`. Nothing is materialised, so
    ``chunks={"time": 1}`` keeps every variable dask backed. The returned
    Dataset owns the file handle: closing it (or leaving a ``with`` block)
    closes the file.

    Args:
        path: ``wrfout`` NetCDF file.
        times: Optional ``slice`` over the file's time axis.
        variables: Raw WRF variable name(s) to pass through (e.g.
            ``"QVAPOR"`` or ``["QVAPOR", "QCLOUD"]``).
        chunks: Dask chunks keyed by the *output* dimension names.

    Returns:
        The CF Dataset.

    Raises:
        ValueError: If no time axis can be built (see :func:`wrf_time`).
        KeyError: If a name in ``variables`` is not in the file.
    """
    if isinstance(variables, str):
        variables = [variables]
    raw = xr.open_dataset(path, decode_times=False)
    close = raw._close  # rename/isel/chunk drop it; re-attached on the output
    raw = raw.rename({k: v for k, v in _DIMS.items() if k in raw.dims})
    if times is not None:
        raw = raw.isel(time=times)
    if chunks:
        raw = raw.chunk(chunks)

    u, v = wrf_wind(raw)
    data_vars: dict[str, xr.DataArray] = {
        "u": u,
        "v": v,
        "w": apply_cf_attrs(
            destagger(_clean(raw["W"]), "level_stag"), _W, overwrite=True
        ),
        "temperature": apply_cf_attrs(
            _clean(wrf_temperature(raw)), "t", overwrite=True
        ),
        "pressure": apply_cf_attrs(
            _clean(raw["P"] + raw["PB"]), _PRESSURE, overwrite=True
        ),
        "pbl_height": apply_cf_attrs(_clean(raw["PBLH"]), "blh", overwrite=True),
        "terrain": apply_cf_attrs(
            _static(_clean(raw["HGT"])), _TERRAIN, overwrite=True
        ),
    }
    for name in variables or ():
        data_vars[name] = raw[name]

    dx, dy = float(raw.attrs["DX"]), float(raw.attrs["DY"])
    z_agl = apply_cf_attrs(_clean(wrf_height_agl(raw)), _Z_AGL, overwrite=True)
    z_agl.attrs["positive"] = "up"
    coords: dict[str, xr.DataArray] = {
        "time": wrf_time(raw),
        "level": xr.DataArray(
            np.arange(raw.sizes["level"]),
            dims="level",
            attrs={"long_name": "Mass level index"},
        ),
        "y": apply_cf_attrs(xr.DataArray(dy * np.arange(raw.sizes["y"]), dims="y"), _Y),
        "x": apply_cf_attrs(xr.DataArray(dx * np.arange(raw.sizes["x"]), dims="x"), _X),
        "z_agl": z_agl,
    }
    for name, key, variable in (("lon", "XLONG", _LON), ("lat", "XLAT", _LAT)):
        if key in raw:
            coords[name] = apply_cf_attrs(
                _static(_clean(raw[key])), variable, overwrite=True
            )

    attrs = {k: raw.attrs[k] for k in _PROJECTION_ATTRS if k in raw.attrs}
    out = xr.Dataset(data_vars, coords=coords, attrs=attrs)
    out.set_close(close)
    return out
