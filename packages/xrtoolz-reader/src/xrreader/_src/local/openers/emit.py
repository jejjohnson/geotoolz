"""EMIT L2B methane-enhancement opener.

Assumes the NetCDF container (the raster on ``(y, x)`` with the plume
enhancement and, optionally, its uncertainty). The distributed
``EMIT_L2B_CH4ENH`` GeoTIFFs are not read here — ``rioxarray`` is not an
``xrreader`` dependency — convert them to NetCDF first (e.g. with
``rioxarray.open_rasterio(...).to_netcdf``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from xrreader._src.local.openers._common import attach_scene_time, promote_lonlat
from xrreader.types import CH4_ENHANCEMENT, apply_cf_attrs


_RASTER_DIMS = {"downtrack": "y", "crosstrack": "x", "ortho_y": "y", "ortho_x": "x"}
_UNCERTAINTY = "ch4_uncertainty"
# Name of the enhancement raster in the distributed EMIT_L2B_CH4ENH files;
# the registry alias (``ch4_enhancement``) is accepted as well.
_PLUME_COMPLEX = "methane_plume_complex"


def open_emit_ch4_l2b(
    path: str | Path,
    *,
    glt_path: str | Path | None = None,
) -> xr.Dataset:
    """Open an EMIT L2B CH4 enhancement scene as a flat CF Dataset.

    The raster carries the plume enhancement (ppm m) as either
    ``methane_plume_complex`` (the distributed variable name) or
    ``ch4_enhancement`` (the registry alias) — renamed to
    ``ch4_enhancement`` — and, when present, ``ch4_uncertainty`` on
    ``(y, x)``. With ``glt_path`` the scene is
    orthorectified through EMIT's geometric lookup table: ``glt_x`` /
    ``glt_y`` hold the 1-based raw-pixel column / row that fills each
    orthorectified cell (``0`` marks no data), and the 2-D ``lon`` /
    ``lat`` of the orthorectified grid come from the GLT file's own
    ``lon`` / ``lat`` variables or, failing that, its GDAL ``geotransform``
    attribute (the full six-coefficient affine, rotation terms included).
    Without a GLT, 2-D ``lon`` / ``lat`` are taken from the raster file
    itself when it has them. A scalar ``time`` variable on the raster
    survives orthorectification and becomes the scene timestamp.

    Args:
        path: NetCDF raster with ``methane_plume_complex`` /
            ``ch4_enhancement`` on ``(y, x)`` (EMIT's ``downtrack`` /
            ``crosstrack`` dims are renamed).
        glt_path: Optional geometric lookup table NetCDF.

    Returns:
        Scene Dataset on ``(time, y, x)`` (``time`` of length 1 when the
        file exposes a timestamp) with 2-D ``lon`` / ``lat`` coordinates.

    Raises:
        KeyError: If neither ``methane_plume_complex`` nor
            ``ch4_enhancement`` is in the raster.
        ValueError: If the GLT exposes neither ``lon``/``lat`` nor a
            ``geotransform``.
    """
    ds = xr.open_dataset(path)
    ds = ds.rename({k: v for k, v in _RASTER_DIMS.items() if k in ds.dims})
    candidates = (CH4_ENHANCEMENT.for_source("emit"), _PLUME_COMPLEX)
    name = next((n for n in candidates if n in ds.data_vars), None)
    if name is None:
        raise KeyError(f"{path!s} has none of {candidates!r}")
    ds = ds.rename({name: CH4_ENHANCEMENT.name}) if name != CH4_ENHANCEMENT.name else ds
    if glt_path is not None:
        ds = _orthorectify(ds, xr.open_dataset(glt_path))
    ds = promote_lonlat(ds)
    ds[CH4_ENHANCEMENT.name] = apply_cf_attrs(ds[CH4_ENHANCEMENT.name], CH4_ENHANCEMENT)
    return attach_scene_time(ds)


def _orthorectify(ds: xr.Dataset, glt: xr.Dataset) -> xr.Dataset:
    """Resample every ``(y, x)`` variable of ``ds`` onto the GLT grid."""
    glt = glt.rename({k: v for k, v in _RASTER_DIMS.items() if k in glt.dims})
    gx = glt["glt_x"].values.astype(np.int64)
    gy = glt["glt_y"].values.astype(np.int64)
    valid = (gx > 0) & (gy > 0)
    iy = np.where(valid, gy - 1, 0)
    ix = np.where(valid, gx - 1, 0)
    out: dict[str, tuple[tuple[str, str], np.ndarray, dict]] = {}
    for name, da in ds.data_vars.items():
        if set(da.dims) != {"y", "x"}:
            continue
        arr = da.transpose("y", "x").values[iy, ix].astype(float)
        arr[~valid] = np.nan
        out[str(name)] = (("y", "x"), arr, dict(da.attrs))
    if "lon" in glt.variables and "lat" in glt.variables:
        lon = glt["lon"].values
        lat = glt["lat"].values
    elif "geotransform" in glt.attrs:
        # GDAL order: [ulx, a, b, uly, d, e] — b / d are the rotation terms.
        ulx, a, b, uly, d, e = (float(v) for v in glt.attrs["geotransform"])
        jj, ii = np.indices(gx.shape)
        lon = ulx + (ii + 0.5) * a + (jj + 0.5) * b
        lat = uly + (ii + 0.5) * d + (jj + 0.5) * e
    else:
        raise ValueError("GLT carries neither lon/lat variables nor a geotransform")
    coords: dict[str, object] = {
        "lon": (
            ("y", "x"),
            lon,
            {"standard_name": "longitude", "units": "degrees_east"},
        ),
        "lat": (
            ("y", "x"),
            lat,
            {"standard_name": "latitude", "units": "degrees_north"},
        ),
    }
    if "time" in ds.variables and ds["time"].ndim == 0:
        coords["time"] = ds["time"]  # scene timestamp, see attach_scene_time
    return xr.Dataset(out, coords=coords, attrs=dict(ds.attrs))
