"""to_xarray / from_xarray as decided in geostack#77 (GA-Q2 = LL-F1).

Writes a scalar ``spatial_ref`` coordinate with ``crs_wkt``, ``spatial_ref``,
pyproj's ``CRS.to_cf()`` attributes and GDAL's ``GeoTransform``; x and y
centre coordinates when the grid is axis-aligned. Reads ``GeoTransform``
first, then the centres; when both exist they must agree to 1e-9 pixel.
No rioxarray import.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyproj

from proto.carrier import GeoArray


def _gdal(t: np.ndarray) -> str:
    a, b, c, d, e, f = (float(v) for v in t)
    # GDAL order (c, a, b, f, d, e), repr so float64 round-trips exactly
    return " ".join(repr(v) for v in (c, a, b, f, d, e))


def _from_gdal(s: str) -> np.ndarray:
    c, a, b, f, d, e = (float(v) for v in s.split())
    return np.array([a, b, c, d, e, f], dtype=np.float64)


def to_xarray(g: GeoArray) -> Any:
    """A DataArray with the carrier's own dims, its grid in ``spatial_ref``."""
    import xarray as xr

    if g.transform.ndim != 1:
        raise ValueError("to_xarray takes one transform; map over the batch dim first")
    t = np.asarray(g.transform, dtype=np.float64)
    a, b, c, d, e, f = t
    H, W = g.shape[-2:]
    attrs: dict[str, Any] = {"GeoTransform": _gdal(t)}
    if g.crs is not None:
        wkt = g.crs.to_wkt()
        attrs |= {k: v for k, v in g.crs.to_cf().items() if k != "crs_wkt"}
        attrs |= {"crs_wkt": wkt, "spatial_ref": wkt}
    coords: dict[str, Any] = {"spatial_ref": ((), 0, attrs)}
    if b == 0 and d == 0:
        coords["x"] = ("x", c + a * (np.arange(W) + 0.5))
        coords["y"] = ("y", f + e * (np.arange(H) + 0.5))
    for dim, lab in g.labels:
        coords[dim] = (dim, list(lab))
    da = xr.DataArray(np.asarray(g.data), dims=g.dims, coords=coords)
    da.encoding["grid_mapping"] = "spatial_ref"
    if g.fill_value is not None:
        da.attrs["_FillValue"] = g.fill_value
    return da


def from_xarray(da: Any) -> GeoArray:
    """The carrier back: GeoTransform first, else centres; both must agree."""
    ref = da.coords.get("spatial_ref")
    t_gdal = None if ref is None else ref.attrs.get("GeoTransform")
    t = None if t_gdal is None else _from_gdal(t_gdal)
    has_xy = "x" in da.coords and "y" in da.coords
    if has_xy:
        x = np.asarray(da.coords["x"].values, dtype=np.float64)
        y = np.asarray(da.coords["y"].values, dtype=np.float64)
    if t is None:
        if not has_xy or len(x) < 2 or len(y) < 2:
            raise ValueError("no GeoTransform and no 1-D x/y centres with two or more points")
        dx, dy = x[1] - x[0], y[1] - y[0]
        t = np.array([dx, 0.0, x[0] - dx / 2, 0.0, dy, y[0] - dy / 2])
    elif has_xy and t[1] == 0 and t[3] == 0:
        a, _, c, _, e, f = t
        ex = c + a * (np.arange(len(x)) + 0.5)
        ey = f + e * (np.arange(len(y)) + 0.5)
        tol_x, tol_y = 1e-9 * abs(a), 1e-9 * abs(e)
        if np.max(np.abs(x - ex)) > tol_x or np.max(np.abs(y - ey)) > tol_y:
            raise ValueError("x/y coordinates disagree with GeoTransform by more than 1e-9 pixel")
    crs = None
    if ref is not None and "crs_wkt" in ref.attrs:
        crs = pyproj.CRS.from_wkt(ref.attrs["crs_wkt"])
    labels = tuple(
        (d, tuple(da.coords[d].values.tolist()))
        for d in da.dims
        if d not in ("y", "x") and d in da.coords
    )
    fill = da.attrs.get("_FillValue")
    return GeoArray(np.asarray(da.data), t, tuple(da.dims), crs, fill, labels)
