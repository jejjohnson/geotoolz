"""Lossless converters between geoarray's carriers and xarray (#537, LL-F).

The CRS encoding is decided (GA-Q2 = LL-F1, geostack#77): a scalar
``spatial_ref`` coordinate holding ``crs_wkt``, ``spatial_ref`` (the same
WKT), pyproj's ``CRS.to_cf()`` attributes and, for a grid, GDAL's
``GeoTransform``; the variable names it as its ``grid_mapping``. This is
what rioxarray, GDAL's netCDF driver, xrtoolz, geopatcher and geocloud
read, and it is written here without importing rioxarray.

| Carrier | xarray |
| --- | --- |
| `GeoArray` | ``DataArray``, same dims; x / y centres when axis-aligned |
| `GeoSwath` | ``DataArray`` with 2-D ``lat`` / ``lon`` auxiliary coordinates |
| `GeoPoints` | a CF Discrete Sampling Geometry ``Dataset`` on ``obs`` |
| `SpectralAxis` | ``band`` names; indexed ``wavelength`` / ``fwhm`` (nm) |
| `VariableAxis` | ``variable`` names and ``variable_units`` |
| `VerticalAxis` | ``level`` values with CF ``units`` / ``positive`` |
| `TimeAxis` | ``time`` as ``datetime64[ns]`` |
| ``fill_value`` | ``_FillValue`` encoding and ``attrs["fill_value"]`` |

Reading a grid trusts, in order: ``GeoTransform``; then the centre
coordinates; then georeader's ``attrs["crs"]`` (with a
``DeprecationWarning``). When a ``GeoTransform`` and centres both exist they
must agree to 10⁻⁹ of a pixel, or the read raises (GF3).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import pyproj
import xarray as xr

from geoarray._src.array import SPATIAL_DIMS, GeoArray
from geoarray._src.attrs import FrozenAttrs
from geoarray._src.backend import is_tracer
from geoarray._src.carriers import GeoPoints, GeoSwath
from geoarray._src.coords import (
    Coordinate,
    SpectralAxis,
    TimeAxis,
    VariableAxis,
    Vertical,
    VerticalAxis,
)
from geoarray._src.errors import GeoArrayTraceError, GridMismatchError
from geoarray._src.grid import apply_transform
from geoarray._src.types import Kind, register_kind_reader


__all__ = [
    "from_dataset",
    "from_datatree",
    "from_xarray",
    "to_dataset",
    "to_datatree",
    "to_xarray",
]

GRID_MAPPING = "spatial_ref"
#: the one per-band mapping: SpectralAxis field -> coordinate on ``band``
SPECTRAL_FIELDS: tuple[tuple[str, str], ...] = (
    ("wavelength_nm", "wavelength"),
    ("fwhm_nm", "fwhm"),
)
#: Vertical kind -> CF standard_name
_VERTICAL_NAMES = {"height": "height", "depth": "depth", "pressure": "air_pressure"}
#: attrs keys the converters own, removed when reading back
_OWN_ATTRS = ("fill_value", "_FillValue", "grid_mapping", "coordinates")
#: tolerance for centres against a GeoTransform, in pixels
_PIXEL_TOL = 1e-9


# -- writing --------------------------------------------------------------------------


def _host(data: Any, what: str) -> np.ndarray:
    if is_tracer(data):
        raise GeoArrayTraceError(
            f"{what} needs concrete data; call it outside jit / grad / vmap"
        )
    return np.asarray(data)


def _gdal(t: np.ndarray) -> str:
    """``T = (a, b, c, d, e, f)`` in GDAL's order (c, a, b, f, d, e), exactly."""
    a, b, c, d, e, f = (float(v) for v in t)
    return " ".join(repr(v) for v in (c, a, b, f, d, e))


def _from_gdal(text: Any) -> np.ndarray:
    values = text.split() if isinstance(text, str) else list(text)
    c, a, b, f, d, e = (float(v) for v in values)
    return np.array([a, b, c, d, e, f], dtype=np.float64)


def _spatial_ref(crs: pyproj.CRS | None, transform: np.ndarray | None) -> Any:
    attrs: dict[str, Any] = {}
    if crs is not None:
        wkt = crs.to_wkt()
        attrs |= {k: v for k, v in crs.to_cf().items() if k != "crs_wkt"}
        attrs |= {"crs_wkt": wkt, "spatial_ref": wkt}
    if transform is not None:
        attrs["GeoTransform"] = _gdal(transform)
    return xr.Variable((), np.int32(0), attrs)


def _coordinate_vars(coords: tuple[Coordinate, ...]) -> dict[str, Any]:
    """The xarray coordinates of the 1-D axes."""
    out: dict[str, Any] = {}
    for c in coords:
        if isinstance(c, SpectralAxis):
            out["band"] = ("band", list(c.names))
            for field, name in SPECTRAL_FIELDS:
                values = getattr(c, field)
                if values is not None:
                    out[name] = ("band", np.asarray(values), {"units": "nm"})
        elif isinstance(c, VariableAxis):
            out["variable"] = ("variable", list(c.names))
            if c.units is not None:
                out["variable_units"] = ("variable", list(c.units))
        elif isinstance(c, VerticalAxis):
            v = c.vertical
            attrs = {"units": v.units, "positive": v.positive, "vertical_kind": v.kind}
            if v.kind in _VERTICAL_NAMES:
                attrs["standard_name"] = _VERTICAL_NAMES[v.kind]
            if v.datum is not None:
                attrs["datum"] = v.datum
            out["level"] = ("level", np.asarray(c.values), attrs)
        elif isinstance(c, TimeAxis):
            out["time"] = ("time", np.asarray(c.to_host().stamps))
    return out


def _attrs_out(attrs: FrozenAttrs, fill: Any) -> dict[str, Any]:
    out = {
        k: list(v) if isinstance(v, tuple) else v for k, v in attrs.to_dict().items()
    }
    if fill is not None:
        # netCDF has no bool attributes: a mask's fill is written as 0 / 1
        out["fill_value"] = int(fill) if isinstance(fill, bool | np.bool_) else fill
    return out


def _encode(da: Any, fill: Any) -> Any:
    da.encoding["grid_mapping"] = GRID_MAPPING
    # netCDF has no bool type (xarray writes int8), so a bool fill stays in attrs
    if fill is not None and da.dtype.kind != "b":
        nan = isinstance(fill, float) and math.isnan(fill)
        da.encoding["_FillValue"] = np.nan if nan else fill
    elif fill is None:
        # no fill: stop xarray writing its default NaN _FillValue for floats
        da.encoding["_FillValue"] = None
    return da


def _with_indexes(obj: Any) -> Any:
    """Index ``wavelength`` (and ``fwhm``) so ``sel(wavelength=…)`` works (LL-F2)."""
    for _, name in SPECTRAL_FIELDS:
        if name in obj.coords and name not in obj.indexes:
            obj = obj.set_xindex(name)
    return obj


def grid_to_xarray(g: GeoArray) -> Any:
    """A `GeoArray` as a ``DataArray`` (see the module table)."""
    if g.batch_shape:
        raise ValueError(
            "to_xarray takes one grid; this carrier has one transform per "
            f"{g.batch_dims} item: select one, or concat the items first"
        )
    t = np.asarray(_host(g.transform, "to_xarray"), dtype=np.float64)
    _, b, _, d, _, _ = (float(v) for v in t)
    coords: dict[str, Any] = {GRID_MAPPING: _spatial_ref(g.crs, t)}
    if b == 0 and d == 0:
        x, y = _axis_centres(t, g.width, g.height)
        coords["x"] = ("x", x, {"axis": "X"})
        coords["y"] = ("y", y, {"axis": "Y"})
    coords |= _coordinate_vars(g.coords)
    da = xr.DataArray(
        _host(g.data, "to_xarray"),
        dims=g.dims,
        coords=coords,
        attrs=_attrs_out(g.attrs, g.fill_value),
    )
    return _with_indexes(_encode(da, g.fill_value))


def swath_to_xarray(s: GeoSwath) -> Any:
    """A `GeoSwath` as a ``DataArray`` with 2-D lat/lon coordinates."""
    geo_dims = (*s.batch_dims, "row", "col")
    coords: dict[str, Any] = {
        GRID_MAPPING: _spatial_ref(s.crs, None),
        "lat": (geo_dims, _host(s.lat, "to_xarray"), _LAT),
        "lon": (geo_dims, _host(s.lon, "to_xarray"), _LON),
    }
    coords |= _coordinate_vars(s.coords)
    da = xr.DataArray(
        _host(s.data, "to_xarray"),
        dims=s.dims,
        coords=coords,
        attrs=_attrs_out(s.attrs, s.fill_value),
    )
    return _with_indexes(_encode(da, s.fill_value))


_LAT = {"standard_name": "latitude", "units": "degrees_north"}
_LON = {"standard_name": "longitude", "units": "degrees_east"}


def _feature_type(p: GeoPoints) -> str:
    if "time" in p.dims:
        return "timeSeries"
    if p.z is not None:
        return "profile"
    if p.time is not None:
        return "trajectory"
    return "point"


def points_to_xarray(p: GeoPoints) -> Any:
    """A `GeoPoints` as a CF Discrete Sampling Geometry ``Dataset`` on ``obs``.

    With a `VariableAxis`, each variable is a data variable (its unit in
    ``units``); otherwise one variable, ``data``. Positions are ``lon`` /
    ``lat`` for a geographic CRS and ``x`` / ``y`` otherwise; ``z``, the
    per-point ``time`` and ``valid`` are coordinates on ``obs``.
    """
    host = p.to_numpy()
    geographic = p.crs is not None and p.crs.is_geographic
    xname, yname = ("lon", "lat") if geographic else ("x", "y")
    coords: dict[str, Any] = {
        GRID_MAPPING: _spatial_ref(p.crs, None),
        xname: ("obs", host.xy[:, 0], _LON if geographic else {"axis": "X"}),
        yname: ("obs", host.xy[:, 1], _LAT if geographic else {"axis": "Y"}),
    }
    if host.z is not None:
        v = p.vertical
        assert v is not None
        attrs = {"units": v.units, "positive": v.positive, "vertical_kind": v.kind}
        if v.kind in _VERTICAL_NAMES:
            attrs["standard_name"] = _VERTICAL_NAMES[v.kind]
        if v.datum is not None:
            attrs["datum"] = v.datum
        coords["z"] = ("obs", host.z, attrs)
    if host.time is not None:
        coords["time"] = ("obs", host.time)
    if host.valid is not None:
        coords["valid"] = ("obs", np.asarray(host.valid, dtype=bool))
    others = tuple(c for c in p.coords if not isinstance(c, VariableAxis))
    coords |= _coordinate_vars(others)
    data = np.asarray(host.data)
    variables: dict[str, Any] = {}
    axis = p.coord("variable") if "variable" in p.dims else None
    if isinstance(axis, VariableAxis):
        position = p.dims.index("variable")
        dims = tuple(d for d in p.dims if d != "variable")
        for k, name in enumerate(axis.names):
            attrs = {} if axis.units is None else {"units": axis.units[k]}
            if p.fill_value is not None:
                attrs["fill_value"] = p.fill_value
            values = np.take(data, k, axis=position)
            variables[name] = _encode(
                xr.DataArray(values, dims=dims, attrs=attrs), p.fill_value
            )
        layout = {"geoarray_variable_axis": position}
    else:
        attrs = {} if p.fill_value is None else {"fill_value": p.fill_value}
        variables["data"] = _encode(
            xr.DataArray(data, dims=p.dims, attrs=attrs), p.fill_value
        )
        layout = {}
    ds_attrs = {
        **_attrs_out(p.attrs, None),
        "featureType": _feature_type(p),
        "geoarray_carrier": "GeoPoints",
        **layout,
    }
    return xr.Dataset(variables, coords=coords, attrs=ds_attrs)


def to_xarray(carrier: Any) -> Any:
    """A carrier as xarray: a ``DataArray`` (grid, swath) or a ``Dataset`` (points).

    Raises:
        TypeError: Not a geoarray carrier.
        ValueError: A batched grid (one transform per item).
        GeoArrayTraceError: Inside ``jit`` / ``grad`` / ``vmap``.

    Example:
        >>> import numpy as np
        >>> from geoarray import GeoArray
        >>> g = GeoArray(np.zeros((2, 3)), (20, 0, 600000, 0, -20, 3550000), 32613,
        ...              dims="y x")
        >>> da = to_xarray(g)
        >>> da.coords["spatial_ref"].attrs["GeoTransform"]
        '600000.0 20.0 0.0 3550000.0 0.0 -20.0'
    """
    if isinstance(carrier, GeoArray):
        return grid_to_xarray(carrier)
    if isinstance(carrier, GeoSwath):
        return swath_to_xarray(carrier)
    if isinstance(carrier, GeoPoints):
        return points_to_xarray(carrier)
    raise TypeError(
        "to_xarray takes a GeoArray, GeoSwath or GeoPoints, got "
        f"{type(carrier).__name__}"
    )


# -- reading --------------------------------------------------------------------------


_HORIZONTAL_NAMES = (("y", "x"), ("lat", "lon"), ("latitude", "longitude"))
_LAT_NAMES = ("lat", "latitude")
_LON_NAMES = ("lon", "longitude")


def _find(names: tuple[str, ...], where: Any) -> str | None:
    for name in names:
        if name in where:
            return name
    return None


def read_kind(obj: Any) -> Kind | None:
    """The kind of an xarray ``DataArray`` (the readable types' kind reader).

    - **swath** — ``lat`` / ``lon`` are 2-D coordinates; the horizontal dims
      are theirs.
    - **grid** — two dims whose 1-D coordinates have CF ``axis`` ``Y`` and
      ``X``, else dims named ``y x``, ``lat lon`` or ``latitude longitude``.
    - **points** — ``lat`` / ``lon`` (or ``y`` / ``x``) are 1-D coordinates
      on one non-index dim.
    """
    if not isinstance(obj, xr.DataArray):
        return None
    coords = obj.coords
    lat, lon = _find(_LAT_NAMES, coords), _find(_LON_NAMES, coords)
    if lat and lon and coords[lat].ndim >= 2 and coords[lat].dims == coords[lon].dims:
        # (…, row, col): leading dims are batch dims, the last two horizontal
        return Kind("swath", tuple(map(str, coords[lat].dims[-2:])))
    by_axis = {
        str(c.attrs.get("axis", "")).upper(): str(c.dims[0])
        for c in coords.values()
        if c.ndim == 1 and c.dims[0] == c.name
    }
    if "Y" in by_axis and "X" in by_axis:
        return Kind("grid", (by_axis["Y"], by_axis["X"]))
    for pair in _HORIZONTAL_NAMES:
        if all(d in obj.dims for d in pair):
            return Kind("grid", pair)
    for ylat, xlon in ((lat, lon), ("y", "x")):
        if ylat and xlon and ylat in coords and xlon in coords:
            cy, cx = coords[ylat], coords[xlon]
            if cy.ndim == cx.ndim == 1 and cy.dims == cx.dims and cy.dims[0] != ylat:
                return Kind("points", (str(cy.dims[0]),))
    return None


def _grid_mapping(obj: Any) -> Any:
    name = (
        obj.encoding.get("grid_mapping")
        or obj.attrs.get("grid_mapping")
        or GRID_MAPPING
    )
    return obj.coords.get(name)


def _crs_of(obj: Any, ref: Any) -> pyproj.CRS | None:
    if ref is not None:
        for key in ("crs_wkt", "spatial_ref"):
            if key in ref.attrs:
                return pyproj.CRS.from_wkt(ref.attrs[key])
    if "crs" in obj.attrs:
        warnings.warn(
            "reading the CRS from attrs['crs'] (georeader's encoding) is "
            "deprecated: write a CF spatial_ref (geoarray.to_xarray)",
            DeprecationWarning,
            stacklevel=4,
        )
        return pyproj.CRS.from_user_input(obj.attrs["crs"])
    return None


def _fill_of(obj: Any) -> Any:
    for value in (
        obj.attrs.get("fill_value"),
        obj.encoding.get("_FillValue"),
        obj.attrs.get("_FillValue"),
    ):
        if value is not None:
            value = value.item() if isinstance(value, np.generic) else value
            if obj.dtype.kind == "b" and value in (0, 1):
                # a mask's fill, written as 0 / 1 (netCDF has no bool)
                return bool(value)
            return value
    return None


def _centres_transform(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """T from regular centre coordinates: the edge origin, not a centre (#463)."""
    if len(x) < 2 or len(y) < 2:
        raise ValueError(
            "from_xarray: no GeoTransform, and a one-pixel axis has no spacing to "
            "read from its coordinates; write a spatial_ref with a GeoTransform"
        )
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    for name, values, step in (("x", x, dx), ("y", y, dy)):
        if not np.allclose(np.diff(values), step, rtol=0, atol=_PIXEL_TOL * abs(step)):
            raise ValueError(
                f"from_xarray: {name} is not regular, so not an affine grid"
            )
    # c = x₀ − a/2, f = y₀ − e/2
    return np.array([dx, 0.0, float(x[0]) - dx / 2, 0.0, dy, float(y[0]) - dy / 2])


def _axis_centres(t: np.ndarray, width: int, height: int) -> tuple[Any, Any]:
    """Axis-aligned centre coordinates: xⱼ = T(j + ½, ·), yᵢ = T(·, i + ½)."""
    x, _ = apply_transform(t, np.arange(width) + 0.5, 0.5)
    _, y = apply_transform(t, 0.5, np.arange(height) + 0.5)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def _check_centres(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
    """GF3: centres must sit at T(j + ½, i + ½) to 10⁻⁹ of a pixel (#464)."""
    a, b, _, d, e, _ = t
    if b != 0 or d != 0:
        return
    ex, ey = _axis_centres(t, len(x), len(y))
    if np.max(np.abs(x - ex), initial=0.0) > _PIXEL_TOL * abs(a) or np.max(
        np.abs(y - ey), initial=0.0
    ) > _PIXEL_TOL * abs(e):
        raise GridMismatchError(
            "from_xarray: the x / y coordinates disagree with the GeoTransform by "
            "more than 1e-9 pixel (edge coordinates under a centre transform?)"
        )


def _coords_in(obj: Any, dims: tuple[str, ...]) -> tuple[Coordinate, ...]:
    out: list[Coordinate] = []
    c = obj.coords
    if "band" in dims and "band" in c:
        fields = {
            field: tuple(np.asarray(c[name].values, dtype=float).tolist())
            for field, name in SPECTRAL_FIELDS
            if name in c and c[name].dims == ("band",)
        }
        out.append(SpectralAxis(tuple(map(str, c["band"].values.tolist())), **fields))
    if "variable" in dims and "variable" in c:
        units = c["variable_units"].values.tolist() if "variable_units" in c else None
        out.append(VariableAxis(tuple(map(str, c["variable"].values.tolist())), units))
    if "level" in dims and "level" in c:
        attrs = c["level"].attrs
        kind = attrs.get("vertical_kind") or {
            v: k for k, v in _VERTICAL_NAMES.items()
        }.get(attrs.get("standard_name"), "level")
        vertical = Vertical(
            kind=kind,
            units=str(attrs.get("units", "1")),
            positive=str(attrs.get("positive", "up")),
            datum=attrs.get("datum"),
        )
        out.append(VerticalAxis(tuple(c["level"].values.tolist()), vertical))
    if "time" in dims and "time" in c and c["time"].dims == ("time",):
        values = np.asarray(c["time"].values)
        if values.dtype.kind == "M":
            out.append(TimeAxis(values))
    return tuple(out)


def _attrs_in(obj: Any) -> dict[str, Any]:
    return {k: v for k, v in obj.attrs.items() if k not in (*_OWN_ATTRS, "crs")}


def grid_from_xarray(da: Any) -> GeoArray:
    """A ``DataArray`` on an affine grid as a `GeoArray` (computing dask data)."""
    kind = read_kind(da)
    if kind is None or kind.kind != "grid":
        raise ValueError(
            f"from_xarray: dims {tuple(da.dims)} have no grid (y x, lat lon, or CF "
            "axis Y / X coordinates)"
        )
    ydim, xdim = kind.horizontal
    rest = [d for d in da.dims if d not in (ydim, xdim)]
    da = da.transpose(*rest, ydim, xdim)
    ref = _grid_mapping(da)
    gdal = None if ref is None else ref.attrs.get("GeoTransform")
    has_xy = (
        xdim in da.coords
        and ydim in da.coords
        and da.coords[xdim].ndim == 1
        and da.coords[ydim].ndim == 1
    )
    if has_xy:
        x = np.asarray(da.coords[xdim].values, dtype=np.float64)
        y = np.asarray(da.coords[ydim].values, dtype=np.float64)
    if gdal is not None:
        t = _from_gdal(gdal)
        if has_xy:
            _check_centres(t, x, y)
    elif has_xy:
        t = _centres_transform(x, y)
    else:
        raise ValueError(
            "from_xarray: no GeoTransform and no 1-D x / y coordinates to read a "
            "grid from"
        )
    crs = _crs_of(da, ref)
    dims = (*map(str, rest), *SPATIAL_DIMS)
    return GeoArray(
        np.asarray(da.data),
        t,
        crs,
        _fill_of(da),
        FrozenAttrs(_attrs_in(da)),
        dims=dims,
        coords=_coords_in(da, dims),
    )


def swath_from_xarray(da: Any) -> GeoSwath:
    """A ``DataArray`` with 2-D lat/lon as a `GeoSwath`."""
    kind = read_kind(da)
    if kind is None or kind.kind != "swath":
        raise ValueError("from_xarray: no 2-D lat / lon coordinates for a swath")
    rowdim, coldim = kind.horizontal
    lat = _find(_LAT_NAMES, da.coords)
    lon = _find(_LON_NAMES, da.coords)
    assert lat is not None and lon is not None
    rest = [d for d in da.dims if d not in (rowdim, coldim)]
    da = da.transpose(*rest, rowdim, coldim)
    rename = {rowdim: "row", coldim: "col"}
    dims = tuple(rename.get(str(d), str(d)) for d in da.dims)
    lat_da = da.coords[lat]
    lat_dims = tuple(lat_da.dims)
    ordered = [d for d in rest if d in lat_dims] + [rowdim, coldim]
    lat_values = np.asarray(lat_da.transpose(*ordered).values)
    lon_values = np.asarray(da.coords[lon].transpose(*ordered).values)
    ref = _grid_mapping(da)
    crs = _crs_of(da, ref) or pyproj.CRS.from_epsg(4326)
    return GeoSwath(
        np.asarray(da.data),
        lat_values,
        lon_values,
        dims=dims,
        coords=_coords_in(da, dims),
        crs=crs,
        fill_value=_fill_of(da),
        attrs=FrozenAttrs(_attrs_in(da)),
    )


def points_from_xarray(ds: Any) -> GeoPoints:
    """A CF DSG ``Dataset`` (as `points_to_xarray` writes it) as a `GeoPoints`."""
    if "obs" not in ds.dims:
        raise ValueError("from_xarray: a point Dataset has an obs dim")
    c = ds.coords
    xname = "lon" if "lon" in c else "x"
    yname = "lat" if "lat" in c else "y"
    xy = np.stack([np.asarray(c[xname].values), np.asarray(c[yname].values)], axis=-1)
    z = vertical = None
    if "z" in c:
        attrs = c["z"].attrs
        vertical = Vertical(
            kind=attrs.get("vertical_kind", "height"),
            units=str(attrs.get("units", "m")),
            positive=str(attrs.get("positive", "up")),
            datum=attrs.get("datum"),
        )
        z = np.asarray(c["z"].values)
    time = (
        np.asarray(c["time"].values)
        if "time" in c and c["time"].dims == ("obs",)
        else None
    )
    valid = np.asarray(c["valid"].values, dtype=bool) if "valid" in c else None
    names = list(ds.data_vars)
    position = ds.attrs.get("geoarray_variable_axis")
    if position is None and names == ["data"]:
        da = ds["data"]
        data = np.asarray(da.values)
        dims = tuple(map(str, da.dims))
        fill = _fill_of(da)
        coords = _coords_in(ds, dims)
    else:
        first = ds[names[0]]
        inner = tuple(map(str, first.dims))
        pos = int(position) if position is not None else 0
        data = np.stack([np.asarray(ds[n].values) for n in names], axis=pos)
        dims = (*inner[:pos], "variable", *inner[pos:])
        units = [ds[n].attrs.get("units") for n in names]
        fill = _fill_of(first)
        axis = VariableAxis(
            tuple(names), None if any(u is None for u in units) else tuple(units)
        )
        coords = (*_coords_in(ds, dims), axis)
    attrs = {
        k: v
        for k, v in ds.attrs.items()
        if k not in ("featureType", "geoarray_carrier", "geoarray_variable_axis")
    }
    ref = c.get(GRID_MAPPING)
    return GeoPoints(
        data,
        xy,
        z=z,
        vertical=vertical,
        time=time,
        valid=valid,
        dims=dims,
        coords=_ordered(coords, dims),
        crs=_crs_of(ds, ref),
        fill_value=fill,
        attrs=FrozenAttrs(attrs),
    )


def _ordered(
    coords: tuple[Coordinate, ...], dims: tuple[str, ...]
) -> tuple[Coordinate, ...]:
    return tuple(sorted(coords, key=lambda c: dims.index(c.dim)))


def from_xarray(obj: Any) -> Any:
    """A ``DataArray`` or a point ``Dataset`` as the matching geoarray carrier.

    A ``DataArray`` with 2-D lat/lon becomes a `GeoSwath`; one on an
    affine grid a `GeoArray` (its horizontal dims named ``y x``); a
    ``Dataset`` with an ``obs`` dim (a CF DSG) a `GeoPoints`. Lazy (dask)
    data is computed.

    Raises:
        ValueError: The object has no georeference geoarray can read, or a
            grid's coordinates are irregular.
        GridMismatchError: Centre coordinates disagree with the
            ``GeoTransform`` by more than 10⁻⁹ of a pixel.
    """
    if isinstance(obj, xr.Dataset):
        return points_from_xarray(obj)
    if not isinstance(obj, xr.DataArray):
        raise TypeError(
            f"from_xarray takes a DataArray or a Dataset, got {type(obj).__name__}"
        )
    kind = read_kind(obj)
    if kind is not None and kind.kind == "swath":
        return swath_from_xarray(obj)
    return grid_from_xarray(obj)


# -- several carriers -----------------------------------------------------------------


def to_dataset(carriers: Mapping[str, GeoArray]) -> Any:
    """Carriers on one grid as the variables of one ``Dataset`` (LL-F3).

    Raises:
        GridMismatchError: The carriers are on different grids; use
            `to_datatree` for several grids.
        ValueError: A shared non-spatial dim has different coordinates
            (``red`` here, ``nir`` there): xarray would align them.
    """
    from geoarray._src.geometry import _require_same_grid

    items = list(carriers.items())
    if not items:
        raise ValueError("to_dataset needs at least one carrier")
    _require_same_grid([g for _, g in items], "to_dataset")
    arrays = {name: grid_to_xarray(g) for name, g in items}
    try:
        # one Dataset shares coordinates: never let xarray align (union) them
        xr.align(*arrays.values(), join="exact", copy=False)
    except ValueError as exc:
        raise ValueError(
            "to_dataset: the carriers' coordinates differ (band names, levels, "
            "times, …), so xarray would align them and insert NaN; give them "
            f"one layout first, or use to_datatree: {exc}"
        ) from exc
    return xr.Dataset(arrays)


def from_dataset(ds: Any) -> dict[str, GeoArray]:
    """Each grid variable of a ``Dataset`` as a `GeoArray`, by name."""
    out = {}
    for name, da in ds.data_vars.items():
        if read_kind(da) is not None:
            out[str(name)] = from_xarray(da)
    return out


def to_datatree(groups: Mapping[str, GeoArray | Mapping[str, GeoArray]]) -> Any:
    """Several grids as one ``DataTree``: one node per grid (LL-F3).

    Args:
        groups: Node name → a carrier (stored under the node's name) or a
            mapping of carriers on that node's grid.

    Example:
        >>> import numpy as np
        >>> from geoarray import GeoArray
        >>> t10 = (10, 0, 600000, 0, -10, 3550000)
        >>> t20 = (20, 0, 600000, 0, -20, 3550000)
        >>> r10 = GeoArray(np.zeros((4, 4)), t10, 32613, dims="y x")
        >>> r20 = GeoArray(np.zeros((2, 2)), t20, 32613, dims="y x")
        >>> sorted(to_datatree({"r10m": r10, "r20m": r20}).children)
        ['r10m', 'r20m']
    """
    nodes = {}
    for name, value in groups.items():
        mapping: Mapping[str, GeoArray] = (
            {name: value} if isinstance(value, GeoArray) else value
        )
        nodes[f"/{name}"] = to_dataset(mapping)
    return xr.DataTree.from_dict(nodes)


def from_datatree(tree: Any) -> dict[str, dict[str, GeoArray]]:
    """Every node of a ``DataTree`` holding grids: node path → its carriers."""
    out = {}
    for node in tree.subtree:
        carriers = from_dataset(node.to_dataset())
        if carriers:
            out[node.path] = carriers
    return out


register_kind_reader(xr.DataArray, read_kind)
