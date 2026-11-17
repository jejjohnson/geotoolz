"""Helpers shared by the scene-style (EMIT / AVIRIS-3 / GHGSat) openers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from xrproducts._src.local.errors import ProductGridError
from xrproducts.types import CH4_ENHANCEMENT, CH4_SENSITIVITY, apply_cf_attrs


_COORD_RENAMES = {"latitude": "lat", "longitude": "lon"}
#: Raw raster dims of the EMIT products (and its lookup table) -> ``y`` / ``x``.
RASTER_DIMS = {"downtrack": "y", "crosstrack": "x", "ortho_y": "y", "ortho_x": "x"}
#: Name of the per-pixel matched-filter uncertainty raster.
UNCERTAINTY = "ch4_uncertainty"
#: No-data value of the EMIT and AVIRIS-3 L2B greenhouse-gas rasters.
PRODUCT_FILL = -9999.0
#: Largest coordinate difference two rasters on one grid may show.
GRID_ATOL = 1e-6
_GRID_COORDS = ("lon", "lat", "x", "y")
_UNCERTAINTY_ATTRS = {
    "long_name": "Methane enhancement uncertainty (1-sigma)",
    "units": "ppm m",
    # SPIIRE's cost treats u as the standard deviation of the enhancement
    # divided by the sensitivity ("truth"); the product user guides do not
    # say which quantity it is, so this default is unverified.
    "uncertainty_of": "truth",
}


def promote_lonlat(ds: xr.Dataset) -> xr.Dataset:
    """Rename ``latitude``/``longitude`` to ``lat``/``lon`` and make them coords.

    Args:
        ds: Dataset whose geolocation may sit in data variables under
            either the CF long names or the short ``lat``/``lon`` names.

    Returns:
        Dataset with ``lon``/``lat`` as coordinates (when present).
    """
    ds = ds.rename({k: v for k, v in _COORD_RENAMES.items() if k in ds.variables})
    promote = [n for n in ("lon", "lat") if n in ds.data_vars]
    return ds.set_coords(promote) if promote else ds


def attach_scene_time(ds: xr.Dataset) -> xr.Dataset:
    """Give a single-scene Dataset a length-1 ``time`` dimension.

    The scene timestamp is taken from a scalar ``time`` variable when the
    file carries one, otherwise from the ``time_coverage_start`` global
    attribute. Datasets that already have a ``time`` dimension, or expose
    neither, are returned unchanged. The length-1 axis is what lets
    :meth:`~xrproducts.LocalL2Source.open` concatenate several scenes
    along ``time``.

    Args:
        ds: Scene dataset on ``(y, x)``.

    Returns:
        Dataset with a ``time`` dimension of length 1 when a timestamp
        could be found.
    """
    if "time" in ds.dims:
        return ds
    if "time" in ds.variables and ds["time"].ndim == 0:
        return ds.set_coords("time").expand_dims("time")
    start = ds.attrs.get("time_coverage_start")
    if start is None:
        return ds
    ts = pd.Timestamp(start)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ds.expand_dims(time=[ts.to_datetime64()])


def open_raster(path: str | Path) -> xr.Dataset:
    """Open a single-band raster NetCDF on ``(y, x)``.

    EMIT's raw dims are renamed to ``y`` / ``x``, a length-1 ``band`` axis
    (what ``rioxarray`` writes for a GeoTIFF) is squeezed out, and
    ``latitude`` / ``longitude`` become ``lat`` / ``lon`` coordinates.

    Args:
        path: NetCDF raster.

    Returns:
        The raster Dataset.
    """
    ds = xr.open_dataset(path)
    ds = ds.rename({k: v for k, v in RASTER_DIMS.items() if k in ds.dims})
    if ds.sizes.get("band") == 1:
        ds = ds.squeeze("band", drop=True)
    return promote_lonlat(ds)


def pick_raster(ds: xr.Dataset, names: Sequence[str], path: str | Path) -> str:
    """Name of the raster variable in ``ds``.

    The first of ``names`` present wins; failing that, the only data
    variable on ``(y, x)`` (a converted GeoTIFF holds one band under a
    generic name).

    Args:
        ds: Raster Dataset.
        names: Accepted variable names, in order of preference.
        path: The file, for the error message.

    Returns:
        The variable name.

    Raises:
        KeyError: If none of ``names`` is present and the file has no
            single ``(y, x)`` variable.
    """
    found = next((n for n in names if n in ds.data_vars), None)
    if found is not None:
        return found
    spatial = [str(n) for n, da in ds.data_vars.items() if {"y", "x"} <= set(da.dims)]
    if len(spatial) == 1:
        return spatial[0]
    raise KeyError(
        f"{path!s} has none of {tuple(names)!r} and {len(spatial)} (y, x) "
        f"variables ({spatial!r}); expected exactly one"
    )


def require_same_grid(scene: xr.Dataset, other: xr.Dataset, path: str | Path) -> None:
    """Check that ``other`` is on the grid of ``scene``.

    The ``(y, x)`` shapes must match, and every coordinate among ``lon``,
    ``lat``, ``x`` and ``y`` that both carry must agree to within
    :data:`GRID_ATOL`.

    Args:
        scene: The enhancement raster.
        other: A companion raster.
        path: The companion's file, named in the error.

    Raises:
        ProductGridError: If the shapes or the coordinates differ.
    """
    shape = (scene.sizes.get("y"), scene.sizes.get("x"))
    other_shape = (other.sizes.get("y"), other.sizes.get("x"))
    if shape != other_shape:
        raise ProductGridError(
            f"{path!s} is on a {other_shape[0]} by {other_shape[1]} (y, x) grid; "
            f"the enhancement is on {shape[0]} by {shape[1]}"
        )
    for name in _GRID_COORDS:
        if name not in scene.variables or name not in other.variables:
            continue
        ref = np.asarray(scene[name].values, dtype=float)
        val = np.asarray(other[name].values, dtype=float)
        if ref.shape != val.shape or not np.allclose(
            ref, val, rtol=0.0, atol=GRID_ATOL, equal_nan=True
        ):
            raise ProductGridError(
                f"{path!s}: {name!r} differs from the enhancement's by more "
                f"than {GRID_ATOL}"
            )


def attach_companions(
    ds: xr.Dataset,
    *,
    sensitivity_path: str | Path | None,
    uncertainty_path: str | Path | None,
) -> xr.Dataset:
    """Add the sensitivity and uncertainty rasters to the enhancement scene.

    Each companion is read with :func:`open_raster`, its variable found
    with :func:`pick_raster` (``ch4_sensitivity`` / ``ch4_uncertainty``
    or the file's only ``(y, x)`` variable), checked against the scene's
    grid and stored as stored: no cast, no masking, no resampling. A
    companion ``ch4_uncertainty`` replaces one the scene file carries.

    Args:
        ds: Enhancement scene on ``(y, x)``.
        sensitivity_path: Sensitivity raster, or ``None``.
        uncertainty_path: Uncertainty raster, or ``None``.

    Returns:
        The scene with ``ch4_sensitivity`` / ``ch4_uncertainty`` added.

    Raises:
        ProductGridError: If a companion is not on the scene's grid.
        KeyError: If a companion has no recognisable raster variable.
    """
    companions = (
        (CH4_SENSITIVITY.name, sensitivity_path),
        (UNCERTAINTY, uncertainty_path),
    )
    for name, path in companions:
        if path is None:
            continue
        other = open_raster(path)
        raw = pick_raster(other, (name,), path)
        require_same_grid(promote_lonlat(ds), other, path)
        da = other[raw]
        ds[name] = xr.Variable(da.dims, da.values, dict(da.attrs))
    return ds


def as_float(da: xr.DataArray) -> xr.DataArray:
    """``da`` with a floating dtype: floats are kept, integers become float64."""
    if np.issubdtype(da.dtype, np.floating):
        return da
    return da.astype(np.float64)


def fill_to_nan(da: xr.DataArray, fill: float) -> xr.DataArray:
    """Replace ``fill`` by ``NaN`` and record it as ``source_fill_value``.

    Args:
        da: Floating-point raster.
        fill: The product's no-data value.

    Returns:
        The masked raster.
    """
    out = da.where(da != fill)
    out.attrs = {**da.attrs, "source_fill_value": fill}
    return out


def finish_rasters(ds: xr.Dataset, *, flare_value: float | None = None) -> xr.Dataset:
    """Float dtype, no-data as ``NaN`` and CF attributes on the CH4 rasters.

    Applies to ``ch4_enhancement``, ``ch4_sensitivity`` and
    ``ch4_uncertainty`` where present. The dtype step comes first and is
    separate from the masking: :data:`PRODUCT_FILL` becomes ``NaN`` in all
    three, and ``flare_value`` (AVIRIS-3 marks flares with ``-1`` in its
    sensitivity and uncertainty) becomes ``NaN`` in the two companions.

    Args:
        ds: Scene Dataset.
        flare_value: Flare marker of the companions, or ``None``.

    Returns:
        The scene with the three rasters finished.
    """
    for name in (CH4_ENHANCEMENT.name, CH4_SENSITIVITY.name, UNCERTAINTY):
        if name not in ds.data_vars:
            continue
        da = as_float(ds[name])
        da = fill_to_nan(da, PRODUCT_FILL)
        if flare_value is not None and name != CH4_ENHANCEMENT.name:
            attrs = {**da.attrs, "flare_value": flare_value}
            da = da.where(da != flare_value)
            da.attrs = attrs
        if name == UNCERTAINTY:
            da.attrs = {**_UNCERTAINTY_ATTRS, **da.attrs}
        else:
            da = apply_cf_attrs(da, name)
        ds[name] = da
    return ds
