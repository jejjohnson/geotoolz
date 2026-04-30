"""Grid-metrics helpers — derive cell widths / volumes from coordinates.

V4.4. Budget operators in :mod:`xr_toolz.budgets` take grid metrics as
explicit constructor arguments rather than guessing them from
coordinates. This module ships :func:`grid_metrics_from_coords`, the
*opt-in* helper that turns a Dataset's lon/lat (and optional depth)
coordinates into the two metric Datasets the budgets API consumes:

- ``volume_metrics`` carrying ``dx``, ``dy``, ``dz``, ``cell_area``,
  ``cell_volume``.
- ``face_metrics`` carrying ``dx_e``, ``dy_n``, ``dz_top``,
  ``area_e``, ``area_n``, ``area_top``.

The helper is the *only* place in xr_toolz that derives metrics from
coordinates. Anywhere downstream — budget primitives, residuals,
control-volume integrals — accepts pre-computed metric Datasets.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from xr_toolz.calc._src.constants import EARTH_RADIUS


def _cell_widths_from_coord(coord: xr.DataArray) -> xr.DataArray:
    """Centre-to-centre width per cell, with edge cells extrapolated."""
    values = np.asarray(coord.values, dtype=float)
    if values.size < 2:
        raise ValueError(
            f"Coord {coord.name!r} has size {values.size}; need at least 2."
        )
    diffs = np.diff(values)
    widths = np.empty_like(values)
    widths[1:-1] = 0.5 * (diffs[:-1] + diffs[1:])
    widths[0] = diffs[0]
    widths[-1] = diffs[-1]
    return xr.DataArray(np.abs(widths), dims=(coord.name,), coords={coord.name: coord})


def _face_widths_from_centres(coord: xr.DataArray) -> xr.DataArray:
    """Spacing between adjacent cell centres; edge face copies its neighbour."""
    values = np.asarray(coord.values, dtype=float)
    diffs = np.abs(np.diff(values))
    face = np.empty_like(values)
    face[:-1] = diffs
    face[-1] = diffs[-1]
    return xr.DataArray(face, dims=(coord.name,), coords={coord.name: coord})


def grid_metrics_from_coords(
    ds: xr.Dataset,
    *,
    lat: str = "lat",
    lon: str = "lon",
    depth: str | None = None,
    sphere: bool = True,
    radius: float = EARTH_RADIUS,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Derive ``(volume_metrics, face_metrics)`` from a Dataset's coords.

    Args:
        ds: Dataset whose coordinates define the grid.
        lat, lon: Names of latitude / longitude coordinates. Spherical
            mode assumes degrees.
        depth: Optional vertical coordinate name. If ``None``, ``dz``
            is omitted from ``volume_metrics`` and ``cell_volume`` is
            populated with the 2-D ``cell_area`` (in m², not m³) so
            downstream code that always reads ``cell_volume`` keeps
            working in 2-D mode.
        sphere: If ``True`` (default), apply spherical metric
            ``dx = R cos(φ) Δλ``, ``dy = R Δφ`` with degrees-to-radians
            conversion. If ``False``, treat ``lon``/``lat`` as Cartesian
            with units already in metres.
        radius: Earth radius (m), used in spherical mode.

    Returns:
        ``(volume_metrics, face_metrics)`` — two Datasets with the
        keys documented in the module docstring.

    Notes:
        Edge cells extrapolate the nearest interior spacing; this is
        adequate for budget closure tests and the demo notebook but
        will bias real-world budgets near the domain boundary.
    """
    if lat not in ds.coords:
        raise ValueError(f"Dataset is missing latitude coord {lat!r}.")
    if lon not in ds.coords:
        raise ValueError(f"Dataset is missing longitude coord {lon!r}.")
    lat_c = ds[lat]
    lon_c = ds[lon]
    if sphere:
        rad = float(np.pi / 180.0)
        dlat_rad = _cell_widths_from_coord(lat_c) * rad
        dlon_rad = _cell_widths_from_coord(lon_c) * rad
        dlat_face_rad = _face_widths_from_centres(lat_c) * rad
        dlon_face_rad = _face_widths_from_centres(lon_c) * rad
        cos_phi = np.cos(lat_c * rad)
        dx = (radius * cos_phi * dlon_rad).rename("dx")
        dy = (radius * dlat_rad).rename("dy")
        dx_e = (radius * cos_phi * dlon_face_rad).rename("dx_e")
        dy_n = (radius * dlat_face_rad).rename("dy_n")
    else:
        dx = _cell_widths_from_coord(lon_c).rename("dx")
        dy = _cell_widths_from_coord(lat_c).rename("dy")
        dx_e = _face_widths_from_centres(lon_c).rename("dx_e")
        dy_n = _face_widths_from_centres(lat_c).rename("dy_n")

    cell_area = (dx * dy).rename("cell_area")
    area_e = (dx_e * dy).rename("area_e")
    area_n = (dx * dy_n).rename("area_n")

    vol_data: dict[str, xr.DataArray] = {
        "dx": dx,
        "dy": dy,
        "cell_area": cell_area,
    }
    face_data: dict[str, xr.DataArray] = {
        "dx_e": dx_e,
        "dy_n": dy_n,
        "area_e": area_e,
        "area_n": area_n,
    }

    if depth is not None:
        if depth not in ds.coords:
            raise ValueError(f"Dataset is missing depth coord {depth!r}.")
        dz = _cell_widths_from_coord(ds[depth]).rename("dz")
        dz_top = _face_widths_from_centres(ds[depth]).rename("dz_top")
        cell_volume = (cell_area * dz).rename("cell_volume")
        area_top = cell_area.rename("area_top")
        vol_data["dz"] = dz
        vol_data["cell_volume"] = cell_volume
        face_data["dz_top"] = dz_top
        face_data["area_top"] = area_top
    else:
        vol_data["cell_volume"] = cell_area.rename("cell_volume")

    return xr.Dataset(vol_data), xr.Dataset(face_data)


__all__ = ["grid_metrics_from_coords"]
