"""Coordinate reference system utilities.

Thin wrappers around :mod:`pyproj` and :mod:`rioxarray`. For full
reprojection / raster I/O, delegate to ``rioxarray`` directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import rioxarray  # noqa: F401  — needed so ds.rio is populated
import xarray as xr
from pyproj import CRS, Transformer


GridResolution = Literal["one_degree", "quarter_degree", "twelfth_degree", "other"]


def assign_crs(ds: xr.Dataset, crs: str = "EPSG:4326") -> xr.Dataset:
    """Attach a CRS to ``ds`` via :mod:`rioxarray`.

    Args:
        ds: Input dataset.
        crs: Any CRS specifier accepted by :class:`pyproj.CRS`.

    Returns:
        Dataset with the CRS attached (``ds.rio.crs``).
    """
    return ds.rio.write_crs(crs)


def get_crs(ds: xr.Dataset) -> CRS | None:
    """Return ``ds.rio.crs`` if set, otherwise ``None``."""
    crs = ds.rio.crs
    return CRS(crs) if crs is not None else None


def reproject(
    ds: xr.Dataset,
    target_crs: str,
    resolution: float | None = None,
    resampling: str = "bilinear",
) -> xr.Dataset:
    """Reproject a raster dataset to ``target_crs`` via :mod:`rioxarray`.

    Args:
        ds: Input dataset with a CRS attached.
        target_crs: Any CRS specifier accepted by :class:`pyproj.CRS`.
        resolution: If provided, resample to this output cell size.
        resampling: Name of the ``rasterio.enums.Resampling`` member.

    Returns:
        Reprojected dataset.
    """
    from rasterio.enums import Resampling

    if not hasattr(Resampling, resampling):
        valid = [r.name for r in Resampling]
        raise ValueError(f"Unknown resampling {resampling!r}; expected one of {valid}.")
    return ds.rio.reproject(
        target_crs,
        resolution=resolution,
        resampling=getattr(Resampling, resampling),
    )


def lonlat_to_xy(
    crs: str,
    lon: Sequence[float] | np.ndarray,
    lat: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert WGS-84 lon/lat to ``crs`` x/y coordinates."""
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.asarray(x), np.asarray(y)


def xy_to_lonlat(
    crs: str,
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert ``crs`` x/y coordinates back to WGS-84 lon/lat."""
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return np.asarray(lon), np.asarray(lat)


def calc_latlon(ds: xr.Dataset) -> xr.Dataset:
    """Add 2-D ``latitude`` / ``longitude`` coords computed from x/y.

    Assumes ``ds`` has 1-D ``x`` and ``y`` coordinates and a CRS attached
    (``ds.rio.crs``). Inf values from the transform are replaced with
    NaN so that downstream masking handles them cleanly.

    Args:
        ds: Input dataset.

    Returns:
        Dataset with additional 2-D ``latitude`` / ``longitude``
        coordinates along ``(y, x)``.
    """
    if ds.rio.crs is None:
        raise ValueError(
            "assign a CRS with assign_crs(ds, ...) before calling calc_latlon."
        )

    xx, yy = np.meshgrid(ds.x.values, ds.y.values)
    lons, lats = xy_to_lonlat(str(ds.rio.crs), xx, yy)
    lons = np.where(np.isfinite(lons), lons, np.nan)
    lats = np.where(np.isfinite(lats), lats, np.nan)

    ds = ds.assign_coords(
        longitude=(("y", "x"), lons),
        latitude=(("y", "x"), lats),
    )
    ds["longitude"].attrs["units"] = "degrees_east"
    ds["latitude"].attrs["units"] = "degrees_north"
    return ds


def get_dataset_resolution(
    ds: xr.Dataset,
    *,
    lon: str = "lon",
    lat: str = "lat",
    rtol: float = 0.05,
) -> GridResolution:
    """Classify a regular lat/lon grid against canonical OceanBench resolutions.

    Returns one of ``"one_degree"``, ``"quarter_degree"``,
    ``"twelfth_degree"``, or ``"other"`` (no canonical resolution
    matches within ``rtol`` of *both* axes). Useful for dispatch — e.g.
    picking the right MDT URL or the right reference dataset for a
    given grid.

    The classifier checks the median absolute spacing of the ``lon``
    and ``lat`` coordinates **independently** — an anisotropic grid
    (e.g. ``dlon=1.1`` but ``dlat=0.9``) is rejected even though its
    averaged spacing would round to 1°. Curvilinear (2-D) lon/lat
    coordinates are rejected with a ``ValueError``: ``np.diff`` on a
    multi-dim array would silently mix endpoints across rows and
    misclassify the grid.

    Args:
        ds: Dataset carrying 1-D ``lon`` / ``lat`` coordinates.
        lon: Longitude coordinate name.
        lat: Latitude coordinate name.
        rtol: Relative tolerance for the canonical-resolution match,
            applied per-axis.

    Returns:
        Resolution label as a ``Literal[...]`` string.

    Raises:
        ValueError: If ``ds[lon]`` or ``ds[lat]`` is not 1-D — the
            classifier only operates on rectilinear grids; curvilinear
            datasets should be regridded first.
    """
    lon_da = ds[lon]
    lat_da = ds[lat]
    if lon_da.ndim != 1:
        raise ValueError(
            f"get_dataset_resolution expects a 1-D {lon!r} coord, got "
            f"shape {tuple(lon_da.shape)}. Curvilinear / 2-D grids are not "
            "supported — regrid onto a rectilinear axis first."
        )
    if lat_da.ndim != 1:
        raise ValueError(
            f"get_dataset_resolution expects a 1-D {lat!r} coord, got "
            f"shape {tuple(lat_da.shape)}. Curvilinear / 2-D grids are not "
            "supported — regrid onto a rectilinear axis first."
        )
    dlon = float(np.median(np.abs(np.diff(np.asarray(lon_da.values, dtype=float)))))
    dlat = float(np.median(np.abs(np.diff(np.asarray(lat_da.values, dtype=float)))))
    candidates: list[tuple[GridResolution, float]] = [
        ("one_degree", 1.0),
        ("quarter_degree", 0.25),
        ("twelfth_degree", 1.0 / 12.0),
    ]
    for label, target in candidates:
        # Require BOTH axes to match the canonical target — averaging
        # would let an anisotropic grid (dlon=1.1, dlat=0.9 → mean=1.0)
        # masquerade as canonical even though neither axis is.
        if abs(dlon - target) < rtol * target and abs(dlat - target) < rtol * target:
            return label
    return "other"
