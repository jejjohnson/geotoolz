"""Generic xarray geoprocessing — domain-agnostic operators.

This submodule hosts everything that applies across Earth-science
domains: coordinate validation, spatial/temporal subsetting, masks,
regridding-adjacent (interpolation), detrending, encoders, spectral
analysis, extremes, binning, and CRS utilities.

The public API is re-exported from :mod:`xr_toolz.geo._src`. For direct
access to a specific Layer-0 module, import e.g.
``xr_toolz.geo._src.detrend``. Layer-1 ``Operator`` wrappers live in
:mod:`xr_toolz.geo.operators`.

Evaluation metrics (``mse``, ``rmse``, …, ``psd_score``, ``find_intercept_1D``)
moved to :mod:`xr_toolz.metrics`. They remain importable from this
module for one release with a :class:`DeprecationWarning`.
"""

from __future__ import annotations

import warnings
from typing import Any

from xr_toolz.geo._src.crs import (
    assign_crs,
    calc_latlon,
    get_crs,
    lonlat_to_xy,
    reproject,
    xy_to_lonlat,
)
from xr_toolz.geo._src.detrend import (
    add_climatology,
    calculate_anomaly,
    calculate_anomaly_smoothed,
    calculate_climatology,
    calculate_climatology_season,
    calculate_climatology_smoothed,
    remove_climatology,
)
from xr_toolz.geo._src.discretize import (
    Grid,
    Period,
    SpaceTimeGrid,
    bin_2d,
    histogram_2d,
    points_to_grid,
)
from xr_toolz.geo._src.encoders import (
    cyclical_encode,
    encode_time_cyclical,
    encode_time_ordinal,
    fourier_features,
    lat_90_to_180,
    lat_180_to_90,
    lon_180_to_360,
    lon_360_to_180,
    positional_encoding,
    random_fourier_features,
    time_rescale,
    time_unrescale,
)
from xr_toolz.geo._src.extremes import (
    block_maxima,
    block_minima,
    pot_exceedances,
    pot_threshold,
    pp_counts,
    pp_stats,
)
from xr_toolz.geo._src.interpolate import (
    coarsen,
    fillnan_rbf,
    fillnan_spatial,
    fillnan_temporal,
    refine,
    resample_time,
)
from xr_toolz.geo._src.masks import (
    add_country_mask,
    add_land_mask,
    add_ocean_mask,
    apply_mask,
)
from xr_toolz.geo._src.subset import (
    select_variables,
    subset_bbox,
    subset_time,
    subset_where,
)
from xr_toolz.geo._src.validation import (
    rename_coords,
    validate_latitude,
    validate_longitude,
)


# Names moved to xr_toolz.metrics — kept importable for one release with
# a deprecation warning fired only on actual access (PEP 562).
_DEPRECATED_METRICS = {
    "bias": "xr_toolz.metrics._src.pixel",
    "correlation": "xr_toolz.metrics._src.pixel",
    "mae": "xr_toolz.metrics._src.pixel",
    "mse": "xr_toolz.metrics._src.pixel",
    "nrmse": "xr_toolz.metrics._src.pixel",
    "r2_score": "xr_toolz.metrics._src.pixel",
    "rmse": "xr_toolz.metrics._src.pixel",
    "find_intercept_1D": "xr_toolz.metrics._src.spectral",
    "psd_error": "xr_toolz.metrics._src.spectral",
    "psd_score": "xr_toolz.metrics._src.spectral",
    "resolved_scale": "xr_toolz.metrics._src.spectral",
}


def __getattr__(name: str) -> Any:
    if name in _DEPRECATED_METRICS:
        from importlib import import_module

        warnings.warn(
            f"xr_toolz.geo.{name} is deprecated; "
            f"import from xr_toolz.metrics instead. "
            f"This re-export will be removed in the next minor release.",
            DeprecationWarning,
            stacklevel=2,
        )
        module = import_module(_DEPRECATED_METRICS[name])
        return getattr(module, name)
    raise AttributeError(f"module 'xr_toolz.geo' has no attribute {name!r}")


__all__ = [
    "Grid",
    "Period",
    "SpaceTimeGrid",
    "add_climatology",
    "add_country_mask",
    "add_land_mask",
    "add_ocean_mask",
    "apply_mask",
    "assign_crs",
    "bin_2d",
    "block_maxima",
    "block_minima",
    "calc_latlon",
    "calculate_anomaly",
    "calculate_anomaly_smoothed",
    "calculate_climatology",
    "calculate_climatology_season",
    "calculate_climatology_smoothed",
    "coarsen",
    "cyclical_encode",
    "encode_time_cyclical",
    "encode_time_ordinal",
    "fillnan_rbf",
    "fillnan_spatial",
    "fillnan_temporal",
    "fourier_features",
    "get_crs",
    "histogram_2d",
    "lat_90_to_180",
    "lat_180_to_90",
    "lon_180_to_360",
    "lon_360_to_180",
    "lonlat_to_xy",
    "points_to_grid",
    "positional_encoding",
    "pot_exceedances",
    "pot_threshold",
    "pp_counts",
    "pp_stats",
    "random_fourier_features",
    "refine",
    "remove_climatology",
    "rename_coords",
    "reproject",
    "resample_time",
    "select_variables",
    "subset_bbox",
    "subset_time",
    "subset_where",
    "time_rescale",
    "time_unrescale",
    "validate_latitude",
    "validate_longitude",
    "xy_to_lonlat",
]
