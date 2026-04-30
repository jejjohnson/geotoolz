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
from xr_toolz.geo._src.extremes import (
    block_maxima,
    block_minima,
    pot_exceedances,
    pot_threshold,
    pp_counts,
    pp_stats,
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

# Encoders moved to xr_toolz.transforms.encoders (D8). Kept importable for
# one release with a deprecation warning fired only on actual access.
_DEPRECATED_ENCODERS = {
    "cyclical_encode": "xr_toolz.transforms._src.encoders.basis",
    "fourier_features": "xr_toolz.transforms._src.encoders.basis",
    "positional_encoding": "xr_toolz.transforms._src.encoders.basis",
    "random_fourier_features": "xr_toolz.transforms._src.encoders.basis",
    "lat_90_to_180": "xr_toolz.transforms._src.encoders.coord_space",
    "lat_180_to_90": "xr_toolz.transforms._src.encoders.coord_space",
    "lon_180_to_360": "xr_toolz.transforms._src.encoders.coord_space",
    "lon_360_to_180": "xr_toolz.transforms._src.encoders.coord_space",
    "encode_time_cyclical": "xr_toolz.transforms._src.encoders.coord_time",
    "encode_time_ordinal": "xr_toolz.transforms._src.encoders.coord_time",
    "time_rescale": "xr_toolz.transforms._src.encoders.coord_time",
    "time_unrescale": "xr_toolz.transforms._src.encoders.coord_time",
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
    if name in _DEPRECATED_ENCODERS:
        from importlib import import_module

        warnings.warn(
            f"xr_toolz.geo.{name} is deprecated; "
            f"import from xr_toolz.transforms.encoders instead. "
            f"This re-export will be removed in the next minor release.",
            DeprecationWarning,
            stacklevel=2,
        )
        module = import_module(_DEPRECATED_ENCODERS[name])
        return getattr(module, name)
    raise AttributeError(f"module 'xr_toolz.geo' has no attribute {name!r}")


__all__ = [
    "add_climatology",
    "add_country_mask",
    "add_land_mask",
    "add_ocean_mask",
    "apply_mask",
    "assign_crs",
    "block_maxima",
    "block_minima",
    "calc_latlon",
    "calculate_anomaly",
    "calculate_anomaly_smoothed",
    "calculate_climatology",
    "calculate_climatology_season",
    "calculate_climatology_smoothed",
    "get_crs",
    "lonlat_to_xy",
    "pot_exceedances",
    "pot_threshold",
    "pp_counts",
    "pp_stats",
    "remove_climatology",
    "rename_coords",
    "reproject",
    "select_variables",
    "subset_bbox",
    "subset_time",
    "subset_where",
    "validate_latitude",
    "validate_longitude",
    "xy_to_lonlat",
]
