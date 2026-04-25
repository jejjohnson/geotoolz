"""Generic xarray geoprocessing — domain-agnostic operators.

This submodule hosts everything that applies across Earth-science
domains: coordinate validation, spatial/temporal subsetting, masks,
regridding-adjacent (interpolation), detrending, encoders, spectral
analysis, pixel + spectral metrics, extremes, binning, and CRS
utilities.

The public API is re-exported from :mod:`xr_toolz.geo._src`. For direct
access to a specific Layer-0 module, import e.g.
``xr_toolz.geo._src.detrend``. Layer-1 ``Operator`` wrappers live in
:mod:`xr_toolz.geo.operators`.
"""

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
from xr_toolz.geo._src.metrics import (
    bias,
    correlation,
    find_intercept_1D,
    mae,
    mse,
    nrmse,
    psd_error,
    psd_score,
    r2_score,
    resolved_scale,
    rmse,
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
    "bias",
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
    "correlation",
    "cyclical_encode",
    "encode_time_cyclical",
    "encode_time_ordinal",
    "fillnan_rbf",
    "fillnan_spatial",
    "fillnan_temporal",
    "find_intercept_1D",
    "fourier_features",
    "get_crs",
    "histogram_2d",
    "lat_90_to_180",
    "lat_180_to_90",
    "lon_180_to_360",
    "lon_360_to_180",
    "lonlat_to_xy",
    "mae",
    "mse",
    "nrmse",
    "points_to_grid",
    "positional_encoding",
    "pot_exceedances",
    "pot_threshold",
    "pp_counts",
    "pp_stats",
    "psd_error",
    "psd_score",
    "r2_score",
    "random_fourier_features",
    "refine",
    "remove_climatology",
    "rename_coords",
    "reproject",
    "resample_time",
    "resolved_scale",
    "rmse",
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
