"""Generic xarray geoprocessing — domain-agnostic operators.

This submodule hosts everything that applies across Earth-science
domains: coordinate validation, spatial/temporal subsetting, masks,
regridding-adjacent (interpolation), detrending, encoders, spectral
analysis, extremes, binning, and CRS utilities.

The public API is re-exported from :mod:`xrtoolz.geo._src`. For direct
access to a specific Layer-0 module, import e.g.
``xrtoolz.geo._src.detrend``. Layer-1 ``Operator`` wrappers live in
:mod:`xrtoolz.geo.operators`.

Evaluation metrics (``mse``, ``rmse``, …, ``psd_score``, ``find_intercept_1D``)
moved to :mod:`xrtoolz.metrics`. They remain importable from this
module for one release with a :class:`DeprecationWarning`.
"""

from __future__ import annotations

import warnings
from typing import Any

from xrtoolz.geo._src.along_track import bandpass_wavelength, median_dx_km
from xrtoolz.geo._src.crs import (
    assign_crs,
    calc_latlon,
    get_crs,
    get_dataset_resolution,
    lonlat_to_xy,
    reproject,
    xy_to_lonlat,
)
from xrtoolz.geo._src.detrend import (
    add_climatology,
    calculate_anomaly,
    calculate_anomaly_smoothed,
    calculate_climatology,
    calculate_climatology_season,
    calculate_climatology_smoothed,
    remove_climatology,
    remove_mean,
)
from xrtoolz.geo._src.extremes import (
    block_maxima,
    block_minima,
    pot_exceedances,
    pot_threshold,
    pp_counts,
    pp_stats,
)
from xrtoolz.geo._src.masks import (
    add_country_mask,
    add_land_mask,
    add_ocean_mask,
    apply_mask,
)
from xrtoolz.geo._src.regions import (
    REGIONS,
    RegionSpec,
    bbox_region,
    custom_region,
    load_region_file,
    polygon_from_geojson,
    region_from_dict,
    region_to_dict,
    resolve_region,
)
from xrtoolz.geo._src.subset import (
    select_variables,
    subset_bbox,
    subset_time,
    subset_to_region,
    subset_where,
)
from xrtoolz.geo._src.validation import (
    check_dataset_coords,
    decode_cf_time,
    rename_coords,
    rename_from_cf_standard_names,
    rename_to_cf_standard_names,
    rename_variables,
    validate_latitude,
    validate_longitude,
    validate_time,
)
from xrtoolz.geo._src.wavelet import (
    cwt2,
    morlet2_ft,
    wvlt_cross_spectrum,
    wvlt_power_spectrum,
)
from xrtoolz.geo._src.wavelet1d import (
    cwt1d,
    dominant_period_map,
    icwt1d,
    wavelet_significance,
)
from xrtoolz.geo._src.wavelet_utils import (
    build_coi_mask,
    geometric_scales,
    scale_to_wavenumber,
    wavenumber_to_scale,
)
from xrtoolz.geo.operators import (
    BandpassWavelength,
    RenameFromCFStandardNames,
    RenameToCFStandardNames,
    WaveletPowerSpectrum,
    WaveletScalogram,
    WaveletSignificance,
)


# Names moved to xrtoolz.metrics — kept importable for one release with
# a deprecation warning fired only on actual access (PEP 562).
_DEPRECATED_METRICS = {
    "bias": "xrtoolz.metrics._src.pixel",
    "correlation": "xrtoolz.metrics._src.pixel",
    "mae": "xrtoolz.metrics._src.pixel",
    "mse": "xrtoolz.metrics._src.pixel",
    "nrmse": "xrtoolz.metrics._src.pixel",
    "r2_score": "xrtoolz.metrics._src.pixel",
    "rmse": "xrtoolz.metrics._src.pixel",
    "find_intercept_1D": "xrtoolz.metrics._src.spectral",
    "psd_error": "xrtoolz.metrics._src.spectral",
    "psd_score": "xrtoolz.metrics._src.spectral",
    "resolved_scale": "xrtoolz.metrics._src.spectral",
}

# Encoders moved to xrtoolz.transforms.encoders (D8). Kept importable for
# one release with a deprecation warning fired only on actual access.
_DEPRECATED_ENCODERS = {
    "cyclical_encode": "xrtoolz.transforms._src.encoders.basis",
    "fourier_features": "xrtoolz.transforms._src.encoders.basis",
    "positional_encoding": "xrtoolz.transforms._src.encoders.basis",
    "random_fourier_features": "xrtoolz.transforms._src.encoders.basis",
    "lat_90_to_180": "xrtoolz.transforms._src.encoders.coord_space",
    "lat_180_to_90": "xrtoolz.transforms._src.encoders.coord_space",
    "lon_180_to_360": "xrtoolz.transforms._src.encoders.coord_space",
    "lon_360_to_180": "xrtoolz.transforms._src.encoders.coord_space",
    "encode_time_cyclical": "xrtoolz.transforms._src.encoders.coord_time",
    "encode_time_ordinal": "xrtoolz.transforms._src.encoders.coord_time",
    "time_rescale": "xrtoolz.transforms._src.encoders.coord_time",
    "time_unrescale": "xrtoolz.transforms._src.encoders.coord_time",
}


def __getattr__(name: str) -> Any:
    if name in _DEPRECATED_METRICS:
        from importlib import import_module

        warnings.warn(
            f"xrtoolz.geo.{name} is deprecated; "
            f"import from xrtoolz.metrics instead. "
            f"This re-export will be removed in the next minor release.",
            DeprecationWarning,
            stacklevel=2,
        )
        module = import_module(_DEPRECATED_METRICS[name])
        return getattr(module, name)
    if name in _DEPRECATED_ENCODERS:
        from importlib import import_module

        warnings.warn(
            f"xrtoolz.geo.{name} is deprecated; "
            f"import from xrtoolz.transforms.encoders instead. "
            f"This re-export will be removed in the next minor release.",
            DeprecationWarning,
            stacklevel=2,
        )
        module = import_module(_DEPRECATED_ENCODERS[name])
        return getattr(module, name)
    raise AttributeError(f"module 'xrtoolz.geo' has no attribute {name!r}")


__all__ = [
    "REGIONS",
    "BandpassWavelength",
    "RegionSpec",
    "RenameFromCFStandardNames",
    "RenameToCFStandardNames",
    "WaveletPowerSpectrum",
    "WaveletScalogram",
    "WaveletSignificance",
    "add_climatology",
    "add_country_mask",
    "add_land_mask",
    "add_ocean_mask",
    "apply_mask",
    "assign_crs",
    "bandpass_wavelength",
    "bbox_region",
    "block_maxima",
    "block_minima",
    "build_coi_mask",
    "calc_latlon",
    "calculate_anomaly",
    "calculate_anomaly_smoothed",
    "calculate_climatology",
    "calculate_climatology_season",
    "calculate_climatology_smoothed",
    "check_dataset_coords",
    "custom_region",
    "cwt1d",
    "cwt2",
    "decode_cf_time",
    "dominant_period_map",
    "geometric_scales",
    "get_crs",
    "get_dataset_resolution",
    "icwt1d",
    "load_region_file",
    "lonlat_to_xy",
    "median_dx_km",
    "morlet2_ft",
    "polygon_from_geojson",
    "pot_exceedances",
    "pot_threshold",
    "pp_counts",
    "pp_stats",
    "region_from_dict",
    "region_to_dict",
    "remove_climatology",
    "remove_mean",
    "rename_coords",
    "rename_from_cf_standard_names",
    "rename_to_cf_standard_names",
    "rename_variables",
    "reproject",
    "resolve_region",
    "scale_to_wavenumber",
    "select_variables",
    "subset_bbox",
    "subset_time",
    "subset_to_region",
    "subset_where",
    "validate_latitude",
    "validate_longitude",
    "validate_time",
    "wavelet_significance",
    "wavenumber_to_scale",
    "wvlt_cross_spectrum",
    "wvlt_power_spectrum",
    "xy_to_lonlat",
]
