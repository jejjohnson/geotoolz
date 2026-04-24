"""Curated AEMET preset catalogue.

AEMET doesn't expose a ``dataset_id`` the way CMEMS does — every product
is addressed by endpoint path + parameters. The presets here play the
same role as the CMEMS preset files: they give each observation family
a stable short ID (``aemet_daily``, ``aemet_hourly``, ...) that maps to
an endpoint and a set of canonical variables.

Adapters dispatch on the ``aemet_kind`` extras entry, which matches the
dataset ID. Users who want to call a specific endpoint directly
(e.g. water-balance products, satellite imagery) can still do so
via :class:`~xr_toolz.data.AemetSource`'s lower-level methods.
"""

from __future__ import annotations

from xr_toolz.data._src.base import DatasetInfo, DatasetKind
from xr_toolz.types import (
    AIR_TEMPERATURE,
    AIR_TEMPERATURE_DAILY_MAX,
    AIR_TEMPERATURE_DAILY_MEAN,
    AIR_TEMPERATURE_DAILY_MIN,
    AIR_TEMPERATURE_MAX,
    AIR_TEMPERATURE_MIN,
    DEW_POINT_TEMPERATURE,
    MEAN_SEA_LEVEL_PRESSURE_HPA,
    PRECIPITATION_AMOUNT,
    RELATIVE_HUMIDITY,
    SOIL_TEMPERATURE_5CM,
    SOIL_TEMPERATURE_20CM,
    SUNSHINE_DURATION,
    SUNSHINE_DURATION_DAILY,
    SURFACE_PRESSURE_HPA,
    SURFACE_PRESSURE_MAX_HPA,
    SURFACE_PRESSURE_MIN_HPA,
    SURFACE_SNOW_THICKNESS,
    VISIBILITY,
    WIND_FROM_DIRECTION,
    WIND_FROM_DIRECTION_DAILY,
    WIND_FROM_DIRECTION_OF_GUST,
    WIND_SPEED,
    WIND_SPEED_DAILY_MEAN,
    WIND_SPEED_OF_GUST,
    WIND_SPEED_OF_GUST_DAILY,
    BBox,
)


# AEMET covers mainland Spain, the Balearic and Canary Islands, Ceuta
# and Melilla. The envelope below is generous to keep antimeridian /
# ocean-buoy edge cases out of the way.
_SPAIN_BBOX = BBox(lon_min=-18.5, lon_max=4.5, lat_min=27.5, lat_max=44.0)


AEMET_DATASETS: dict[str, DatasetInfo] = {
    "aemet_stations": DatasetInfo(
        dataset_id="aemet_stations",
        source="aemet",
        title="AEMET — station inventory (climatological network)",
        kind=DatasetKind.STATIONS,
        variables=(),
        spatial_coverage=_SPAIN_BBOX,
        license="AEMET OpenData (attribution required)",
        notes=(
            "Full climatological station network, updated daily. "
            "No time dimension — returns one row per station."
        ),
        extras={"aemet_kind": "aemet_stations"},
    ),
    "aemet_daily": DatasetInfo(
        dataset_id="aemet_daily",
        source="aemet",
        title="AEMET — daily climatological values",
        kind=DatasetKind.STATIONS,
        variables=(
            AIR_TEMPERATURE_DAILY_MEAN,
            AIR_TEMPERATURE_DAILY_MIN,
            AIR_TEMPERATURE_DAILY_MAX,
            PRECIPITATION_AMOUNT,
            WIND_SPEED_DAILY_MEAN,
            WIND_SPEED_OF_GUST_DAILY,
            WIND_FROM_DIRECTION_DAILY,
            SURFACE_PRESSURE_MAX_HPA,
            SURFACE_PRESSURE_MIN_HPA,
            SUNSHINE_DURATION_DAILY,
        ),
        spatial_coverage=_SPAIN_BBOX,
        temporal_coverage=("1920-01-01", "present"),
        license="AEMET OpenData (attribution required)",
        notes=(
            "Per-station daily values. Endpoint caps each request at "
            "~180 days; AemetSource chunks longer windows automatically."
        ),
        extras={"aemet_kind": "aemet_daily"},
    ),
    "aemet_hourly": DatasetInfo(
        dataset_id="aemet_hourly",
        source="aemet",
        title="AEMET — hourly conventional observations",
        kind=DatasetKind.STATIONS,
        variables=(
            AIR_TEMPERATURE,
            AIR_TEMPERATURE_MIN,
            AIR_TEMPERATURE_MAX,
            DEW_POINT_TEMPERATURE,
            RELATIVE_HUMIDITY,
            PRECIPITATION_AMOUNT,
            SURFACE_PRESSURE_HPA,
            MEAN_SEA_LEVEL_PRESSURE_HPA,
            WIND_SPEED,
            WIND_FROM_DIRECTION,
            WIND_SPEED_OF_GUST,
            WIND_FROM_DIRECTION_OF_GUST,
            SUNSHINE_DURATION,
            VISIBILITY,
            SURFACE_SNOW_THICKNESS,
            SOIL_TEMPERATURE_5CM,
            SOIL_TEMPERATURE_20CM,
        ),
        spatial_coverage=_SPAIN_BBOX,
        temporal_coverage=("rolling-24h", "present"),
        license="AEMET OpenData (attribution required)",
        notes=(
            "Rolling ~24 hours of hourly observations. Historical hourly "
            "data is not exposed via this endpoint; use ``aemet_daily`` "
            "for archived records."
        ),
        extras={"aemet_kind": "aemet_hourly"},
    ),
    "aemet_monthly": DatasetInfo(
        dataset_id="aemet_monthly",
        source="aemet",
        title="AEMET — monthly and annual climatological aggregates",
        kind=DatasetKind.STATIONS,
        variables=(
            AIR_TEMPERATURE_DAILY_MEAN,
            AIR_TEMPERATURE_DAILY_MIN,
            AIR_TEMPERATURE_DAILY_MAX,
            PRECIPITATION_AMOUNT,
            WIND_SPEED_DAILY_MEAN,
            WIND_SPEED_OF_GUST_DAILY,
            SURFACE_PRESSURE_HPA,
            SURFACE_PRESSURE_MAX_HPA,
            SURFACE_PRESSURE_MIN_HPA,
            SUNSHINE_DURATION_DAILY,
        ),
        spatial_coverage=_SPAIN_BBOX,
        temporal_coverage=("1920-01-01", "present"),
        license="AEMET OpenData (attribution required)",
        notes=(
            "Month- and year-level aggregates per station. "
            "AemetSource chunks long year ranges into 3-year (36-month) "
            "requests — AEMET's per-request cap on this endpoint."
        ),
        extras={"aemet_kind": "aemet_monthly"},
    ),
    "aemet_normals": DatasetInfo(
        dataset_id="aemet_normals",
        source="aemet",
        title="AEMET — climate normals (1981-2010)",
        kind=DatasetKind.STATIONS,
        variables=(
            AIR_TEMPERATURE_DAILY_MEAN,
            AIR_TEMPERATURE_DAILY_MIN,
            AIR_TEMPERATURE_DAILY_MAX,
            PRECIPITATION_AMOUNT,
            SUNSHINE_DURATION_DAILY,
        ),
        spatial_coverage=_SPAIN_BBOX,
        temporal_coverage=("1981-01-01", "2010-12-31"),
        license="AEMET OpenData (attribution required)",
        notes="Monthly climatology normals (12-month cycle) per station.",
        extras={"aemet_kind": "aemet_normals"},
    ),
    "aemet_extremes": DatasetInfo(
        dataset_id="aemet_extremes",
        source="aemet",
        title="AEMET — record extremes (precipitation / temperature / wind)",
        kind=DatasetKind.STATIONS,
        variables=(),
        spatial_coverage=_SPAIN_BBOX,
        license="AEMET OpenData (attribution required)",
        notes=(
            "Station records for one ``parameter`` in {P, T, V}. "
            "Returned as ragged records rather than a time series."
        ),
        extras={"aemet_kind": "aemet_extremes", "parameter": "T"},
    ),
    "aemet_pollution": DatasetInfo(
        dataset_id="aemet_pollution",
        source="aemet",
        title="AEMET — EMEP background pollution network",
        kind=DatasetKind.STATIONS,
        variables=(),
        spatial_coverage=_SPAIN_BBOX,
        license="AEMET OpenData (attribution required)",
        notes=(
            "Background pollution observations (ozone, aerosols, acid "
            "deposition) from the EMEP-aligned network. Sampling cadence "
            "varies per analyte."
        ),
        extras={"aemet_kind": "aemet_pollution"},
    ),
}
