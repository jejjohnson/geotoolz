"""CDS in-situ observation presets (surface-land + surface-marine).

The variable lists below are **pinned to the live CDS form schema**,
fetched from
``<cdsapi_url>/retrieve/v1/processes/<dataset_id>``. Re-running that
request is cheap and the canonical way to spot drift in the CDS
vocabulary; if variables disappear from the CDS side, update these
entries and bump the curated list.

Both products return a zip bundle with CSV inside (``data_format=csv``)
or a NetCDF file (``data_format=netcdf``). They accept ``area``
server-side and take ``year`` as a **single** string per request;
:class:`~xr_toolz.data._src.cds.archive.CDSInsituArchive` chunks by
year to respect this limit.

Land requires ``time_aggregation`` ∈ ``{sub_daily, daily, monthly}`` on
every request; marine does not (marine ships a single aggregation tier
baked into the product).
"""

from __future__ import annotations

from xr_toolz.data._src.base import DatasetInfo, DatasetKind
from xr_toolz.data._src.cds.profiles import INSITU_LAND, INSITU_MARINE
from xr_toolz.types import (
    AIR_TEMPERATURE,
    DEW_POINT_TEMPERATURE,
    FRESH_SNOW,
    MEAN_SEA_LEVEL_PRESSURE_HPA,
    PRECIPITATION_AMOUNT,
    SEA_LEVEL_PRESSURE,
    SEA_SURFACE_TEMPERATURE_INSITU,
    SNOW_WATER_EQUIVALENT,
    SURFACE_PRESSURE_HPA,
    SURFACE_SNOW_THICKNESS,
    WIND_FROM_DIRECTION,
    WIND_SPEED,
    BBox,
)


# CDS surface-land ``variable`` enum (schema 2.0.0, fetched 2026-04-24):
# accumulated_precipitation, air_pressure, air_pressure_at_sea_level,
# air_temperature, dew_point_temperature, fresh_snow, snow_depth,
# snow_water_equivalent, wind_from_direction, wind_speed.
INSITU_LAND_PRESET = DatasetInfo(
    dataset_id="insitu-observations-surface-land",
    source="cds",
    title="Global land surface atmospheric variables from 1763 to present",
    kind=DatasetKind.STATIONS,
    variables=(
        AIR_TEMPERATURE,
        DEW_POINT_TEMPERATURE,
        SURFACE_PRESSURE_HPA,
        MEAN_SEA_LEVEL_PRESSURE_HPA,
        WIND_SPEED,
        WIND_FROM_DIRECTION,
        PRECIPITATION_AMOUNT,
        SURFACE_SNOW_THICKNESS,
        FRESH_SNOW,
        SNOW_WATER_EQUIVALENT,
    ),
    spatial_coverage=BBox.global_(),
    temporal_coverage=("1763-01-01", "present"),
    license="Varies — see the dataset's licences tab on the CDS portal.",
    form_profile=INSITU_LAND,
    notes=(
        "Zip-of-CSV per request. Requires ``time_aggregation`` "
        "∈ {'sub_daily', 'daily', 'monthly'}. ``year`` is a single "
        "string per request — CDSInsituArchive chunks by year. Use "
        "``area`` for server-side bbox filtering."
    ),
)


# CDS surface-marine ``variable`` enum (schema 2.0.0, fetched 2026-04-24):
# air_pressure_at_sea_level, air_temperature, dew_point_temperature,
# water_temperature, wind_from_direction, wind_speed.
INSITU_MARINE_PRESET = DatasetInfo(
    dataset_id="insitu-observations-surface-marine",
    source="cds",
    title="Global marine surface observations from 1850 to present",
    kind=DatasetKind.STATIONS,
    variables=(
        AIR_TEMPERATURE,
        DEW_POINT_TEMPERATURE,
        SEA_LEVEL_PRESSURE,
        SEA_SURFACE_TEMPERATURE_INSITU,
        WIND_SPEED,
        WIND_FROM_DIRECTION,
    ),
    spatial_coverage=BBox.global_(),
    temporal_coverage=("1850-01-01", "present"),
    license="Varies — see the dataset's licences tab on the CDS portal.",
    form_profile=INSITU_MARINE,
    notes=(
        "Zip-of-CSV from ships / buoys / drifters / fixed platforms. "
        "``year`` is a single string per request — CDSInsituArchive "
        "chunks by year. Use ``area`` for server-side bbox filtering. "
        "No ``time_aggregation`` — marine ships a single aggregation tier."
    ),
)


INSITU_DATASETS: dict[str, DatasetInfo] = {
    INSITU_LAND_PRESET.dataset_id: INSITU_LAND_PRESET,
    INSITU_MARINE_PRESET.dataset_id: INSITU_MARINE_PRESET,
}
