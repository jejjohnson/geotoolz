"""Small curated catalog of Climate Data Store products."""

from __future__ import annotations

from xr_toolz.data._src.base import DatasetInfo
from xr_toolz.types import (
    D2M,
    MSL,
    SP,
    SSRD,
    T2M,
    TP,
    U10,
    V10,
    BBox,
)


CDS_DATASETS: dict[str, DatasetInfo] = {
    "reanalysis-era5-single-levels": DatasetInfo(
        dataset_id="reanalysis-era5-single-levels",
        source="cds",
        title="ERA5 — Single levels (surface/near-surface, hourly)",
        variables=(T2M, D2M, U10, V10, MSL, TP, SP, SSRD),
        spatial_coverage=BBox.global_(),
        temporal_coverage=("1940-01-01", "present"),
        license="Copernicus Climate Change Service",
    ),
    "reanalysis-era5-pressure-levels": DatasetInfo(
        dataset_id="reanalysis-era5-pressure-levels",
        source="cds",
        title="ERA5 — Pressure levels (hourly)",
        variables=(),
        spatial_coverage=BBox.global_(),
        temporal_coverage=("1940-01-01", "present"),
        license="Copernicus Climate Change Service",
    ),
    "reanalysis-era5-land": DatasetInfo(
        dataset_id="reanalysis-era5-land",
        source="cds",
        title="ERA5-Land — Hourly land-surface reanalysis",
        variables=(T2M, D2M, TP, SP),
        spatial_coverage=BBox.global_(),
        temporal_coverage=("1950-01-01", "present"),
        license="Copernicus Climate Change Service",
    ),
}
