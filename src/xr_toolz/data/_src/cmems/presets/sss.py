"""CMEMS sea-surface-salinity multi-observation presets (MULTIOBS)."""

from __future__ import annotations

from xr_toolz.data._src.base import DatasetInfo, DatasetKind
from xr_toolz.types import DENS, SOS, BBox


SSS_DATASETS: dict[str, DatasetInfo] = {
    "cmems_obs-mob_glo_phy-sss_my_multi_P1D": DatasetInfo(
        dataset_id="cmems_obs-mob_glo_phy-sss_my_multi_P1D",
        source="cmems",
        title="MULTIOBS L4 — Global SSS + density (reprocessed, 0.125°, daily)",
        kind=DatasetKind.GRIDDED,
        variables=(SOS, DENS),
        spatial_coverage=BBox.global_(),
        temporal_coverage=("1993-01-01", "present"),
        license="Copernicus Marine Service",
    ),
    "cmems_obs-mob_glo_phy-sss_my_multi_P1M": DatasetInfo(
        dataset_id="cmems_obs-mob_glo_phy-sss_my_multi_P1M",
        source="cmems",
        title="MULTIOBS L4 — Global SSS + density (reprocessed, 0.125°, monthly)",
        kind=DatasetKind.GRIDDED,
        variables=(SOS, DENS),
        spatial_coverage=BBox.global_(),
        temporal_coverage=("1993-01-01", "present"),
        license="Copernicus Marine Service",
    ),
    "cmems_obs-mob_glo_phy-sss_nrt_multi_P1D": DatasetInfo(
        dataset_id="cmems_obs-mob_glo_phy-sss_nrt_multi_P1D",
        source="cmems",
        title="MULTIOBS L4 — Global SSS + density (NRT, 0.125°, daily)",
        kind=DatasetKind.GRIDDED,
        variables=(SOS, DENS),
        spatial_coverage=BBox.global_(),
        license="Copernicus Marine Service",
    ),
    "cmems_obs-mob_glo_phy-sss_nrt_multi_P1M": DatasetInfo(
        dataset_id="cmems_obs-mob_glo_phy-sss_nrt_multi_P1M",
        source="cmems",
        title="MULTIOBS L4 — Global SSS + density (NRT, 0.125°, monthly)",
        kind=DatasetKind.GRIDDED,
        variables=(SOS, DENS),
        spatial_coverage=BBox.global_(),
        license="Copernicus Marine Service",
    ),
}
