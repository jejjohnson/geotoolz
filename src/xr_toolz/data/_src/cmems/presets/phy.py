"""CMEMS physics reanalysis presets (GLORYS)."""

from __future__ import annotations

from xr_toolz.data._src.base import DatasetInfo, DatasetKind
from xr_toolz.types import SO, SSH, SST, UO, VO, BBox


PHY_DATASETS: dict[str, DatasetInfo] = {
    "cmems_mod_glo_phy_my_0.083deg_P1D-m": DatasetInfo(
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        source="cmems",
        title="GLORYS12 — Global ocean physics reanalysis, 1/12°, daily mean",
        kind=DatasetKind.GRIDDED,
        variables=(SST, SSH, UO, VO, SO),
        spatial_coverage=BBox.global_(),
        temporal_coverage=("1993-01-01", "present"),
        license="Copernicus Marine Service",
    ),
}
