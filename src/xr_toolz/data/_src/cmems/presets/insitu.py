"""CMEMS in-situ observation presets (CORA T/S profiles).

In-situ products are not gridded — the client returns a collection of
trajectories / profiles. :attr:`DatasetInfo.kind` is set accordingly so
downstream code can branch on it.
"""

from __future__ import annotations

from xr_toolz.data._src.base import DatasetInfo, DatasetKind
from xr_toolz.types import SO, SST, BBox


INSITU_DATASETS: dict[str, DatasetInfo] = {
    "cmems_obs-ins_glo_phy-temp-sal_my_cora_irr": DatasetInfo(
        dataset_id="cmems_obs-ins_glo_phy-temp-sal_my_cora_irr",
        source="cmems",
        title=(
            "CORA — Global in-situ T/S discrete profiles (Argo + CTD + "
            "XBT + gliders + WOD)"
        ),
        kind=DatasetKind.PROFILES,
        variables=(SST, SO),
        spatial_coverage=BBox.global_(),
        temporal_coverage=("1950-01-01", "2021-12-31"),
        license="Copernicus Marine Service",
        notes=(
            "Non-gridded — output is a collection of vertical profiles at "
            "scattered lat/lon/time. The returned container is a Dataset "
            "with a trajectory-style layout, not a regular cube."
        ),
    ),
    "cmems_obs-ins_glo_phy-temp-sal_my_easycora_irr": DatasetInfo(
        dataset_id="cmems_obs-ins_glo_phy-temp-sal_my_easycora_irr",
        source="cmems",
        title="EasyCORA — Simplified global in-situ T/S profiles",
        kind=DatasetKind.PROFILES,
        variables=(SST, SO),
        spatial_coverage=BBox.global_(),
        temporal_coverage=("1950-01-01", "2021-12-31"),
        license="Copernicus Marine Service",
        notes="Non-gridded — see CORA notes.",
    ),
}
