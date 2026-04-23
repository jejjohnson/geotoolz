"""CMEMS sea-surface-height observation presets (DUACS L4 + L3).

The L3 along-track product ships one ``dataset_id`` per satellite
mission; we expose the common reference missions. Users who need a
specific mission not in this list can pass the ``dataset_id`` directly
to :class:`~xr_toolz.data.CMEMSSource`.
"""

from __future__ import annotations

from xr_toolz.data._src.base import DatasetInfo, DatasetKind
from xr_toolz.types import ADT, SLA, UGOS, VGOS, BBox


SSH_DATASETS: dict[str, DatasetInfo] = {
    # ---- L4 gridded (reprocessed) --------------------------------------
    "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D": DatasetInfo(
        dataset_id="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D",
        source="cmems",
        title="DUACS L4 — Global ocean gridded SLA (reprocessed, daily, 0.25°)",
        kind=DatasetKind.GRIDDED,
        variables=(SLA, ADT, UGOS, VGOS),
        spatial_coverage=BBox.global_(),
        temporal_coverage=("1993-01-01", "present"),
        license="Copernicus Marine Service",
    ),
    # ---- L3 along-track (per-mission) ----------------------------------
    "cmems_obs-sl_glo_phy-ssh_my_s3a-l3-duacs_PT1S_202411": DatasetInfo(
        dataset_id="cmems_obs-sl_glo_phy-ssh_my_s3a-l3-duacs_PT1S_202411",
        source="cmems",
        title="DUACS L3 along-track SLA — Sentinel-3A (reprocessed)",
        kind=DatasetKind.ALONGTRACK,
        variables=(SLA,),
        spatial_coverage=BBox.global_(),
        license="Copernicus Marine Service",
    ),
    "cmems_obs-sl_glo_phy-ssh_my_s3b-l3-duacs_PT1S_202411": DatasetInfo(
        dataset_id="cmems_obs-sl_glo_phy-ssh_my_s3b-l3-duacs_PT1S_202411",
        source="cmems",
        title="DUACS L3 along-track SLA — Sentinel-3B (reprocessed)",
        kind=DatasetKind.ALONGTRACK,
        variables=(SLA,),
        spatial_coverage=BBox.global_(),
        license="Copernicus Marine Service",
    ),
    "cmems_obs-sl_glo_phy-ssh_my_s6a-lr-l3-duacs_PT1S_202411": DatasetInfo(
        dataset_id="cmems_obs-sl_glo_phy-ssh_my_s6a-lr-l3-duacs_PT1S_202411",
        source="cmems",
        title="DUACS L3 along-track SLA — Sentinel-6A (reprocessed)",
        kind=DatasetKind.ALONGTRACK,
        variables=(SLA,),
        spatial_coverage=BBox.global_(),
        license="Copernicus Marine Service",
    ),
    "cmems_obs-sl_glo_phy-ssh_my_j3-l3-duacs_PT1S_202411": DatasetInfo(
        dataset_id="cmems_obs-sl_glo_phy-ssh_my_j3-l3-duacs_PT1S_202411",
        source="cmems",
        title="DUACS L3 along-track SLA — Jason-3 (reprocessed)",
        kind=DatasetKind.ALONGTRACK,
        variables=(SLA,),
        spatial_coverage=BBox.global_(),
        license="Copernicus Marine Service",
    ),
    "cmems_obs-sl_glo_phy-ssh_my_al-l3-duacs_PT1S_202411": DatasetInfo(
        dataset_id="cmems_obs-sl_glo_phy-ssh_my_al-l3-duacs_PT1S_202411",
        source="cmems",
        title="DUACS L3 along-track SLA — Saral/AltiKa (reprocessed)",
        kind=DatasetKind.ALONGTRACK,
        variables=(SLA,),
        spatial_coverage=BBox.global_(),
        license="Copernicus Marine Service",
    ),
    "cmems_obs-sl_glo_phy-ssh_my_swon-l3-duacs_PT1S_202411": DatasetInfo(
        dataset_id="cmems_obs-sl_glo_phy-ssh_my_swon-l3-duacs_PT1S_202411",
        source="cmems",
        title="DUACS L3 along-track SLA — SWOT nadir (reprocessed)",
        kind=DatasetKind.ALONGTRACK,
        variables=(SLA,),
        spatial_coverage=BBox.global_(),
        license="Copernicus Marine Service",
    ),
}
