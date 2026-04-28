"""CMEMS sea-surface-temperature observation presets (OSTIA + ODYSSEA).

OSTIA L4 (SST_GLO_SST_L4_REP_OBSERVATIONS_010_011) uses a legacy
dataset naming scheme — the modern ``copernicusmarine`` client accepts
the uppercase "METOFFICE-…" form directly as ``dataset_id``.
"""

from __future__ import annotations

from xr_toolz.data._src.base import DatasetInfo, DatasetKind
from xr_toolz.types import ANALYSED_SST, ICE_CONC, SST_OBS, BBox


SST_DATASETS: dict[str, DatasetInfo] = {
    # ---- OSTIA L4 reprocessed ------------------------------------------
    "METOFFICE-GLO-SST-L4-REP-OBS-SST": DatasetInfo(
        dataset_id="METOFFICE-GLO-SST-L4-REP-OBS-SST",
        source="cmems",
        title="OSTIA L4 — Global SST + sea-ice (reprocessed, 0.05°, daily)",
        kind=DatasetKind.GRIDDED,
        variables=(ANALYSED_SST, ICE_CONC),
        spatial_coverage=BBox.global_(),
        temporal_coverage=("1981-10-01", "present"),
        license="Copernicus Marine Service",
        notes=(
            "Legacy-format dataset_id (not cmems_-prefixed). The "
            "copernicusmarine client accepts both forms."
        ),
    ),
    # ---- OSTIA L4 NRT --------------------------------------------------
    "METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2": DatasetInfo(
        dataset_id="METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2",
        source="cmems",
        title="OSTIA L4 — Global SST + sea-ice (NRT, 0.05°, daily)",
        kind=DatasetKind.GRIDDED,
        variables=(ANALYSED_SST, ICE_CONC),
        spatial_coverage=BBox.global_(),
        license="Copernicus Marine Service",
    ),
    # ---- ODYSSEA L3S multi-sensor (reprocessed) ------------------------
    "cmems_obs-sst_glo_phy_my_l3s_P1D-m_202311": DatasetInfo(
        dataset_id="cmems_obs-sst_glo_phy_my_l3s_P1D-m_202311",
        source="cmems",
        title="ODYSSEA L3S — Global multi-sensor SST (reprocessed, 0.1°, daily)",
        kind=DatasetKind.GRIDDED,
        variables=(SST_OBS,),
        spatial_coverage=BBox.global_(),
        temporal_coverage=("1982-01-01", "present"),
        license="Copernicus Marine Service",
    ),
    # ---- L3 single-sensor groups (daily, gridded with cloud gaps) ------
    "cmems_obs-sst_glo_phy_l3s_gir_P1D-m": DatasetInfo(
        dataset_id="cmems_obs-sst_glo_phy_l3s_gir_P1D-m",
        source="cmems",
        title="CMEMS L3 — Geostationary IR SST (multi-platform, 0.05°, daily)",
        kind=DatasetKind.GRIDDED,
        variables=(SST_OBS,),
        spatial_coverage=BBox.global_(),
        license="Copernicus Marine Service",
        notes="Geostationary infrared (e.g. SEVIRI / GOES). Cloud-affected.",
    ),
    "cmems_obs-sst_glo_phy_l3s_pir_P1D-m": DatasetInfo(
        dataset_id="cmems_obs-sst_glo_phy_l3s_pir_P1D-m",
        source="cmems",
        title="CMEMS L3 — Polar-orbiting IR SST (multi-platform, 0.1°, daily)",
        kind=DatasetKind.GRIDDED,
        variables=(SST_OBS,),
        spatial_coverage=BBox.global_(),
        license="Copernicus Marine Service",
        notes="Polar IR (e.g. AVHRR, SLSTR). Cloud-affected.",
    ),
    "cmems_obs-sst_glo_phy_l3s_pmw_P1D-m": DatasetInfo(
        dataset_id="cmems_obs-sst_glo_phy_l3s_pmw_P1D-m",
        source="cmems",
        title="CMEMS L3 — Passive microwave SST (multi-platform, 0.25°, daily)",
        kind=DatasetKind.GRIDDED,
        variables=(SST_OBS,),
        spatial_coverage=BBox.global_(),
        license="Copernicus Marine Service",
        notes=(
            "Passive microwave (e.g. AMSR2). Coarser resolution but "
            "sees through non-precipitating clouds."
        ),
    ),
}
