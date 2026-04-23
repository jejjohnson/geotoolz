"""CMEMS sea-surface-temperature observation presets (OSTIA + ODYSSEA).

OSTIA L4 (SST_GLO_SST_L4_REP_OBSERVATIONS_010_011) uses a legacy
dataset naming scheme — the modern ``copernicusmarine`` client accepts
the uppercase "METOFFICE-…" form directly as ``dataset_id``.
"""

from __future__ import annotations

from xr_toolz.data._src.base import DatasetInfo, DatasetKind
from xr_toolz.types import ICE_CONC, SST, BBox


SST_DATASETS: dict[str, DatasetInfo] = {
    # ---- OSTIA L4 reprocessed ------------------------------------------
    "METOFFICE-GLO-SST-L4-REP-OBS-SST": DatasetInfo(
        dataset_id="METOFFICE-GLO-SST-L4-REP-OBS-SST",
        source="cmems",
        title="OSTIA L4 — Global SST + sea-ice (reprocessed, 0.05°, daily)",
        kind=DatasetKind.GRIDDED,
        variables=(SST, ICE_CONC),
        spatial_coverage=BBox.global_(),
        temporal_coverage=("1981-10-01", "present"),
        license="Copernicus Marine Service",
        notes=(
            "Legacy-format dataset_id (not cmems_-prefixed). The "
            "copernicusmarine client accepts both forms."
        ),
    ),
    # ---- ODYSSEA L3S multi-sensor (reprocessed) ------------------------
    "cmems_obs-sst_glo_phy_my_l3s_P1D-m_202311": DatasetInfo(
        dataset_id="cmems_obs-sst_glo_phy_my_l3s_P1D-m_202311",
        source="cmems",
        title="ODYSSEA L3S — Global multi-sensor SST (reprocessed, 0.1°, daily)",
        kind=DatasetKind.GRIDDED,
        variables=(SST,),
        spatial_coverage=BBox.global_(),
        temporal_coverage=("1982-01-01", "present"),
        license="Copernicus Marine Service",
    ),
}
