"""CMEMS biogeochemistry reanalysis presets (global PISCES-based)."""

from __future__ import annotations

from xr_toolz.data._src.base import DatasetInfo, DatasetKind
from xr_toolz.types import (
    CHL,
    NO3,
    O2,
    PH,
    PHYC,
    PO4,
    SI,
    SPCO2,
    ZOOC,
    BBox,
)


BGC_DATASETS: dict[str, DatasetInfo] = {
    "cmems_mod_glo_bgc_my_0.25deg_P1D-m_202406": DatasetInfo(
        dataset_id="cmems_mod_glo_bgc_my_0.25deg_P1D-m_202406",
        source="cmems",
        title="Global BGC reanalysis (PISCES) — 0.25°, daily mean",
        kind=DatasetKind.GRIDDED,
        variables=(CHL, NO3, PO4, SI, O2, PHYC, ZOOC, PH, SPCO2),
        spatial_coverage=BBox.global_(),
        temporal_coverage=("1993-01-01", "present"),
        license="Copernicus Marine Service",
    ),
    "cmems_mod_glo_bgc_my_0.25deg_P1M-m_202406": DatasetInfo(
        dataset_id="cmems_mod_glo_bgc_my_0.25deg_P1M-m_202406",
        source="cmems",
        title="Global BGC reanalysis (PISCES) — 0.25°, monthly mean",
        kind=DatasetKind.GRIDDED,
        variables=(CHL, NO3, PO4, SI, O2, PHYC, ZOOC, PH, SPCO2),
        spatial_coverage=BBox.global_(),
        temporal_coverage=("1993-01-01", "present"),
        license="Copernicus Marine Service",
    ),
}
