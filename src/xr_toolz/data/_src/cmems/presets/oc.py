"""CMEMS ocean-colour presets (GlobColour L4 multi-sensor, reprocessed).

GlobColour is organised by theme (plankton / transparency / optics /
reflectance / primary production) — each theme is a separate
``dataset_id`` so each is registered individually.
"""

from __future__ import annotations

from xr_toolz.data._src.base import DatasetInfo, DatasetKind
from xr_toolz.types import (
    BBP443,
    CHL,
    KD490,
    PP,
    RRS412,
    RRS443,
    RRS490,
    RRS510,
    RRS555,
    RRS670,
    SPM,
    ZSD,
    BBox,
)


OC_DATASETS: dict[str, DatasetInfo] = {
    # ---- Plankton (chlorophyll) ----------------------------------------
    "cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M": DatasetInfo(
        dataset_id="cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M",
        source="cmems",
        title="GlobColour L4 — Global chlorophyll (reprocessed, 4 km, monthly)",
        kind=DatasetKind.GRIDDED,
        variables=(CHL,),
        spatial_coverage=BBox.global_(),
        temporal_coverage=("1997-09-01", "present"),
        license="Copernicus Marine Service",
    ),
    "cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D": DatasetInfo(
        dataset_id="cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D",
        source="cmems",
        title="GlobColour L4 — Global chlorophyll gap-free (reprocessed, 4 km, daily)",
        kind=DatasetKind.GRIDDED,
        variables=(CHL,),
        spatial_coverage=BBox.global_(),
        temporal_coverage=("1997-09-01", "present"),
        license="Copernicus Marine Service",
    ),
    # ---- Plankton L3 daily (multi-sensor + per-sensor OLCI) ------------
    "cmems_obs-oc_glo_bgc-plankton_my_l3-multi-4km_P1D": DatasetInfo(
        dataset_id="cmems_obs-oc_glo_bgc-plankton_my_l3-multi-4km_P1D",
        source="cmems",
        title=(
            "GlobColour L3 — Global chlorophyll multi-sensor (reprocessed, 4 km, daily)"
        ),
        kind=DatasetKind.GRIDDED,
        variables=(CHL,),
        spatial_coverage=BBox.global_(),
        license="Copernicus Marine Service",
        notes="Multi-sensor merged L3, daily. Cloud-gapped.",
    ),
    "cmems_obs-oc_glo_bgc-plankton_my_l3-olci-4km_P1D": DatasetInfo(
        dataset_id="cmems_obs-oc_glo_bgc-plankton_my_l3-olci-4km_P1D",
        source="cmems",
        title=(
            "GlobColour L3 — Global chlorophyll, OLCI Sentinel-3 "
            "(reprocessed, 4 km, daily)"
        ),
        kind=DatasetKind.GRIDDED,
        variables=(CHL,),
        spatial_coverage=BBox.global_(),
        license="Copernicus Marine Service",
        notes="OLCI Sentinel-3 single-sensor L3 binned to 4 km. Cloud-gapped.",
    ),
    "cmems_obs-oc_glo_bgc-plankton_my_l3-olci-300m_P1D": DatasetInfo(
        dataset_id="cmems_obs-oc_glo_bgc-plankton_my_l3-olci-300m_P1D",
        source="cmems",
        title=(
            "GlobColour L3 — Global chlorophyll, OLCI Sentinel-3 "
            "(reprocessed, 300 m, daily)"
        ),
        kind=DatasetKind.GRIDDED,
        variables=(CHL,),
        spatial_coverage=BBox.global_(),
        license="Copernicus Marine Service",
        notes="OLCI Sentinel-3 native 300 m resolution. Heavy data volume.",
    ),
    # ---- Transparency L3 / L4 gap-free ---------------------------------
    "cmems_obs-oc_glo_bgc-transp_my_l3-multi-4km_P1D": DatasetInfo(
        dataset_id="cmems_obs-oc_glo_bgc-transp_my_l3-multi-4km_P1D",
        source="cmems",
        title="GlobColour L3 — Global transparency (reprocessed, 4 km, daily)",
        kind=DatasetKind.GRIDDED,
        variables=(KD490, ZSD, SPM),
        spatial_coverage=BBox.global_(),
        license="Copernicus Marine Service",
    ),
    "cmems_obs-oc_glo_bgc-transp_my_l4-gapfree-multi-4km_P1D": DatasetInfo(
        dataset_id="cmems_obs-oc_glo_bgc-transp_my_l4-gapfree-multi-4km_P1D",
        source="cmems",
        title="GlobColour L4 — Global transparency gap-free (reprocessed, 4 km, daily)",
        kind=DatasetKind.GRIDDED,
        variables=(KD490, ZSD),
        spatial_coverage=BBox.global_(),
        license="Copernicus Marine Service",
    ),
    # ---- Transparency (Kd490 / Secchi / SPM) ---------------------------
    "cmems_obs-oc_glo_bgc-transp_my_l4-multi-4km_P1M": DatasetInfo(
        dataset_id="cmems_obs-oc_glo_bgc-transp_my_l4-multi-4km_P1M",
        source="cmems",
        title="GlobColour L4 — Global transparency (reprocessed, 4 km, monthly)",
        kind=DatasetKind.GRIDDED,
        variables=(KD490, ZSD, SPM),
        spatial_coverage=BBox.global_(),
        temporal_coverage=("1997-09-01", "present"),
        license="Copernicus Marine Service",
    ),
    # ---- Optics (backscattering / absorption) --------------------------
    "cmems_obs-oc_glo_bgc-optics_my_l4-multi-4km_P1M": DatasetInfo(
        dataset_id="cmems_obs-oc_glo_bgc-optics_my_l4-multi-4km_P1M",
        source="cmems",
        title=(
            "GlobColour L4 — Global inherent optical properties "
            "(reprocessed, 4 km, monthly)"
        ),
        kind=DatasetKind.GRIDDED,
        variables=(BBP443,),
        spatial_coverage=BBox.global_(),
        temporal_coverage=("1997-09-01", "present"),
        license="Copernicus Marine Service",
    ),
    # ---- Reflectance (Rrs wavelength family) ---------------------------
    "cmems_obs-oc_glo_bgc-reflectance_my_l4-multi-4km_P1M": DatasetInfo(
        dataset_id="cmems_obs-oc_glo_bgc-reflectance_my_l4-multi-4km_P1M",
        source="cmems",
        title=(
            "GlobColour L4 — Global remote-sensing reflectance "
            "(reprocessed, 4 km, monthly)"
        ),
        kind=DatasetKind.GRIDDED,
        variables=(RRS412, RRS443, RRS490, RRS510, RRS555, RRS670),
        spatial_coverage=BBox.global_(),
        temporal_coverage=("1997-09-01", "present"),
        license="Copernicus Marine Service",
    ),
    # ---- Primary production --------------------------------------------
    "cmems_obs-oc_glo_bgc-pp_my_l4-multi-4km_P1M": DatasetInfo(
        dataset_id="cmems_obs-oc_glo_bgc-pp_my_l4-multi-4km_P1M",
        source="cmems",
        title="GlobColour L4 — Global primary production (reprocessed, 4 km, monthly)",
        kind=DatasetKind.GRIDDED,
        variables=(PP,),
        spatial_coverage=BBox.global_(),
        temporal_coverage=("1997-09-01", "present"),
        license="Copernicus Marine Service",
    ),
}
