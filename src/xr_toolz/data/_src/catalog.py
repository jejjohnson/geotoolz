"""Unified dataset catalog: short name -> (source, dataset_id).

Short names are a thin convenience layer — they map a memorable label
like ``"glorys12.daily"`` to the fully-qualified ``(source, dataset_id)``
tuple that adapters actually consume. Users can always bypass the
catalog and pass a ``dataset_id`` directly to the adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from xr_toolz.data._src.aemet.catalog import AEMET_DATASETS
from xr_toolz.data._src.base import DatasetInfo
from xr_toolz.data._src.cds.catalog import CDS_DATASETS
from xr_toolz.data._src.cmems.catalog import CMEMS_DATASETS


@dataclass(frozen=True)
class CatalogEntry:
    """Short-name-to-source mapping with optional request defaults."""

    source: str
    dataset_id: str
    defaults: dict[str, Any] | None = None


CATALOG: dict[str, CatalogEntry] = {
    # ---- CMEMS — physics reanalysis ------------------------------------
    "glorys12.daily": CatalogEntry(
        source="cmems",
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
    ),
    # ---- CMEMS — sea level --------------------------------------------
    "duacs.sla": CatalogEntry(
        source="cmems",
        dataset_id="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D",
    ),
    "duacs.alongtrack.s3a": CatalogEntry(
        source="cmems",
        dataset_id="cmems_obs-sl_glo_phy-ssh_my_s3a-l3-duacs_PT1S_202411",
    ),
    "duacs.alongtrack.s3b": CatalogEntry(
        source="cmems",
        dataset_id="cmems_obs-sl_glo_phy-ssh_my_s3b-l3-duacs_PT1S_202411",
    ),
    "duacs.alongtrack.s6a": CatalogEntry(
        source="cmems",
        dataset_id="cmems_obs-sl_glo_phy-ssh_my_s6a-lr-l3-duacs_PT1S_202411",
    ),
    "duacs.alongtrack.swot": CatalogEntry(
        source="cmems",
        dataset_id="cmems_obs-sl_glo_phy-ssh_my_swon-l3-duacs_PT1S_202411",
    ),
    # ---- CMEMS — SST --------------------------------------------------
    "ostia.sst": CatalogEntry(
        source="cmems",
        dataset_id="METOFFICE-GLO-SST-L4-REP-OBS-SST",
    ),
    "odyssea.sst": CatalogEntry(
        source="cmems",
        dataset_id="cmems_obs-sst_glo_phy_my_l3s_P1D-m_202311",
    ),
    # ---- CMEMS — SSS --------------------------------------------------
    "multiobs.sss.daily": CatalogEntry(
        source="cmems",
        dataset_id="cmems_obs-mob_glo_phy-sss_my_multi_P1D",
    ),
    "multiobs.sss.monthly": CatalogEntry(
        source="cmems",
        dataset_id="cmems_obs-mob_glo_phy-sss_my_multi_P1M",
    ),
    # ---- CMEMS — ocean colour -----------------------------------------
    "globcolour.chl.monthly": CatalogEntry(
        source="cmems",
        dataset_id="cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M",
    ),
    "globcolour.chl.daily": CatalogEntry(
        source="cmems",
        dataset_id="cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D",
    ),
    "globcolour.transparency": CatalogEntry(
        source="cmems",
        dataset_id="cmems_obs-oc_glo_bgc-transp_my_l4-multi-4km_P1M",
    ),
    "globcolour.optics": CatalogEntry(
        source="cmems",
        dataset_id="cmems_obs-oc_glo_bgc-optics_my_l4-multi-4km_P1M",
    ),
    "globcolour.reflectance": CatalogEntry(
        source="cmems",
        dataset_id="cmems_obs-oc_glo_bgc-reflectance_my_l4-multi-4km_P1M",
    ),
    "globcolour.pp": CatalogEntry(
        source="cmems",
        dataset_id="cmems_obs-oc_glo_bgc-pp_my_l4-multi-4km_P1M",
    ),
    # ---- CMEMS — in-situ ----------------------------------------------
    "cora.ts": CatalogEntry(
        source="cmems",
        dataset_id="cmems_obs-ins_glo_phy-temp-sal_my_cora_irr",
    ),
    "easycora.ts": CatalogEntry(
        source="cmems",
        dataset_id="cmems_obs-ins_glo_phy-temp-sal_my_easycora_irr",
    ),
    # ---- CMEMS — biogeochemistry --------------------------------------
    "glorys12.bgc.daily": CatalogEntry(
        source="cmems",
        dataset_id="cmems_mod_glo_bgc_my_0.25deg_P1D-m_202406",
    ),
    "glorys12.bgc.monthly": CatalogEntry(
        source="cmems",
        dataset_id="cmems_mod_glo_bgc_my_0.25deg_P1M-m_202406",
    ),
    # ---- CDS — ERA5 ---------------------------------------------------
    "era5.single_levels": CatalogEntry(
        source="cds",
        dataset_id="reanalysis-era5-single-levels",
    ),
    "era5.pressure_levels": CatalogEntry(
        source="cds",
        dataset_id="reanalysis-era5-pressure-levels",
    ),
    "era5.land": CatalogEntry(
        source="cds",
        dataset_id="reanalysis-era5-land",
    ),
    # ---- AEMET — station observations ---------------------------------
    "aemet.stations": CatalogEntry(source="aemet", dataset_id="aemet_stations"),
    "aemet.daily": CatalogEntry(source="aemet", dataset_id="aemet_daily"),
    "aemet.hourly": CatalogEntry(source="aemet", dataset_id="aemet_hourly"),
    "aemet.monthly": CatalogEntry(source="aemet", dataset_id="aemet_monthly"),
    "aemet.normals": CatalogEntry(source="aemet", dataset_id="aemet_normals"),
    "aemet.extremes": CatalogEntry(source="aemet", dataset_id="aemet_extremes"),
    "aemet.pollution": CatalogEntry(source="aemet", dataset_id="aemet_pollution"),
}


def all_entries() -> dict[str, CatalogEntry]:
    """Return a shallow copy of the unified catalog."""
    return dict(CATALOG)


def describe(name: str) -> DatasetInfo:
    """Look up a :class:`DatasetInfo` by short name."""
    entry = CATALOG[name]
    if entry.source == "cmems":
        return CMEMS_DATASETS[entry.dataset_id]
    if entry.source == "cds":
        return CDS_DATASETS[entry.dataset_id]
    if entry.source == "aemet":
        return AEMET_DATASETS[entry.dataset_id]
    raise KeyError(f"Unknown source {entry.source!r}")
