"""Curated catalog of Copernicus Marine products.

Assembled from per-family preset modules under
:mod:`xrreader._src.cmems.presets`. Each family keeps its own
module so the catalog stays legible — add new products to the
appropriate preset file, not here.
"""

from __future__ import annotations

from xrreader._src.base import DatasetInfo
from xrreader._src.cmems.presets.bgc import BGC_DATASETS
from xrreader._src.cmems.presets.insitu import INSITU_DATASETS
from xrreader._src.cmems.presets.oc import OC_DATASETS
from xrreader._src.cmems.presets.phy import PHY_DATASETS
from xrreader._src.cmems.presets.ssh import SSH_DATASETS
from xrreader._src.cmems.presets.sss import SSS_DATASETS
from xrreader._src.cmems.presets.sst import SST_DATASETS


CMEMS_DATASETS: dict[str, DatasetInfo] = {
    **PHY_DATASETS,
    **SSH_DATASETS,
    **SST_DATASETS,
    **SSS_DATASETS,
    **OC_DATASETS,
    **INSITU_DATASETS,
    **BGC_DATASETS,
}
