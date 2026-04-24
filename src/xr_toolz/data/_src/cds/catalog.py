"""Curated catalog of Climate Data Store products.

Assembled from per-family preset modules under
:mod:`xr_toolz.data._src.cds.presets`. Each family keeps its own
module so the catalog stays legible — add new products to the
appropriate preset file, not here.
"""

from __future__ import annotations

from xr_toolz.data._src.base import DatasetInfo
from xr_toolz.data._src.cds.presets.insitu import INSITU_DATASETS
from xr_toolz.data._src.cds.presets.reanalysis import REANALYSIS_DATASETS


CDS_DATASETS: dict[str, DatasetInfo] = {
    **REANALYSIS_DATASETS,
    **INSITU_DATASETS,
}
