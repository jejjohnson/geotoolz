"""Curated catalog of local L2 products.

Each entry names a product *opener* rather than a remote dataset: the
``dataset_id`` is the same short name that :data:`xrreader.CATALOG`
uses, and :class:`~xrreader.LocalL2Source` maps it to the function that
turns one file into a flat CF Dataset.
"""

from __future__ import annotations

from xrreader._src.base import DatasetInfo, DatasetKind
from xrreader.types import (
    CH4_ENHANCEMENT,
    CH4_PRIOR_PROFILE,
    COLUMN_AK,
    DRY_AIR_SUBCOLUMNS,
    QA_VALUE,
    SP,
    XCH4,
    XCH4_BC,
    XCH4_PRECISION,
)


LOCAL_DATASETS: dict[str, DatasetInfo] = {
    "tropomi.ch4": DatasetInfo(
        dataset_id="tropomi.ch4",
        source="local",
        title="Sentinel-5P TROPOMI CH4 L2 (OFFL / RPRO)",
        kind=DatasetKind.SWATH,
        variables=(
            XCH4,
            XCH4_BC,
            XCH4_PRECISION,
            COLUMN_AK,
            CH4_PRIOR_PROFILE,
            DRY_AIR_SUBCOLUMNS,
            QA_VALUE,
            SP,
        ),
        spatial_coverage=None,
        temporal_coverage=("2018-04-30", "present"),
        doi="10.5270/S5P-3lcdqiv",
        license="Copernicus Sentinel data — free and open",
        notes=(
            "One granule per orbit on (time, scanline, ground_pixel) with "
            "a 'layer' axis for the averaging kernel / a-priori profile. "
            "Flattened from the PRODUCT + SUPPORT_DATA groups by "
            "open_tropomi_ch4_l2; qa_value < 0.5 is masked by default."
        ),
    ),
    "emit.ch4": DatasetInfo(
        dataset_id="emit.ch4",
        source="local",
        title="EMIT L2B methane enhancement (CH4ENH / CH4UNCERT)",
        kind=DatasetKind.SCENE,
        variables=(CH4_ENHANCEMENT,),
        temporal_coverage=("2022-08-01", "present"),
        doi="10.5067/EMIT/EMITL2BCH4ENH.001",
        license="NASA Earthdata — open",
        notes=(
            "Per-scene plume enhancement raster (ppm m) on (y, x); "
            "orthorectified through the geometric lookup table when "
            "glt_path is given. NetCDF container assumed."
        ),
    ),
    "ghgsat.ch4": DatasetInfo(
        dataset_id="ghgsat.ch4",
        source="local",
        title="GHGSat CH4 L2 per-plume product",
        kind=DatasetKind.SCENE,
        variables=(XCH4, CH4_ENHANCEMENT),
        temporal_coverage=("2016-06-01", "present"),
        license="GHGSat data licence (restricted)",
        notes=(
            "Per-plume NetCDF on (y, x) with 2-D lon/lat; carries xch4 "
            "(ppb) and/or ch4_enhancement depending on the release."
        ),
    ),
}
