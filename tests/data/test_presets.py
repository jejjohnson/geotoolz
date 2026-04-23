"""Coverage for the CMEMS preset catalog split."""

from __future__ import annotations

import pytest

from xr_toolz.data import CATALOG, DatasetKind, describe
from xr_toolz.data._src.cmems.catalog import CMEMS_DATASETS
from xr_toolz.data._src.cmems.presets.bgc import BGC_DATASETS
from xr_toolz.data._src.cmems.presets.insitu import INSITU_DATASETS
from xr_toolz.data._src.cmems.presets.oc import OC_DATASETS
from xr_toolz.data._src.cmems.presets.phy import PHY_DATASETS
from xr_toolz.data._src.cmems.presets.ssh import SSH_DATASETS
from xr_toolz.data._src.cmems.presets.sss import SSS_DATASETS
from xr_toolz.data._src.cmems.presets.sst import SST_DATASETS


# ---- Preset shapes --------------------------------------------------------


@pytest.mark.parametrize(
    ("preset", "name"),
    [
        (PHY_DATASETS, "phy"),
        (SSH_DATASETS, "ssh"),
        (SST_DATASETS, "sst"),
        (SSS_DATASETS, "sss"),
        (OC_DATASETS, "oc"),
        (INSITU_DATASETS, "insitu"),
        (BGC_DATASETS, "bgc"),
    ],
)
def test_each_preset_file_is_non_empty_and_well_typed(preset, name):
    assert preset, f"{name} preset is empty"
    for key, info in preset.items():
        # Key == dataset_id (keeps lookups O(1)).
        assert info.dataset_id == key
        assert info.source == "cmems"
        assert info.variables, f"{key} has no variables"
        assert isinstance(info.kind, DatasetKind)


def test_catalog_unions_all_presets_without_collisions():
    merged_size = (
        len(PHY_DATASETS)
        + len(SSH_DATASETS)
        + len(SST_DATASETS)
        + len(SSS_DATASETS)
        + len(OC_DATASETS)
        + len(INSITU_DATASETS)
        + len(BGC_DATASETS)
    )
    assert len(CMEMS_DATASETS) == merged_size


# ---- DatasetKind semantics ------------------------------------------------


def test_insitu_is_tagged_profiles_not_gridded():
    for info in INSITU_DATASETS.values():
        assert info.kind is DatasetKind.PROFILES


def test_alongtrack_ssh_is_tagged_alongtrack():
    alongtrack = [i for i in SSH_DATASETS.values() if "l3-duacs" in i.dataset_id]
    assert alongtrack, "expected at least one along-track L3 entry"
    for info in alongtrack:
        assert info.kind is DatasetKind.ALONGTRACK


def test_gridded_l4_products_default_to_gridded():
    assert (
        PHY_DATASETS["cmems_mod_glo_phy_my_0.083deg_P1D-m"].kind is DatasetKind.GRIDDED
    )
    assert SST_DATASETS["METOFFICE-GLO-SST-L4-REP-OBS-SST"].kind is DatasetKind.GRIDDED


# ---- Short-name catalog --------------------------------------------------


@pytest.mark.parametrize(
    "short_name",
    [
        "glorys12.daily",
        "duacs.sla",
        "duacs.alongtrack.s3a",
        "ostia.sst",
        "odyssea.sst",
        "multiobs.sss.daily",
        "globcolour.chl.monthly",
        "globcolour.transparency",
        "globcolour.reflectance",
        "cora.ts",
        "glorys12.bgc.monthly",
    ],
)
def test_short_name_resolves_to_catalogued_dataset(short_name):
    entry = CATALOG[short_name]
    info = describe(short_name)
    assert info.dataset_id == entry.dataset_id
    assert info.source == entry.source


def test_variable_aliases_translate_for_cmems_products():
    # The catalog must expose variables whose CMEMS alias is well-defined.
    info = describe("duacs.sla")
    cmems_names = {v.for_source("cmems") for v in info.variables}
    assert {"sla", "adt", "ugos", "vgos"} <= cmems_names
