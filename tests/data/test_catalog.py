"""Unified catalog lookup."""

from __future__ import annotations

import pytest

from xr_toolz.data import CATALOG, all_entries, describe


def test_catalog_contains_expected_short_names():
    assert "era5.single_levels" in CATALOG
    assert "glorys12.daily" in CATALOG


def test_describe_returns_dataset_info_for_known_name():
    info = describe("era5.single_levels")
    assert info.source == "cds"
    assert info.dataset_id == "reanalysis-era5-single-levels"


def test_describe_rejects_unknown_name():
    with pytest.raises(KeyError):
        describe("not_a_real_shortname")


def test_all_entries_returns_copy():
    snap = all_entries()
    snap["junk"] = snap["era5.single_levels"]  # must not affect module-level dict
    assert "junk" not in CATALOG
