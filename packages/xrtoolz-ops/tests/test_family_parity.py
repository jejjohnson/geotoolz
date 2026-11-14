"""Gate D1: a family name shared by geoops and xrops means the same job (#362)."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "family_parity.py"
spec = importlib.util.spec_from_file_location("family_parity", SCRIPT)
assert spec is not None and spec.loader is not None
fp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp)


def test_every_shared_family_names_its_job() -> None:
    geo, xr = fp.families("geoops"), fp.families("xrops")
    assert fp.problems(geo, xr, fp.SAME_JOB) == []
    assert set(geo) & set(xr) == set(fp.SAME_JOB)


def test_the_checked_in_table_is_current() -> None:
    geo, xr = fp.families("geoops"), fp.families("xrops")
    assert fp.PAGE.read_text(encoding="utf-8") == fp.table(geo, xr, fp.SAME_JOB), (
        "docs/api/families.md is stale: run scripts/family_parity.py"
    )


def test_a_renamed_family_fails_the_check() -> None:
    geo, xr = fp.families("geoops"), fp.families("xrops")
    # xrops renames its masks to geoops's "mask"-less word, and takes "qa"
    renamed = {("masking" if k == "mask" else k): v for k, v in xr.items()}
    renamed["qa"] = "quality flags"
    found = fp.problems(geo, renamed, fp.SAME_JOB)
    assert any("'qa' is a family of both" in p for p in found)
    assert any("'mask' is in SAME_JOB but is no longer" in p for p in found)
