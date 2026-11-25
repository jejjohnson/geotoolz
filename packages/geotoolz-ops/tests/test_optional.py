"""Imports of an optional extra: the module, or an error that names the extra."""

from __future__ import annotations

import json.decoder
from pathlib import Path

import pytest

from geoops.carrier import optional_module, optional_name, optional_root


def test_an_installed_module() -> None:
    assert optional_module("json.decoder", "base", feature="doc") is json.decoder
    assert optional_root("json.decoder", "base", feature="doc") is json
    assert optional_name("json", "loads", "base", feature="doc") is json.loads


def test_a_missing_module_names_the_caller_and_the_extra() -> None:
    expected = (
        "ModelOp.save needs 'absent', which is not installed; "
        "install it with: pip install 'geotoolz-ops[learn]'"
    )
    for call in (
        lambda: optional_module("absent", "learn", feature="ModelOp.save"),
        lambda: optional_root("absent.part", "learn", feature="ModelOp.save"),
        lambda: optional_name("absent", "name", "learn", feature="ModelOp.save"),
    ):
        with pytest.raises(ImportError) as caught:
            call()
        assert str(caught.value) == expected


def test_a_broken_module_keeps_its_own_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "gt_broken.py").write_text("raise ImportError('wrong version')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ImportError, match=r"^wrong version$"):
        optional_module("gt_broken", "base", feature="doc")
