"""The umbrella installs the five packages and exports nothing of its own."""

from __future__ import annotations

import importlib

import pytest

import geotoolz


@pytest.mark.parametrize(
    "module", ["geoops", "geopatcher", "geocatalog", "geocloud", "geoproducts"]
)
def test_each_package_imports(module: str) -> None:
    assert importlib.import_module(module).__name__ == module


def test_umbrella_exports_only_its_version() -> None:
    assert geotoolz.__all__ == ["__version__"]


def test_old_core_paths_are_gone() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("geotoolz.carrier")
