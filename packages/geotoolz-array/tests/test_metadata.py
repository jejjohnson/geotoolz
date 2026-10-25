"""The published metadata asks for every pipekit API geoarray imports."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version


PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# The first release of each that geoarray needs: pipekit 0.1.3 ships
# `pipekit.on_import` (the lazy JAX pytree registration); pipekit-array 0.0.5
# registers `DimArray` the same way, so neither imports JAX on import.
NEEDED = {"pipekit": Version("0.1.3"), "pipekit-array": Version("0.0.5")}


@pytest.mark.parametrize("name", sorted(NEEDED))
def test_dependency_floor_provides_the_api_geoarray_imports(name: str) -> None:
    deps = tomllib.loads(PYPROJECT.read_text())["project"]["dependencies"]
    (requirement,) = [Requirement(dep) for dep in deps if Requirement(dep).name == name]
    too_old = [
        version
        for version in (Version("0.0.1"), Version("0.0.4"), Version("0.1.2"))
        if version < NEEDED[name] and requirement.specifier.contains(version)
    ]
    assert not too_old, f"{requirement} admits {too_old}, older than {NEEDED[name]}"
