"""Import contracts from the root ``[tool.importlinter]`` (gate P2, #481)."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
LINT_IMPORTS = Path(sys.executable).with_name("lint-imports")


def _contract(name: str) -> dict:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"][
        "importlinter"
    ]
    return next(c for c in config["contracts"] if c["name"] == name)


def _lint(
    *args: str, cwd: Path, env_path: Path | None = None
) -> subprocess.CompletedProcess:
    if not LINT_IMPORTS.exists():
        pytest.fail("lint-imports is not installed: uv sync --group lint")
    env = None if env_path is None else {"PYTHONPATH": str(env_path), "PATH": ""}
    return subprocess.run(
        [str(LINT_IMPORTS), *args], cwd=cwd, env=env, capture_output=True, text=True
    )


def test_the_contracts_hold() -> None:
    result = _lint(cwd=ROOT)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("module", ["plumax", "radtrax"])
def test_a_lazy_import_of_a_geomodels_package_breaks_the_contract(
    tmp_path: Path, module: str
) -> None:
    """The root contract, pointed at a package that imports ``module`` in a function."""
    contract = _contract("geotoolz never imports plumax or radtrax")
    package = tmp_path / "fakegeo"
    package.mkdir()
    (package / "__init__.py").write_text(f"def later():\n    import {module}\n")
    forbidden = ", ".join(f'"{m}"' for m in contract["forbidden_modules"])
    (tmp_path / ".importlinter.toml").write_text(
        '[tool.importlinter]\nroot_packages = ["fakegeo"]\n'
        "include_external_packages = true\n\n"
        "[[tool.importlinter.contracts]]\n"
        f'name = "{contract["name"]}"\ntype = "{contract["type"]}"\n'
        f'source_modules = ["fakegeo"]\nforbidden_modules = [{forbidden}]\n'
    )
    result = _lint("--config", ".importlinter.toml", cwd=tmp_path, env_path=tmp_path)
    assert result.returncode != 0
    assert f"fakegeo -> {module}" in result.stdout
