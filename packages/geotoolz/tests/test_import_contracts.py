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


GEOMODELS = "geotoolz never imports plumax or radtrax"
NO_JAX = "catalog, cloud and products never import JAX or coordax"


@pytest.mark.parametrize(
    ("name", "module"),
    [
        (GEOMODELS, "plumax"),
        (GEOMODELS, "radtrax"),
        (NO_JAX, "jax"),
        (NO_JAX, "coordax"),
    ],
)
def test_a_lazy_import_breaks_the_contract(
    tmp_path: Path, name: str, module: str
) -> None:
    """The root contract, pointed at a package that imports ``module`` in a function."""
    contract = _contract(name)
    package = tmp_path / "fakegeo"
    package.mkdir()
    (package / "__init__.py").write_text(f"def later():\n    import {module}\n")
    forbidden = ", ".join(f'"{m}"' for m in contract["forbidden_modules"])
    indirect = str(contract.get("allow_indirect_imports", False)).lower()
    (tmp_path / ".importlinter.toml").write_text(
        '[tool.importlinter]\nroot_packages = ["fakegeo"]\n'
        "include_external_packages = true\n\n"
        "[[tool.importlinter.contracts]]\n"
        f'name = "{contract["name"]}"\ntype = "{contract["type"]}"\n'
        f'source_modules = ["fakegeo"]\nforbidden_modules = [{forbidden}]\n'
        f"allow_indirect_imports = {indirect}\n"
    )
    result = _lint("--config", ".importlinter.toml", cwd=tmp_path, env_path=tmp_path)
    assert result.returncode != 0
    assert f"fakegeo -> {module}" in result.stdout


def test_io_packages_do_not_load_jax_even_when_installed() -> None:
    """Gate G4 (#528), the runtime half: importing them never loads JAX.

    geopatcher too (#529): its batches are NumPy until ``to_jax()``; only
    the `[train]` integration imports JAX, when it runs.
    """
    pytest.importorskip("jax")
    code = (
        "import sys\n"
        "import geocatalog, geocloud, geoproducts\n"
        "import geopatcher, geopatcher.run, geopatcher.fields\n"
        "loaded = {'jax', 'jaxlib', 'coordax'} & set(sys.modules)\n"
        "assert not loaded, f'loaded {sorted(loaded)}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
