"""What CI runs for a change (scripts/ci_select.py), on real and typical PRs."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "ci_select.py"
spec = importlib.util.spec_from_file_location("ci_select", SCRIPT)
assert spec is not None and spec.loader is not None
cs = importlib.util.module_from_spec(spec)
# dataclasses look their module up in sys.modules
sys.modules[spec.name] = cs
spec.loader.exec_module(cs)

WS = cs.workspace()
ALL = tuple(sorted(WS))
GEO = ("geotoolz-array", "geotoolz-cloud", "geotoolz-ops", "geotoolz-patcher")


def test_package_directories_are_their_names() -> None:
    # the selection maps packages/<directory>/ to a package by its name
    dirs = {p.parent.name for p in (cs.ROOT / "packages").glob("*/pyproject.toml")}
    assert dirs == set(WS)


def test_the_graph_is_read_from_the_packages() -> None:
    reach = {name: sorted(cs.reach(WS, name) - {name}) for name in WS}
    assert reach == {
        "geotoolz": sorted(n for n in ALL if n.startswith("geotoolz-")),
        "geotoolz-array": ["geotoolz-cloud", "geotoolz-ops", "geotoolz-patcher"],
        "geotoolz-catalog": ["geotoolz-array", "geotoolz-cloud", "geotoolz-patcher"],
        "geotoolz-cloud": ["geotoolz-array"],
        "geotoolz-ops": ["geotoolz-array", "geotoolz-cloud", "geotoolz-patcher"],
        "geotoolz-patcher": ["geotoolz-array", "geotoolz-cloud", "geotoolz-ops"],
        "geotoolz-products": [*GEO],
        "xrtoolz": [*GEO, "xrtoolz-ops", "xrtoolz-products"],
        "xrtoolz-ops": [*GEO, "xrtoolz-products"],
        "xrtoolz-products": [],
    }


#: name -> (changed files, packages tested, docs build, repo checks, packaging)
CASES = {
    # jejjohnson/geotoolz#676
    "docs only": (
        [
            "README.md",
            "docs/assets/diagrams/geostack-intro.html",
            "docs/assets/diagrams/geostack-intro.png",
            "docs/index.md",
        ],
        (),
        True,
        True,
        False,
    ),
    "one geoops module": (
        ["packages/geotoolz-ops/src/geoops/indices/_src/ndvi.py"],
        (
            "geotoolz",
            "geotoolz-array",
            "geotoolz-ops",
            "geotoolz-patcher",
            "geotoolz-products",
            "xrtoolz",
            "xrtoolz-ops",
        ),
        True,
        True,
        True,
    ),
    "one geoops test": (
        ["packages/geotoolz-ops/tests/test_indices.py"],
        ("geotoolz-ops",),
        False,
        False,
        False,
    ),
    "one geoarray module": (
        ["packages/geotoolz-array/src/geoarray/_src/grid.py"],
        tuple(n for n in ALL if n != "xrtoolz-products"),
        True,
        True,
        True,
    ),
    "one geocatalog module": (
        ["packages/geotoolz-catalog/src/geocatalog/_src/build.py"],
        ("geotoolz", "geotoolz-catalog"),
        True,
        True,
        True,
    ),
    "one xrproducts module": (
        ["packages/xrtoolz-products/src/xrproducts/cmems.py"],
        ("xrtoolz", "xrtoolz-ops", "xrtoolz-products"),
        True,
        True,
        True,
    ),
    "a package README": (["packages/geotoolz-cloud/README.md"], (), True, True, False),
    "a changelog": (["packages/xrtoolz/CHANGELOG.md"], (), True, False, False),
    "notebooks only": (["notebooks/xarray/example.ipynb"], (), False, False, False),
    "a script": (["scripts/capabilities.py"], (), False, True, False),
    "mkdocs.yml": (["mkdocs.yml"], (), True, False, False),
    "a LICENSE": (
        ["packages/geotoolz-ops/LICENSE"],
        ("geotoolz-ops",),
        False,
        False,
        True,
    ),
    "the lock": (["uv.lock"], ALL, True, True, True),
    # jejjohnson/geotoolz#655 changed the lock with geoops
    "geoops with the lock": (
        ["packages/geotoolz-ops/src/geoops/geom/_src/grid.py", "uv.lock"],
        ALL,
        True,
        True,
        True,
    ),
    "the tests workflow": ([".github/workflows/ci.yml"], ALL, False, True, True),
    "a new package": (["packages/geotoolz-new/pyproject.toml"], ALL, True, True, True),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_pull_request_runs_what_it_can_affect(case: str) -> None:
    files, packages, docs, checks, build = CASES[case]
    got = cs.select(files, pull_request=True, packages=WS)
    assert (got.packages, got.docs, got.checks, got.build) == (
        tuple(sorted(packages)),
        docs,
        checks,
        build,
    )
    assert got.pythons == ("3.13",)


def test_main_runs_everything_on_both_pythons() -> None:
    got = cs.select([], pull_request=False, packages=WS)
    assert got.packages == ALL and got.pythons == ("3.12", "3.13")
    assert got.docs and got.checks and got.build


def test_a_workflow_file_runs_that_workflow_in_full() -> None:
    docs_only = ["docs/index.md"]
    base = cs.select(
        [*docs_only, ".github/workflows/base-install.yml"],
        pull_request=True,
        packages=WS,
    )
    assert base.base == ALL and base.typecheck == () and base.packages == ()
    ty = cs.select([".github/workflows/typecheck.yml"], pull_request=True, packages=WS)
    assert ty.typecheck == ALL and ty.base == ()
    docs = cs.select([".github/workflows/docs.yml"], pull_request=True, packages=WS)
    assert docs.docs and docs.packages == ()


def test_too_many_files_to_list_runs_everything() -> None:
    files = [f"docs/page{i}.md" for i in range(cs.API_FILE_LIMIT)]
    assert cs.select(files, pull_request=True, packages=WS).packages == ALL


def test_outputs_are_what_the_workflows_read() -> None:
    out = cs.select(
        ["packages/xrtoolz-products/src/xrproducts/cmems.py"],
        pull_request=True,
        packages=WS,
    ).outputs()
    assert out["any"] == "true" and out["docs"] == "true"
    assert '{"package":"xrtoolz-ops","python":"3.13"}' in out["matrix"]
    assert out["base"] == '["xrtoolz", "xrtoolz-ops", "xrtoolz-products"]'
    empty = cs.select(
        ["notebooks/xarray/example.ipynb"], pull_request=True, packages=WS
    ).outputs()
    assert empty["any"] == "false" and empty["matrix"] == '{"include":[]}'
