"""Select what CI runs for a change: which packages to test, and which checks.

Every workflow takes its selection from here (``.github/workflows/changes.yml``),
so a change runs the jobs it can affect and nothing else.

- On ``main`` (any event but a pull request) everything runs, on Python 3.12
  and 3.13. A pull request runs on 3.13.
- Shared config (the root ``pyproject.toml``, ``uv.lock``, ``conftest.py``,
  this script and ``changes.yml``) runs everything.
- A change under ``packages/<name>/`` runs that package, except its README
  and changelog, which only the docs build and checks read. A change to its
  ``src/`` or ``pyproject.toml`` also runs every package that reaches it:
  through its dependencies and extras (CI installs every extra), or through
  what its tests import. The graph is read from the packages themselves.
- A workflow file runs that workflow in full.
- The docs site builds when its inputs change: ``docs/``, ``mkdocs.yml``,
  any package's ``src/`` (mkdocstrings renders the docstrings), changelog or
  README.
- The repository checks (capability index, docs examples, docs paths,
  dependency rules, import contracts, family parity) run on any package's
  ``src/`` or ``pyproject.toml``, and on ``docs/``, the READMEs and
  ``scripts/``.
- The packaging check runs on any package's ``src/``, ``pyproject.toml`` or
  LICENSE.
- Notebooks alone run nothing.

    uv run python scripts/ci_select.py --base origin/main   # what a PR from HEAD runs
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import cyclopts
from loguru import logger


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ".github/workflows/"
#: files that can change what any job does
SHARED = {
    "pyproject.toml",
    "uv.lock",
    "conftest.py",
    ".python-version",
    "scripts/ci_select.py",
    WORKFLOWS + "changes.yml",
}
#: the GitHub API lists at most this many files of a pull request
API_FILE_LIMIT = 3000

app = cyclopts.App(help=__doc__.split("\n\n")[0])


@dataclass(frozen=True)
class Package:
    name: str
    modules: frozenset[str]
    #: workspace packages named by its dependencies or any of its extras
    deps: frozenset[str]
    #: workspace packages its tests import (beyond the package itself)
    test_imports: frozenset[str] = field(default=frozenset())


@dataclass(frozen=True)
class Selection:
    packages: tuple[str, ...]
    pythons: tuple[str, ...]
    base: tuple[str, ...]
    typecheck: tuple[str, ...]
    docs: bool
    checks: bool
    build: bool

    def outputs(self) -> dict[str, str]:
        """The GitHub step outputs, as ``name=value`` strings."""
        include = [
            {"package": p, "python": v} for p in self.packages for v in self.pythons
        ]
        return {
            "matrix": json.dumps({"include": include}, separators=(",", ":")),
            "any": str(bool(include)).lower(),
            "packages": json.dumps(list(self.packages)),
            "base": json.dumps(list(self.base)),
            "typecheck": json.dumps(list(self.typecheck)),
            "docs": str(self.docs).lower(),
            "checks": str(self.checks).lower(),
            "build": str(self.build).lower(),
        }


def _norm(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def _requirement_name(spec: str) -> str:
    for stop in "[<>=!~ ;@(":
        spec = spec.split(stop, 1)[0]
    return _norm(spec.strip())


def _imported_modules(path: Path) -> set[str]:
    """Top-level modules a file imports, including ``importorskip("x")``."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and node.args:
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", "")
            )
            arg = node.args[0]
            literal = isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            if name in ("importorskip", "import_module") and literal:
                found.add(arg.value.split(".")[0])
    return found


def workspace(root: Path = ROOT) -> dict[str, Package]:
    """Every workspace package with its dependencies and its tests' imports."""
    metas: dict[str, tuple[Path, dict]] = {}
    for pyproject in sorted((root / "packages").glob("*/pyproject.toml")):
        meta = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        metas[_norm(meta["project"]["name"])] = (pyproject.parent, meta["project"])
    module_of: dict[str, str] = {}
    for name, (path, _) in metas.items():
        for module in (path / "src").glob("*"):
            if module.is_dir() or module.suffix == ".py":
                module_of[module.stem] = name
    packages: dict[str, Package] = {}
    for name, (path, project) in metas.items():
        specs = list(project.get("dependencies", []))
        for extra in project.get("optional-dependencies", {}).values():
            specs.extend(extra)
        deps = {_requirement_name(s) for s in specs} & metas.keys()
        imported: set[str] = set()
        for f in (path / "tests").rglob("*.py") if (path / "tests").is_dir() else []:
            imported |= _imported_modules(f)
        tests = {module_of[m] for m in imported if m in module_of}
        packages[name] = Package(
            name=name,
            modules=frozenset(m for m, p in module_of.items() if p == name),
            deps=frozenset(deps - {name}),
            test_imports=frozenset(tests - {name}),
        )
    return packages


def closure(packages: Mapping[str, Package], start: Iterable[str]) -> set[str]:
    """The packages in ``start`` and everything they depend on, transitively."""
    seen: set[str] = set()
    todo = list(start)
    while todo:
        name = todo.pop()
        if name in seen or name not in packages:
            continue
        seen.add(name)
        todo.extend(packages[name].deps)
    return seen


def reach(packages: Mapping[str, Package], name: str) -> set[str]:
    """What a change can reach ``name`` through: its closure and its tests'."""
    return closure(packages, {name, *packages[name].test_imports})


def _package_of(path: str) -> tuple[str, str] | None:
    """(package directory, the rest of the path) for a path under packages/."""
    parts = path.split("/")
    if len(parts) >= 3 and parts[0] == "packages":
        return parts[1], "/".join(parts[2:])
    return None


def select(
    files: Iterable[str],
    *,
    pull_request: bool,
    packages: Mapping[str, Package] | None = None,
) -> Selection:
    """What a change to ``files`` runs."""
    packages = packages if packages is not None else workspace()
    names = tuple(sorted(packages))
    everything = Selection(names, ("3.12", "3.13"), names, names, True, True, True)
    if not pull_request:
        return everything
    files = sorted(set(files))
    if len(files) >= API_FILE_LIMIT or any(f in SHARED for f in files):
        return Selection(names, ("3.13",), names, names, True, True, True)

    own: set[str] = set()
    lib: set[str] = set()
    docs = checks = build = False
    workflows = {f.removeprefix(WORKFLOWS) for f in files if f.startswith(WORKFLOWS)}
    for f in files:
        where = _package_of(f)
        if where is not None:
            directory, rest = where
            if directory not in packages:
                # a package added or removed: its graph is not the one on disk
                return Selection(names, ("3.13",), names, names, True, True, True)
            if rest in ("CHANGELOG.md", "README.md"):
                # read only by the docs build and the docs-path checks
                docs = True
                checks = checks or rest == "README.md"
                continue
            own.add(directory)
            if rest.startswith("src/") or rest == "pyproject.toml":
                lib.add(directory)
                docs = checks = build = True
            if rest == "LICENSE":
                build = True
        elif f.startswith("docs/") or f == "mkdocs.yml":
            docs = True
            checks = checks or f.startswith("docs/")
        elif f == "README.md" or f.startswith("scripts/"):
            checks = True
        elif f == "LICENSE":
            build = True

    selected = {name for name in packages if name in own or reach(packages, name) & lib}
    if "ci.yml" in workflows:
        selected = set(names)
        checks = build = True
    chosen = tuple(sorted(selected))
    return Selection(
        packages=chosen,
        pythons=("3.13",),
        base=names if "base-install.yml" in workflows else chosen,
        typecheck=names if "typecheck.yml" in workflows else chosen,
        docs=docs or "docs.yml" in workflows,
        checks=checks,
        build=build,
    )


def _changed_files(base: str) -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in out.stdout.splitlines() if line]


@app.default
def main(
    *,
    event: str = "pull_request",
    base: str | None = None,
    files_from: Path | None = None,
) -> None:
    """Print the selection, and write it to ``$GITHUB_OUTPUT`` when set.

    Args:
        event: The GitHub event; anything but ``pull_request`` runs everything.
        base: Compare ``HEAD`` with this ref (``git diff BASE...HEAD``).
        files_from: A file listing the changed paths, one per line (CI).
    """
    pull_request = event == "pull_request"
    files: list[str] = []
    if pull_request:
        if files_from is not None:
            files = [
                line.strip()
                for line in files_from.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        elif base is not None:
            files = _changed_files(base)
        else:
            logger.error("a pull request needs --base or --files-from")
            sys.exit(2)
    selection = select(files, pull_request=pull_request)
    logger.info(f"{len(files)} changed files ({event})")
    for key, value in selection.outputs().items():
        logger.info(f"{key}: {value}")
    target = os.environ.get("GITHUB_OUTPUT")
    if target:
        with open(target, "a", encoding="utf-8") as out:
            for key, value in selection.outputs().items():
                out.write(f"{key}={value}\n")


if __name__ == "__main__":
    app()
