"""Generate, or check, the capability index: every public name in the stack.

``docs/capabilities.md`` lists each public object of the five packages once,
at its deepest public home, with the first line of its docstring, followed by
the public API of the upstream composition framework (pipekit_ops) and the shared
private toolkits contributors build on. Agents and people search
it before writing a helper ("reuse before you write", ``AGENTS.md``).

It also writes ``docs/operators/api/backends.md``, the array backends each
public geoops operator runs on (its ``backends``), and
``docs/operators/api/coupling.md``, which dims each one mixes (its
``coupling``).

Usage::

    make capabilities                                   # rewrite the index
    uv run python scripts/capabilities.py --check       # CI: fail if stale

Run it in the full environment (``make install``: every package and extra);
a reader whose extra is missing refuses to import.

``--check`` also fails when two packages export the same bare name for
different objects, outside the patterns the layout allows on purpose
(``ALLOWED_SHARED_NAMES`` below), or when a package exports a name that
pipekit_ops already exports for something else (outside the same allowlist).
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import pkgutil
import sys
import types
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "capabilities.md"
BACKENDS = ROOT / "docs" / "operators" / "api" / "backends.md"
COUPLING = ROOT / "docs" / "operators" / "api" / "coupling.md"

PACKAGES = {
    "geoops": "geotoolz-ops",
    "geopatcher": "geotoolz-patcher",
    "geocatalog": "geotoolz-catalog",
    "geocloud": "geotoolz-cloud",
    "geoproducts": "geotoolz-products",
    "geoarray": "geotoolz-array",
}

# Upstream frameworks whose public API is listed (and whose names our
# packages must not shadow): the Operator / Sequential / Graph core.
UPSTREAM = ("pipekit_ops",)

# Shared private toolkits: modules whose helpers every family / reader of the
# package builds on. Listed by path, read with ``ast`` (nothing imported).
TOOLKITS = {
    "geoops": ["packages/geotoolz-ops/src/geoops/_src"],
    "geocatalog": ["packages/geotoolz-catalog/src/geocatalog/_src/utils"],
    "geoproducts": ["packages/geotoolz-products/src/geoproducts/_src"],
}

# Bare names two packages may both export, and why. Everything else that
# collides across packages fails ``--check``.
ALLOWED_SHARED_NAMES: dict[str, str] = {
    "Fold": (
        "two folds at two levels: pipekit_ops.Fold reduces chunks associatively "
        "in parallel; geopatcher.temporal.aggregation.Fold is a stateful left "
        "fold across temporal patches"
    ),
    "download": (
        "one verb at two levels: geocloud.files.download fetches any URI; each "
        "sensor's `aws.download` takes that sensor's file records"
    ),
    "stack": (
        "one verb at two levels: geoarray.stack joins carriers already on one "
        "grid along a new axis; geoproducts.stack reads several product readers "
        "onto one grid first"
    ),
}


# Objects exported under more than one name (aliases), allowed for now.
ALLOWED_ALIASES: set[str] = set()


def _allowed_by_layout(owners: dict[str, object]) -> bool:
    """Collisions the package layout makes on purpose."""
    modules = {getattr(obj, "__module__", "") for obj in owners.values()}
    # A sensor's presets bind a geotoolz operator to its band names under the
    # operator's own name (geoproducts.goes.presets.NDVI).
    if any(m.endswith(".presets") for m in modules):
        return True
    # geoops.patch_ops re-exports the geopatcher pipekit bridge objects.
    return len({id(obj) for obj in owners.values()}) == 1


def _public_modules(package: str) -> list[types.ModuleType]:
    root = importlib.import_module(package)
    modules = [root]
    for info in pkgutil.walk_packages(root.__path__, f"{package}."):
        if any(part.startswith("_") for part in info.name.split(".")):
            continue
        modules.append(importlib.import_module(info.name))
    return modules


def _rank(module_name: str, obj: object) -> tuple[bool, bool, int]:
    """Order candidate homes: inside the defining package first (a re-export
    from another package never hides the original), then the shallowest home
    below the package root (``geoops.indices`` over ``geoops``;
    ``geoproducts.himawari`` over ``geoproducts.himawari.reader``).
    """
    origin = (getattr(obj, "__module__", None) or "").split(".")[0]
    depth = module_name.count(".")
    return module_name.split(".")[0] != origin, depth == 0, depth


def _stable_repr(obj: object) -> str:
    """``repr`` with set members sorted: string hashing is randomised per run."""
    if isinstance(obj, set | frozenset):
        items = ", ".join(sorted(repr(item) for item in obj))
        return f"{type(obj).__name__}({{{items}}})"
    return repr(obj)


def _summary(obj: object, *, home: tuple[str, ...] = tuple(PACKAGES)) -> str:
    if not callable(obj):
        value = _stable_repr(obj)
        return f"`{value if len(value) <= 60 else value[:57] + '...'}`".replace(
            "|", "\\|"
        )
    origin = (getattr(obj, "__module__", None) or "").split(".")[0]
    if origin and origin not in home:
        return f"Re-exported from `{origin}`."
    doc = inspect.getdoc(obj) or ""
    line = doc.strip().splitlines()[0] if doc.strip() else ""
    return line.replace("|", "\\|")


def _kind(obj: object) -> str:
    if inspect.isclass(obj):
        return "class"
    if callable(obj):
        return "function"
    return "constant"


def collect() -> tuple[dict[str, dict[str, list[tuple[str, str, str]]]], list[str]]:
    """``{package: {module: [(name, kind, summary)]}}`` and the collisions."""
    homes: dict[int, tuple[str, str, object]] = {}
    spellings: dict[int, set[str]] = defaultdict(set)
    for package in PACKAGES:
        for module in _public_modules(package):
            for name in getattr(module, "__all__", []):
                if name.startswith("__"):
                    continue
                obj = getattr(module, name)
                if isinstance(obj, types.ModuleType):
                    continue
                # An object re-exported elsewhere is still one object. Scalars
                # can't be told apart by id (equal small ints / short strings
                # are shared across unrelated constants), so a scalar is one
                # capability per (package, name, value): a facade re-exporting
                # its submodule's constant collapses to one home.
                scalar = isinstance(obj, int | float | str | bytes) or obj is None
                key = hash((package, name, _stable_repr(obj))) if scalar else id(obj)
                spellings[key].add(name)
                if key not in homes or _rank(module.__name__, obj) < _rank(
                    homes[key][0], homes[key][2]
                ):
                    homes[key] = (module.__name__, name, obj)

    index: dict[str, dict[str, list[tuple[str, str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    by_name: dict[str, dict[str, object]] = defaultdict(dict)
    for module_name, name, obj in homes.values():
        package = module_name.split(".")[0]
        index[package][module_name].append((name, _kind(obj), _summary(obj)))
        by_name[name][f"{module_name}.{name}"] = obj

    collisions = []
    for key, names in spellings.items():
        if len(names) > 1 and not names <= ALLOWED_ALIASES:
            module_name, _, _ = homes[key]
            collisions.append(
                f"{module_name}.{homes[key][1]} is also exported as "
                f"{', '.join(sorted(names - {homes[key][1]}))} (an alias)"
            )
    for name, owners in sorted(by_name.items()):
        packages = {path.split(".")[0] for path in owners}
        if len(packages) < 2 or name in ALLOWED_SHARED_NAMES:
            continue
        if not _allowed_by_layout(owners):
            collisions.append(f"{name}: {', '.join(sorted(owners))}")
    for package in UPSTREAM:
        upstream = importlib.import_module(package)
        for name in upstream.__all__:
            if name in ALLOWED_SHARED_NAMES:
                continue
            for path, obj in by_name.get(name, {}).items():
                if obj is not getattr(upstream, name):
                    collisions.append(f"{name}: {path} shadows {package}.{name}")
    return index, collisions


def _upstream_rows(package: str) -> list[tuple[str, str, str]]:
    module = importlib.import_module(package)
    rows = []
    for name in sorted(module.__all__):
        obj = getattr(module, name)
        rows.append((name, _kind(obj), _summary(obj, home=(package,))))
    return rows


def _toolkit_rows(directory: Path, src: Path) -> list[tuple[str, str, str]]:
    rows = []
    for path in sorted(directory.glob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        doc = (ast.get_docstring(tree) or "").strip().splitlines()
        names = [
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
            and not node.name.startswith("_")
        ]
        module = ".".join(path.relative_to(src).with_suffix("").parts)
        rows.append((module, doc[0] if doc else "", ", ".join(f"`{n}`" for n in names)))
    return rows


def render() -> tuple[str, list[str]]:
    index, collisions = collect()
    total = sum(len(rows) for modules in index.values() for rows in modules.values())
    out = [
        "# Capability index",
        "",
        "<!-- Generated by scripts/capabilities.py — do not edit by hand. -->",
        "",
        f"Every public name in the stack ({total} of them), each listed once at",
        "its public home, with the first line of its docstring. Search this page",
        "before writing a helper: if what you need is here, compose it; if it is",
        "almost here, extend it where it lives. Regenerate with",
        "`make capabilities` after changing a public API (CI checks it is",
        "current).",
        "",
    ]
    for package, dist in PACKAGES.items():
        out += [f"## `{package}` ({dist})", ""]
        for module_name in sorted(index[package]):
            out += [
                f"### `{module_name}`",
                "",
                "| Name | Kind | What it does |",
                "|---|---|---|",
            ]
            for name, kind, summary in sorted(index[package][module_name]):
                out.append(f"| `{name}` | {kind} | {summary} |")
            out.append("")
    for package in UPSTREAM:
        out += [
            f"## Upstream: `{package}`",
            "",
            "The composition framework every operator subclasses. Use these",
            "instead of writing your own control flow, caching, observation,",
            "QC, parallel map or serialisation; a carrier-agnostic need belongs",
            "upstream, not here.",
            "",
            "| Name | Kind | What it does |",
            "|---|---|---|",
        ]
        for name, kind, summary in _upstream_rows(package):
            out.append(f"| `{name}` | {kind} | {summary} |")
        out.append("")
    out += [
        "## Shared private toolkits",
        "",
        "Not public API — the plumbing a package's own families and readers",
        "share. Contributors use these instead of writing new helpers; a missing",
        "helper is added here, not inline in one caller.",
        "",
    ]
    for package, directories in TOOLKITS.items():
        src = ROOT / "packages" / PACKAGES[package] / "src"
        for directory in directories:
            dotted = ".".join(Path(directory).relative_to(src.relative_to(ROOT)).parts)
            out += [
                f"### `{dotted}`",
                "",
                "| Module | Purpose | Helpers |",
                "|---|---|---|",
            ]
            for module, purpose, names in _toolkit_rows(ROOT / directory, src):
                out.append(f"| `{module}` | {purpose.replace('|', '\\|')} | {names} |")
            out.append("")
    return "\n".join(out).rstrip() + "\n", collisions


def _public_operators() -> dict[str, list[tuple[str, type]]]:
    """Every public geoops operator class, by family, sorted by name."""
    from pipekit_ops import Operator

    import geoops

    rows: dict[str, list[tuple[str, type]]] = defaultdict(list)
    for name in sorted(getattr(geoops, "__all__", dir(geoops))):
        obj = getattr(geoops, name, None)
        if not (inspect.isclass(obj) and issubclass(obj, Operator)):
            continue
        if not obj.__module__.startswith("geoops."):
            continue
        rows[obj.__module__.split(".")[1]].append((name, obj))
    return rows


def render_backends() -> str:
    """The backends table: every public geoops operator and where it runs."""
    rows = {
        family: [(name, tuple(cls.backends or ())) for name, cls in ops]
        for family, ops in _public_operators().items()
    }
    n_jax = sum("jax" in b for ops in rows.values() for _, b in ops)
    n_all = sum(len(ops) for ops in rows.values())
    out = [
        "# Backends",
        "",
        "<!-- Generated by scripts/capabilities.py (make capabilities). -->",
        "",
        "Where each public operator runs: its `backends` class attribute.",
        "An operator refuses a carrier on a backend it does not list with",
        "`pipekit_ops.BackendError` before it runs, so a JAX carrier is never",
        "moved to NumPy silently: call `to_numpy()` first, or `to_jax()`",
        "after the NumPy-only steps. Operators that list `jax` match their",
        "NumPy result on a JAX carrier and trace once under `jax.jit` for",
        "same-shape inputs on different grids (the contract test checks",
        "both).",
        "",
        f"{n_jax} of {n_all} operators run on JAX today.",
        "",
        "| Family | Operator | Backends |",
        "|---|---|---|",
    ]
    for family in sorted(rows):
        for name, backends in rows[family]:
            out.append(f"| `{family}` | `{name}` | {', '.join(backends)} |")
    return "\n".join(out) + "\n"


def render_coupling() -> str:
    """The coupling table: every public geoops operator and the dims it mixes."""
    rows = _public_operators()
    counts: dict[str, int] = defaultdict(int)
    for ops in rows.values():
        for _, cls in ops:
            counts[cls.coupling.name] += 1
    out = [
        "# Coupling",
        "",
        "<!-- Generated by scripts/capabilities.py (make capabilities). -->",
        "",
        "Which dims each public operator mixes: its `coupling` class attribute",
        "(`geoops.carrier.Coupling`). An operator couples a dim when some output",
        "depends on two or more input values that differ along it; re-indexing",
        "alone (a selection, a crop, a flip) is an effect, not a coupling. See",
        "[Concepts](../concepts.md#coupling-which-dims-an-operator-mixes).",
        "",
        "- **Reach** is the spatial reach in pixels at the default settings:",
        "  a tile reproduces the whole-scene result in its interior when it",
        "  overlaps its neighbours by at least this much. *global* means the",
        "  operator reads the whole scene, so every tile gives a different",
        "  answer. Some operators compute it from their arguments",
        "  (`spatial_reach()`, read with `geoops.carrier.reach_of`).",
        "- **Fitted over** lists the further dims a fitted state is estimated",
        "  over (a scaler applies per pixel but fits over every pixel).",
        "- **Kind** is what comes out: a raster (*transform*), a list of",
        "  rasters (*fan-out*), a table, geometry or value (*conversion*);",
        "  readers are *sources* and writers *sinks*.",
        "",
        "The contract test (`tests/test_operator_coupling.py`) probes every",
        "operator it can build and fails when one mixes a dim it does not",
        "declare.",
        "",
        "| Class | Operators |",
        "|---|---|",
    ]
    for name, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        out.append(f"| {name} | {n} |")
    out += [
        "",
        "| Family | Operator | Class | Effects | Reach | Fitted over | Kind |",
        "|---|---|---|---|---|---|---|",
    ]
    for family in sorted(rows):
        for name, cls in rows[family]:
            c = cls.coupling
            effects = ", ".join(f"{d} {e}" for d, e in c.effects)
            if "space" not in c.couples:
                reach = ""
            else:
                reach = "global" if c.reach is None else str(c.reach)
            if getattr(cls, "spatial_reach", None) is not None and reach:
                reach += " (by argument)"
            fit = ", ".join(sorted(c.fit))
            kind = "" if c.kind == "transform" else c.kind
            if not c.reads_values:
                kind = (kind + ", " if kind else "") + "grid only"
            out.append(
                f"| `{family}` | `{name}` | {c.name} | {effects} | {reach} "
                f"| {fit} | {kind} |"
            )
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if stale")
    args = parser.parse_args()
    text, collisions = render()
    status = 0
    if collisions:
        print("Names exported by two packages for different objects:")
        print("\n".join(f"  {c}" for c in collisions))
        print("Rename one, or add it to ALLOWED_SHARED_NAMES with a reason.")
        status = 1
    outputs = (
        (INDEX, text),
        (BACKENDS, render_backends()),
        (COUPLING, render_coupling()),
    )
    for path, content in outputs:
        if args.check:
            current = path.read_text(encoding="utf-8") if path.exists() else ""
            if current != content:
                print(f"{path.relative_to(ROOT)} is stale; run scripts/capabilities.py")
                status = 1
        else:
            path.write_text(content, encoding="utf-8")
            print(f"wrote {path.relative_to(ROOT)}")
    return status


if __name__ == "__main__":
    sys.exit(main())
