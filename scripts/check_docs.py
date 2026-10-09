"""Check the docs' code examples: they parse, and every stack name resolves.

Scans the Python code in the docs site (``docs/**/*.md`` fences and
notebook code cells), the root README and every package README, and for
each example:

* **parses** it (top-level ``await`` allowed; doctest ``>>>`` / ``...``
  prompts stripped), so a typo'd example fails here;
* **resolves** every name it takes from the stack — ``import geotoolz as
  gz`` then ``gz.NDVI``, ``from geocloud.cog import write_cog``,
  ``gp.spatial.window.Hann`` — by importing the module and walking the
  attributes, so a rename breaks CI instead of rotting in an example;
* **requires imports** for the stack's conventional aliases (``gz``,
  ``gp``, ``gc``), so an example cannot lean on an earlier fence.

Only the five packages and pipekit are resolved; third-party names and
the example's own variables are left alone. Run it in the full
environment (``make install``)::

    uv run python scripts/check_docs.py          # exit 1 on any problem

A fence that is deliberately not Python-valid (a sketch with ``<...>``
placeholders) is marked with an HTML comment on the line before it:
``<!-- docs-check: skip -->``.
"""

from __future__ import annotations

import ast
import importlib
import json
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
STACK = ("geotoolz", "geopatcher", "geocatalog", "geocloud", "geoproducts", "pipekit")
SKIP_MARK = "<!-- docs-check: skip -->"
# The stack's conventional aliases: a fence that uses one must import it.
ALIASES = {"gz": "geotoolz", "gp": "geopatcher", "gc": "geocatalog"}
_FENCE = re.compile(r"^(?P<indent>[ \t]*)```(?P<lang>[\w+-]*)[^\n]*$")
_TOP_LEVEL_AWAIT = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT


@dataclass(frozen=True)
class Example:
    path: Path
    line: int
    code: str


def _sources() -> list[Path]:
    files = sorted((ROOT / "docs").rglob("*.md")) + sorted(
        (ROOT / "docs").rglob("*.ipynb")
    )
    files += [ROOT / "README.md", *sorted((ROOT / "packages").glob("*/README.md"))]
    return [f for f in files if "capabilities.md" not in f.name]


def _fences(path: Path) -> Iterator[Example]:
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        match = _FENCE.match(lines[i])
        if not match or match["lang"] not in ("python", "py", "pycon"):
            i += 1
            continue
        indent = len(match["indent"])
        skip = i > 0 and lines[i - 1].strip() == SKIP_MARK
        start = i + 1
        i += 1
        while i < len(lines) and not lines[i].strip().startswith("```"):
            i += 1
        body = [
            line[indent:] if line[:indent].isspace() else line
            for line in lines[start:i]
        ]
        i += 1
        if not skip:
            yield Example(path, start + 1, "\n".join(body))


def _cells(path: Path) -> Iterator[Example]:
    for n, cell in enumerate(json.loads(path.read_text(encoding="utf-8"))["cells"], 1):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            # IPython magics and shell escapes are not Python.
            code = "\n".join(
                "" if line.lstrip().startswith(("%", "!")) else line
                for line in source.splitlines()
            )
            yield Example(path, n, code)


def _strip_prompts(code: str) -> str:
    lines = code.splitlines()
    if not any(line.lstrip().startswith(">>>") for line in lines):
        return code
    kept = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith((">>> ", "... ")) or stripped in (">>>", "..."):
            kept.append(stripped[4:])
    return "\n".join(kept)


@cache
def _module(name: str) -> ModuleType | None:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _resolve(dotted: str) -> bool:
    """Import the longest module prefix of ``dotted``, ``getattr`` the rest."""
    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        obj: object | None = _module(".".join(parts[:cut]))
        if obj is None:
            continue
        for attr in parts[cut:]:
            if not hasattr(obj, attr):
                return False
            obj = getattr(obj, attr)
        return True
    return False


def _dotted(node: ast.AST) -> str | None:
    names = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return ".".join([node.id, *reversed(names)])
    return None


def _check(example: Example, imported: set[str] | None = None) -> list[str]:
    """Check one example; ``imported`` carries aliases a notebook's earlier
    cells bound, and gains the ones this example binds."""
    imported = set() if imported is None else imported
    code = _strip_prompts(example.code)
    try:
        tree = compile(
            code, str(example.path), "exec", ast.PyCF_ONLY_AST | _TOP_LEVEL_AWAIT
        )
    except SyntaxError as exc:
        return [f"does not parse: {exc.msg} (example line {exc.lineno})"]
    _mark_parents(tree)
    aliases: dict[str, str] = {}
    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in STACK:
                    if not _resolve(alias.name):
                        problems.append(f"import {alias.name}: no such module")
                    local = alias.asname or alias.name.split(".")[0]
                    aliases[local] = alias.name if alias.asname else local
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            if node.module.split(".")[0] not in STACK:
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                target = f"{node.module}.{alias.name}"
                if not _resolve(target):
                    problems.append(
                        f"from {node.module} import {alias.name}: not found"
                    )
                aliases[alias.asname or alias.name] = target
    bound = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.For, ast.With))
        for target in ast.walk(node)
        if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and not isinstance(
            getattr(node, "_parent", None), ast.Attribute
        ):
            dotted = _dotted(node)
            if dotted is None:
                continue
            head, _, rest = dotted.partition(".")
            if head in ALIASES and not {*aliases, *bound, *imported} >= {head}:
                problems.append(f"uses {head} without importing {ALIASES[head]}")
            elif head in aliases and rest:
                full = f"{aliases[head]}.{rest}"
                if not _resolve(full):
                    problems.append(f"{dotted}: not found (as {full})")
    imported.update(aliases)
    return problems


def _mark_parents(tree: ast.AST) -> None:
    """Link children to parents, so only the outermost node of an
    attribute chain (``gz.viz.RGBRecipe``, not ``gz.viz``) is resolved."""
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child._parent = parent  # type: ignore[attr-defined]


def main() -> int:
    failures = 0
    checked = 0
    for path in _sources():
        notebook = path.suffix == ".ipynb"
        examples = _cells(path) if notebook else _fences(path)
        imported: set[str] = set()  # a notebook's cells share one namespace
        for example in examples:
            checked += 1
            problems = _check(example, imported if notebook else None)
            for problem in dict.fromkeys(problems):
                failures += 1
                rel = path.relative_to(ROOT)
                print(f"{rel}:{example.line}: {problem}")
    print(f"checked {checked} examples: {failures} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
