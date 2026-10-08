"""The catalog docs name things that exist, under names that still hold.

Scans the catalog docs (``docs/catalog/**/*.md`` and its notebooks), the
package README and the geocatalog sources (docstrings, comments and
messages) and checks two things:

* every ``geocatalog.<...>`` dotted path resolves: import the longest
  importable module prefix, then ``getattr`` the rest, so a renamed or
  removed name breaks this test instead of silently rotting in prose;
* no text uses a name or location the monorepo retired — the pre-monorepo
  ``docs/design/`` tree and standalone docs site, ``geotoolz.patch``
  (now ``geopatcher``), install commands naming the ``geocatalog`` /
  ``geopatcher`` *distributions* (they are ``geotoolz-catalog`` /
  ``geotoolz-patcher``), and private ``_src`` paths in the user docs.
"""

from __future__ import annotations

import functools
import importlib
import re
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "packages" / "geotoolz-catalog"
DOCS = REPO_ROOT / "docs" / "catalog"

# ``geocatalog.x.y`` not preceded by a word char, dot, slash or hyphen —
# which skips URLs, file paths and qualified names of other packages — and
# not a glob (``geocatalog.load_*``).
_DOTTED = re.compile(r"(?<![\w./\-])geocatalog((?:\.[A-Za-z_]\w*)+)(?![\w*])")

# ``from geocatalog[.ns] import a, b as c`` on one line (code and prose).
_FROM_IMPORT = re.compile(r"\bfrom (geocatalog(?:\.\w+)*) import ([\w, ]+)")

# Paths that are *meant* not to resolve, keyed by (file relative to the
# repo root, path as written). Keep this small and say why.
ALLOWED_UNRESOLVED: dict[tuple[str, str], str] = {}

# Retired names and locations: (pattern, why, scope). Scope "all" covers
# docs, README and sources; "docs" only the user-facing docs and README
# (sources legitimately import from `geocatalog._src`).
RETIRED: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(r"(?<![\w/])docs/design/"),
        "pre-monorepo path; the design docs live in docs/catalog/design/",
        "all",
    ),
    (
        re.compile(r"jejjohnson\.github\.io/geocatalog"),
        "standalone docs site; use jejjohnson.github.io/geotoolz/catalog/",
        "all",
    ),
    (re.compile(r"\bgeotoolz\.patch\b"), "the patcher is `geopatcher`", "all"),
    (
        re.compile(r"(?:pip install|uv add)\s+['\"]?geocatalog\b"),
        "the distribution is geotoolz-catalog",
        "all",
    ),
    (
        re.compile(r"(?:pip install|uv add)\s+['\"]?geopatcher\b"),
        "the distribution is geotoolz-patcher",
        "all",
    ),
    (
        re.compile(r"(?<![\w-])geocatalog\["),
        "extras belong to the geotoolz-catalog distribution",
        "all",
    ),
    (re.compile(r"\b[Pp]hase [0-9]\b"), "pre-release planning phases", "all"),
    (re.compile(r"\bgeocatalog\._src\b"), "private path in user docs", "docs"),
]


def _docs_files() -> list[Path]:
    files = sorted(DOCS.rglob("*.md")) + sorted(DOCS.rglob("*.ipynb"))
    return [*files, PACKAGE_ROOT / "README.md"]


def _source_files() -> list[Path]:
    return sorted((PACKAGE_ROOT / "src" / "geocatalog").rglob("*.py"))


@functools.cache
def _lines() -> list[tuple[str, str, int, str]]:
    """``(scope, file, lineno, line)`` for every scanned line."""
    found = []
    for scope, paths in (("docs", _docs_files()), ("src", _source_files())):
        for path in paths:
            rel = path.relative_to(REPO_ROOT).as_posix()
            text = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), 1):
                found.append((scope, rel, lineno, line))
    return found


@functools.cache
def _dotted_paths() -> list[tuple[str, int, str]]:
    return [
        (rel, lineno, match.group(0))
        for _, rel, lineno, line in _lines()
        for match in _DOTTED.finditer(line)
    ]


class _OptionalDependencyMissing(Exception):
    """A geocatalog module exists but needs a third-party package that is absent."""


def _import(name: str) -> ModuleType | None:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name is not None and not exc.name.startswith("geocatalog"):
            raise _OptionalDependencyMissing(exc.name) from exc
        return None


@functools.cache
def _resolves(dotted: str) -> bool:
    parts = dotted.split(".")
    for split in range(len(parts), 0, -1):
        obj = _import(".".join(parts[:split]))
        if obj is None:
            continue
        for attr in parts[split:]:
            if not hasattr(obj, attr):
                return False
            obj = getattr(obj, attr)
        return True
    return False


def test_scanner_finds_paths() -> None:
    """Guard the scanner itself: an empty scan would pass vacuously."""
    paths = {dotted for _, _, dotted in _dotted_paths()}
    assert "geocatalog.patch.field_for" in paths
    assert len(paths) > 30


def test_resolver_rejects_missing_names() -> None:
    assert _resolves("geocatalog.storage.CatalogBundle.ingest")
    assert _resolves("geocatalog._src.utils.retry.retry_transient_io")
    assert not _resolves("geocatalog.GeoCatalog.ingest")
    assert not _resolves("geocatalog._src.objstore")


def test_all_dotted_paths_resolve() -> None:
    unresolved = []
    for rel, lineno, dotted in _dotted_paths():
        if (rel, dotted) in ALLOWED_UNRESOLVED:
            continue
        try:
            ok = _resolves(dotted)
        except _OptionalDependencyMissing:
            continue  # the module exists; its extra is not installed here
        if not ok:
            unresolved.append(f"{rel}:{lineno}: {dotted}")
    assert not unresolved, "unresolvable dotted paths:\n" + "\n".join(unresolved)


def test_from_imports_resolve() -> None:
    """``from geocatalog.x import y`` names a public home that holds ``y``."""
    unresolved = []
    for _, rel, lineno, line in _lines():
        for match in _FROM_IMPORT.finditer(line):
            for item in match.group(2).split(","):
                name = item.split(" as ")[0].strip()
                if not name:
                    continue
                dotted = f"{match.group(1)}.{name}"
                try:
                    ok = _resolves(dotted)
                except _OptionalDependencyMissing:
                    continue
                if not ok:
                    unresolved.append(f"{rel}:{lineno}: {dotted}")
    assert not unresolved, "unresolvable imports:\n" + "\n".join(unresolved)


def test_allowlist_entries_are_still_needed() -> None:
    present = {(rel, dotted) for rel, _, dotted in _dotted_paths()}
    for key in ALLOWED_UNRESOLVED:
        assert key in present, f"allowlist entry no longer occurs: {key}"
        assert not _resolves(key[1]), f"allowlist entry now resolves: {key}"


@pytest.mark.parametrize(
    ("pattern", "why", "scope"), RETIRED, ids=[why for _, why, _ in RETIRED]
)
def test_no_retired_names(pattern: re.Pattern[str], why: str, scope: str) -> None:
    hits = [
        f"{rel}:{lineno}: {line.strip()}"
        for line_scope, rel, lineno, line in _lines()
        if (scope == "all" or line_scope == scope) and pattern.search(line)
    ]
    assert not hits, f"{why}:\n" + "\n".join(hits)
