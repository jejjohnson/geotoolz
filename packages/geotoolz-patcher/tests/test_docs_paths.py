"""The patcher docs name things that exist, under names that still hold.

Scans the patcher docs (``docs/patcher/**/*.md`` and its notebooks), the
package README and the geopatcher sources (docstrings, comments and
messages) and checks two things:

* every ``geopatcher.<...>`` dotted path resolves: import the longest
  importable module prefix, then ``getattr`` the rest, so a renamed or
  removed name breaks this test instead of silently rotting in prose;
* no text uses a name or location the monorepo retired — the
  pre-monorepo ``docs/<page>.md`` / ``docs/design/`` paths, the standalone
  ``geopatcher`` repository and docs site, ``geotoolz.patch`` (now
  ``geopatcher``), install commands naming the ``geopatcher``
  *distribution* (it is ``geotoolz-patcher``), bare ``#NN`` references to
  the standalone repo's issues (qualify them as ``jejjohnson/geopatcher#NN``),
  and private ``_src`` paths in the user docs.
"""

from __future__ import annotations

import functools
import importlib
import re
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "packages" / "geotoolz-patcher"
DOCS = REPO_ROOT / "docs" / "patcher"

# ``geopatcher.x.y`` not preceded by a word char, dot, slash or hyphen —
# which skips URLs, file paths and qualified names of other packages — and
# not a glob (``geopatcher.Spatial*``).
_DOTTED = re.compile(r"(?<![\w./\-])geopatcher((?:\.[A-Za-z_]\w*)+)(?![\w*])")

# Paths that are *meant* not to resolve, keyed by (file relative to the
# repo root, path as written). Keep this small and say why.
_OTEL = "an OpenTelemetry span / attribute name in the hook example"
ALLOWED_UNRESOLVED: dict[tuple[str, str], str] = {
    ("docs/patcher/observability.md", f"geopatcher.{name}"): _OTEL
    for name in ("patch", "anchor", "runtime_s", "bytes")
}

# Retired names and locations: (pattern, why, scope). Scope "all" covers
# docs, README and sources; "docs" only the user-facing docs and README
# (sources legitimately import from `geopatcher._src`).
RETIRED: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(
            r"(?<![\w/])docs/(?:patching|concepts|decisions|observability"
            r"|quickstart|design|notebooks|recipes)\b"
        ),
        "pre-monorepo path; the patcher docs live in docs/patcher/",
        "all",
    ),
    (
        re.compile(r"packages/geotoolz-patcher/docs\b"),
        "the patcher has no package-local docs; they live in docs/patcher/",
        "all",
    ),
    (
        re.compile(r"github\.com/jejjohnson/geopatcher\b(?!/(?:issues|pull)/)"),
        "standalone repository; the code lives in jejjohnson/geotoolz",
        "all",
    ),
    (
        re.compile(r"jejjohnson\.github\.io/geopatcher"),
        "standalone docs site; use jejjohnson.github.io/geotoolz/patcher/",
        "all",
    ),
    (re.compile(r"\bgeotoolz\.patch\b"), "the patcher is `geopatcher`", "all"),
    (
        re.compile(r"(?:pip install|uv add)\s+['\"]?geopatcher\b"),
        "the distribution is geotoolz-patcher",
        "all",
    ),
    (
        re.compile(r"(?<![\w-])geopatcher\["),
        "extras belong to the geotoolz-patcher distribution",
        "all",
    ),
    (
        # `#NN` not part of a word, path, URL, Markdown anchor `](#...)` or
        # an already-qualified `owner/repo#NN`; geotoolz's own issues that
        # touch the patcher are 3-digit.
        re.compile(r"(?<![\w/#])(?<!\]\()#[1-9][0-9]?(?![\w-])"),
        "pre-monorepo issue number; qualify as jejjohnson/geopatcher#NN",
        "all",
    ),
    (re.compile(r"\bgeopatcher\._src\b"), "private path in user docs", "docs"),
]


def _docs_files() -> list[Path]:
    files = sorted(DOCS.rglob("*.md")) + sorted(DOCS.rglob("*.ipynb"))
    return [*files, PACKAGE_ROOT / "README.md"]


def _source_files() -> list[Path]:
    return sorted((PACKAGE_ROOT / "src" / "geopatcher").rglob("*.py"))


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
    """A geopatcher module exists but needs a third-party package that is absent."""


def _import(name: str) -> ModuleType | None:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name is not None and not exc.name.startswith("geopatcher"):
            raise _OptionalDependencyMissing(exc.name) from exc
        return None
    except ImportError as exc:
        # `geopatcher.integrations.pipekit` raises a friendly ImportError
        # when its extra is absent.
        raise _OptionalDependencyMissing(name) from exc


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
    assert "geopatcher.RasterField" in paths
    assert len(paths) > 30


def test_resolver_rejects_missing_names() -> None:
    assert _resolves("geopatcher.SpatialPatcher.split")
    assert _resolves("geopatcher._src.walk")
    assert not _resolves("geopatcher.SpatialPatcher.nope")
    assert not _resolves("geopatcher._src.nope")


def test_retired_patterns_catch_known_shapes() -> None:
    def hits(text: str) -> list[str]:
        return [why for pattern, why, _ in RETIRED if pattern.search(text)]

    assert hits("see docs/patching.md") and hits("issue #18") and hits("(#9)")
    assert hits("pip install 'geopatcher[grid]'")
    assert not hits("see docs/patcher/patching.md")
    assert not hits("jejjohnson/geopatcher#18 and #185 and [x](#3-wrap) [y](#9)")
    assert not hits("https://github.com/jejjohnson/geopatcher/issues/18")


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
