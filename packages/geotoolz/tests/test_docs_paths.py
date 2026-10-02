"""Every ``geotoolz.<...>`` / ``gz.<...>`` dotted path in the docs resolves.

Scans the docs site (``docs/**/*.md`` and the notebooks next to it), the
root and package READMEs, and the geotoolz sources (docstrings, comments
and messages) for dotted paths rooted at ``geotoolz`` or its ``gz`` alias,
and resolves each one: import the longest importable module prefix, then
``getattr`` the rest. A renamed or removed name then breaks this test
instead of silently rotting in prose.
"""

from __future__ import annotations

import functools
import importlib
import re
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "packages" / "geotoolz"

# ``geotoolz.x.y`` / ``gz.x.y`` not preceded by a word char, dot, slash or
# hyphen — which skips URLs and paths (``github.com/jejjohnson/geotoolz.git``,
# ``packages/geotoolz/...``) and qualified names of other packages.
_DOTTED = re.compile(r"(?<![\w./\-])(geotoolz|gz)((?:\.[A-Za-z_]\w*)+)")

# Paths that are *meant* not to resolve, keyed by (file relative to the repo
# root, path as written). Keep this small and say why.
ALLOWED_UNRESOLVED: dict[tuple[str, str], str] = {
    # Explains that augment.Compose is deliberately *not* top-level.
    ("docs/api/augment.md", "gz.Compose"): "negative example",
}


def _scanned_files() -> list[Path]:
    docs = REPO_ROOT / "docs"
    files = sorted(docs.rglob("*.md")) + sorted(docs.rglob("*.ipynb"))
    files += [REPO_ROOT / "README.md", PACKAGE_ROOT / "README.md"]
    files += sorted((PACKAGE_ROOT / "src" / "geotoolz").rglob("*.py"))
    return files


@functools.cache
def _dotted_paths() -> list[tuple[str, int, str]]:
    found = []
    for path in _scanned_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in _DOTTED.finditer(line):
                found.append((rel, lineno, match.group(0)))
    return found


class _OptionalDependencyMissing(Exception):
    """A geotoolz module exists but needs a third-party package that is absent."""


def _import(name: str) -> ModuleType | None:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name is not None and not exc.name.startswith("geotoolz"):
            raise _OptionalDependencyMissing(exc.name) from exc
        return None


@functools.cache
def _resolves(dotted: str) -> bool:
    parts = ["geotoolz", *dotted.split(".")[1:]]
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
    assert "geotoolz._src.wrap.wrap_like" in paths
    assert len(paths) > 50


def test_resolver_rejects_missing_names() -> None:
    assert _resolves("gz.NDVI")
    assert _resolves("geotoolz.radiometry._src.array.dn_to_radiance")
    assert _resolves("geotoolz.Sequential.get_config")
    assert not _resolves("geotoolz.geom.RasterToPoints")
    assert not _resolves("geotoolz.presets")
    assert not _resolves("gz.mask.MaskFromGeometry")


def test_all_dotted_paths_resolve() -> None:
    unresolved = []
    for rel, lineno, dotted in _dotted_paths():
        if (rel, dotted) in ALLOWED_UNRESOLVED:
            continue
        try:
            ok = _resolves(dotted)
        except _OptionalDependencyMissing:
            # e.g. ``geotoolz.patch_ops`` without the ``[patch]`` extra: the
            # module exists, its dependency is just not installed here.
            continue
        if not ok:
            unresolved.append(f"{rel}:{lineno}: {dotted}")
    assert not unresolved, "unresolvable dotted paths:\n" + "\n".join(unresolved)


def test_allowlist_entries_are_still_needed() -> None:
    """A stale allowlist entry (fixed or deleted text) must be removed."""
    present = {(rel, dotted) for rel, _, dotted in _dotted_paths()}
    for key in ALLOWED_UNRESOLVED:
        assert key in present, f"allowlist entry no longer occurs: {key}"
        assert not _resolves(key[1]), f"allowlist entry now resolves: {key}"
