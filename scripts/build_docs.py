#!/usr/bin/env python3
"""Build and assemble the two-tool documentation site.

The docs are produced by two generators (see ``docs/README.md``):

- **mystmd** builds the prose half — home page, each package's concepts,
  how-tos, tutorials and design pages, notebooks — into ``docs/_build/html``.
- **Zensical + mkdocstrings** builds the API reference into ``site/`` from
  ``mkdocs.yml``, and publishes a Sphinx-compatible ``objects.inv`` alongside it.
  The API pages live beside each package's prose (``docs/<pkg>/api/``), so
  Zensical reads ``docs/`` and excludes everything else.

This script runs both, assembles them into ``public/`` (prose at the root, API
reference under ``/reference/``), repairs the cross-references between them
(including the API pages' links into prose pages, which it points at the pages
mystmd built),
and then verifies that every internal link in the assembled site resolves —
including the ones that cross from one generator's output into the other's,
which neither tool can check on its own.

Usage:
    python scripts/build_docs.py            # full build into public/
    python scripts/build_docs.py --check    # validate sources, no HTML render

``--check`` runs both generators in their strict validation modes without
rendering the MyST theme, which needs network access to fetch the template.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import zlib
from functools import partial
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cyclopts
from loguru import logger


REPO = Path(__file__).resolve().parent.parent
MKDOCS_OUT = REPO / "site"
MYST_DIR = REPO / "docs"
MYST_OUT = MYST_DIR / "_build" / "html"
PUBLIC = REPO / "public"
API_SUBDIR = "reference"
# Plain files served at the site root, next to the prose pages. mystmd publishes
# only pages it renders, so these are copied in by `assemble`.
ROOT_FILES: tuple[str, ...] = ("llms.txt",)
# Where the API reference starts; `/reference/` itself redirects here.
API_HOME = "api/"

# Must match `references.api` in docs/myst.yml. mystmd only loads intersphinx
# inventories over http(s), so the freshly built Zensical output is served here
# during the MyST build and the resulting URLs are rewritten afterwards.
API_PORT = 8910
# No trailing slash: mystmd echoes the configured inventory URL into the page
# config it embeds for hydration, and it normalises away the trailing slash.
# Matching the bare origin rewrites that echo as well as the links, so the leak
# check below can be absolute about it.
API_ORIGIN = f"http://127.0.0.1:{API_PORT}"


# ---------------------------------------------------------------------------
# Sphinx inventory
# ---------------------------------------------------------------------------


def parse_inventory(data: bytes) -> dict[str, str]:
    """Parse a Sphinx ``objects.inv`` v2 payload.

    Args:
        data: Raw bytes of an ``objects.inv`` file.

    Returns:
        A mapping of object name to the URI recorded for it, with the ``$``
        abbreviation (meaning "the anchor equals the object name") expanded.

    Raises:
        ValueError: If the payload is not a version 2 inventory.
    """
    header, _, rest = data.partition(b"\n")
    if b"version 2" not in header:
        msg = f"unsupported inventory header: {header!r}"
        raise ValueError(msg)
    # Three more comment lines follow the version banner.
    for _ in range(3):
        _, _, rest = rest.partition(b"\n")

    entries: dict[str, str] = {}
    body = zlib.decompress(rest).decode("utf-8")
    for line in body.splitlines():
        match = re.match(r"(?x)(.+?)\s+(\S+)\s+(-?\d+)\s+(\S*)\s+(.*)", line)
        if match is None:
            continue
        name, _role, _prio, uri, _display = match.groups()
        entries[name] = uri.replace("$", name) if uri.endswith("$") else uri
    return entries


def anchor_case_map(entries: dict[str, str]) -> dict[str, str]:
    """Map lowercased anchors back to their true, case-sensitive spelling.

    mystmd 1.11.0 lowercases the fragment of every intersphinx-resolved
    cross-reference, which breaks links into mkdocstrings' case-sensitive
    anchors. This mapping is what lets [`restore_anchor_case`][] undo that.

    Anchors that are ambiguous after lowercasing are omitted rather than
    guessed at.

    Args:
        entries: Inventory mapping from [`parse_inventory`][].

    Returns:
        A mapping of lowercased anchor to correctly cased anchor, containing
        only anchors that differ from their lowercased form and are unique.
    """
    seen: dict[str, set[str]] = {}
    for uri in entries.values():
        _, _, anchor = uri.partition("#")
        if anchor:
            seen.setdefault(anchor.lower(), set()).add(anchor)
    return {
        lowered: next(iter(variants))
        for lowered, variants in seen.items()
        if len(variants) == 1 and next(iter(variants)) != lowered
    }


# ---------------------------------------------------------------------------
# HTML rewriting
# ---------------------------------------------------------------------------

_HREF = re.compile(r'href="([^"]*)"')
_NAV = re.compile(r'"nav":\s*(\[.*?\])', re.DOTALL)


def nav_base_url_problems(html: str) -> list[str]:
    """Find site-nav URLs that the theme will prefix with ``BASE_URL`` twice.

    The MyST theme re-renders the site nav from the config it embeds for
    hydration, prepending ``BASE_URL`` to any URL that starts with ``/``. A nav
    entry that already carries the deployment prefix therefore ends up doubled
    (``/project/project/page/``) and 404s — but only after hydration, so the
    static HTML looks correct and [`verify_links`][] cannot see it.

    Nav URLs must be absolute, which the theme treats as external and leaves
    alone.

    Args:
        html: A rendered page's source.

    Returns:
        A list of offending nav URLs; empty when every entry is safe.
    """
    problems: list[str] = []
    for block in _NAV.findall(html):
        for url in re.findall(r'"url"\s*:\s*"([^"]*)"', block):
            if url.startswith("/"):
                problems.append(url)
    return sorted(set(problems))


def restore_anchor_case(html: str, mapping: dict[str, str]) -> tuple[str, int]:
    """Repair lowercased API anchors in one HTML document.

    Args:
        html: The document source.
        mapping: Lowercased-to-true anchor map from [`anchor_case_map`][].

    Returns:
        The repaired source and the number of anchors changed.
    """
    count = 0

    def fix(match: re.Match[str]) -> str:
        nonlocal count
        href = match.group(1)
        base, sep, anchor = href.partition("#")
        if not sep or anchor not in mapping:
            return match.group(0)
        count += 1
        return f'href="{base}#{mapping[anchor]}"'

    return _HREF.sub(fix, html), count


def rewrite_api_origin(html: str, origin: str, replacement: str) -> tuple[str, int]:
    """Replace the temporary localhost API origin with the deployed path.

    Args:
        html: The document source.
        origin: The origin baked in during the build, e.g.
            ``http://127.0.0.1:8910/``.
        replacement: What to put in its place, e.g. ``/reference/``.

    Returns:
        The rewritten source and the number of replacements made.
    """
    return html.replace(origin, replacement), html.count(origin)


# ---------------------------------------------------------------------------
# Link verification
# ---------------------------------------------------------------------------


class _LinkCollector(HTMLParser):
    """Collect ``href`` targets and element ids from a document."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if (ident := values.get("id")) is not None:
            self.ids.add(ident)
        if tag == "a" and (name := values.get("name")) is not None:
            self.ids.add(name)
        if tag == "a" and (href := values.get("href")) is not None:
            self.hrefs.append(href)


def _collect(path: Path) -> _LinkCollector:
    parser = _LinkCollector()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    return parser


def _resolve(root: Path, page: Path, href: str) -> Path | None:
    """Resolve an internal href to the file on disk that should serve it."""
    target = href.split("?", 1)[0]
    if target.startswith("/"):
        candidate = root / target.lstrip("/")
    else:
        candidate = (page.parent / target).resolve()
    if candidate.is_dir():
        candidate = candidate / "index.html"
    elif not candidate.suffix:
        indexed = Path(f"{candidate}/index.html")
        candidate = indexed if indexed.exists() else candidate
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def verify_links(root: Path, base_url: str = "") -> list[str]:
    """Check that every internal link in the assembled site resolves.

    Both the target document and, when the link carries a fragment, the anchor
    inside it must exist. This is the only check in the pipeline that sees
    both generators' output at once.

    Args:
        root: The assembled site root, normally ``public/``.
        base_url: Deployment path prefix to strip from absolute links, so a
            site built for ``/project/`` validates against the local tree.

    Returns:
        A list of human-readable problems; empty when the site is sound.
    """
    root = root.resolve()
    problems: list[str] = []
    ids_cache: dict[Path, set[str]] = {}

    # A 404 page is served in place of any missing path, so its root-absolute
    # links are correct at runtime even though they do not resolve from where
    # the file happens to sit. Index its anchors, but do not check its links.
    pages = sorted(root.rglob("*.html"))
    for page in pages:
        ids_cache[page] = _collect(page).ids

    for page in pages:
        if page.name == "404.html":
            continue
        rel = page.relative_to(root)
        for href in _collect(page).hrefs:
            if not href or href.startswith(("http://", "https://", "mailto:", "#")):
                continue
            if base_url and href.startswith(f"{base_url}/"):
                href = href[len(base_url) :]
            path_part, _, anchor = href.partition("#")
            if not path_part:
                continue
            target = _resolve(root, page, path_part)
            if target is None or not target.exists():
                problems.append(f"{rel}: dead link -> {href}")
                continue
            if anchor:
                ids = ids_cache.get(target)
                if ids is None:
                    ids = _collect(target).ids
                    ids_cache[target] = ids
                if anchor not in ids:
                    problems.append(f"{rel}: missing anchor -> {href}")
    return problems


# ---------------------------------------------------------------------------
# Source links to files mystmd does not build
# ---------------------------------------------------------------------------

#: prose folders whose files mystmd never renders as pages
UNBUILT_DIRS = ("api", "templates", "hooks")
_SOURCE_LINK = re.compile(r"\]\(([^)\s]+)")
_EXCLUDE_BLOCK = re.compile(r"^  exclude:\n((?:    - .+\n)+)", re.MULTILINE)


def excluded(root: Path) -> list[str]:
    """The ``project.exclude`` patterns of ``myst.yml``: the API pages and tooling."""
    match = _EXCLUDE_BLOCK.search((root / "myst.yml").read_text(encoding="utf-8"))
    if match is None:
        return []
    return [line.strip()[2:] for line in match.group(1).splitlines()]


def is_excluded(rel: str, patterns: list[str]) -> bool:
    """Whether mystmd skips ``rel`` (a path relative to ``docs/``)."""
    return any(
        fnmatch.fnmatch(rel, pattern)
        or (pattern.endswith("/**") and rel.startswith(pattern[:-2]))
        for pattern in patterns
    )


# fenced blocks and inline code, whose links are examples rather than links
_CODE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)
_TOC_FILE = re.compile(r"^\s*-?\s*file:\s*(\S+)\s*$", re.MULTILINE)


def toc_files(root: Path) -> set[str]:
    """The pages the MyST table of contents builds, relative to ``root``."""
    return set(_TOC_FILE.findall((root / "myst.yml").read_text(encoding="utf-8")))


def _prose_texts(root: Path) -> list[tuple[str, str]]:
    """The Markdown of every prose page, keyed by its path relative to ``root``.

    Walks symlinked folders too (design pages linked in from packages), and
    reads ``.md`` files and notebook markdown cells.
    """
    texts = []
    patterns = excluded(root)
    for folder, dirs, files in os.walk(root, followlinks=True):
        rel_dir = Path(folder).relative_to(root)
        if rel_dir.parts and rel_dir.parts[0] in ("_build", *UNBUILT_DIRS):
            dirs.clear()
            continue
        for name in sorted(files):
            page = Path(folder) / name
            rel = (rel_dir / name).as_posix()
            if page.suffix not in (".md", ".ipynb") or is_excluded(rel, patterns):
                continue
            if not page.is_file():
                continue
            if page.suffix == ".md":
                texts.append((rel, page.read_text(encoding="utf-8")))
            else:
                cells = json.loads(page.read_text(encoding="utf-8"))["cells"]
                md = [
                    "".join(c["source"]) for c in cells if c["cell_type"] == "markdown"
                ]
                texts.append((rel, "\n".join(md)))
    return texts


def source_link_problems(root: Path) -> list[str]:
    """Relative prose links to a file or folder mystmd does not build as a page.

    mystmd publishes a linked file it does not render, such as an API page
    under ``docs/api/`` or a Markdown page left out of the ``toc``, as a raw
    download. The link then resolves, so ``verify_links`` passes it, but the
    reader gets a Markdown file. A link to a folder does not resolve at all.
    Link to an API object with ``xref:api#<name>``, to an API page by its
    deployed ``/reference/`` URL, and to a prose page only if it is in the
    ``toc``.

    Args:
        root: The MyST project directory (``docs/``).

    Returns:
        One message per offending link.
    """
    built = toc_files(root)
    patterns = excluded(root)
    problems = []
    for rel, text in _prose_texts(root):
        for target in _SOURCE_LINK.findall(_CODE.sub("", text)):
            path = target.split("#", 1)[0]
            if not path or ":" in path.split("/", 1)[0]:
                continue
            linked = os.path.normpath((Path(rel).parent / path).as_posix())
            first = linked.split("/", 1)[0]
            if first in UNBUILT_DIRS:
                why = f"is in docs/{first}/, which mystmd publishes as a raw download"
            elif is_excluded(linked, patterns) and Path(linked).suffix == ".md":
                why = "is an API page, which mystmd publishes as a raw download"
            elif path.endswith("/") or (root / linked).is_dir():
                why = "is a folder, not a page"
            elif Path(linked).suffix in (".md", ".ipynb") and linked not in built:
                why = "is not in the toc, so mystmd publishes it as a raw download"
            else:
                continue
            problems.append(f"{rel}: {target!r} {why}")
    return problems


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _run(command: list[str], cwd: Path) -> None:
    logger.info("$ {}", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def _serve(directory: Path) -> ThreadingHTTPServer:
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    server = ThreadingHTTPServer(("127.0.0.1", API_PORT), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def build_api() -> None:
    """Build the Zensical API reference into ``site/``.

    ``docs/hooks`` goes on ``PYTHONPATH`` so Zensical can import the
    ``xarray_options`` Markdown extension that ``mkdocs.yml`` names.
    """
    command = ["uv", "run", "--group", "docs", "zensical", "build", "--strict"]
    logger.info("$ {}", " ".join(command))
    hooks = str(MYST_DIR / "hooks")
    path = os.pathsep.join(filter(None, [hooks, os.environ.get("PYTHONPATH")]))
    env = {**os.environ, "PYTHONPATH": path}
    subprocess.run(command, cwd=REPO, check=True, env=env)


def build_prose(*, html: bool) -> None:
    """Build the MyST prose site.

    Args:
        html: Render the themed HTML site. When false, only the site content
            and cross-references are built and validated, which needs no
            network access for the theme template.
    """
    target = "--html" if html else "--site"
    # mystmd caches fetched inventories. Purge them so the prose is always
    # validated against the API reference just built, not a previous run's.
    for stale in (MYST_DIR / "_build" / "cache").glob("xrefs-intersphinx-*"):
        stale.unlink()
    server = _serve(MKDOCS_OUT)
    try:
        _run(["myst", "build", target, "--strict"], MYST_DIR)
    finally:
        server.shutdown()


def prose_slugs(myst_out: Path) -> dict[str, str]:
    """Map each prose source path (relative to ``docs/``) to its MyST URL path.

    mystmd writes a ``<slug>.json`` per page that records the source file it
    came from; the home page's slug ``index`` is served at the site root.
    """
    slugs: dict[str, str] = {}
    for page in myst_out.glob("*.json"):
        try:
            data = json.loads(page.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(data, dict) or not {"location", "slug"} <= data.keys():
            continue
        slug = data["slug"]
        slugs[data["location"].lstrip("/")] = "" if slug == "index" else f"{slug}/"
    return slugs


def rewrite_prose_links(
    api_root: Path, slugs: dict[str, str], base_url: str
) -> tuple[int, list[str]]:
    """Point the API pages' links into prose pages at the pages mystmd built.

    The API pages link to the prose beside them by file
    (``../operators-concepts.md``). Zensical excludes those files, so it
    renders each link as the directory URL the page would have had
    (``../operators-concepts/``), under ``/reference/``. This
    resolves every such URL back to its source file and replaces it with the
    URL mystmd gave that file, so a link always lands on the page its author
    meant, whatever slug suffix mystmd chose.

    Args:
        api_root: The API reference inside the assembled site.
        slugs: Source path to MyST URL path, from [`prose_slugs`][].
        base_url: Deployment path prefix, e.g. ``/geotoolz``.

    Returns:
        The number of links rewritten, and one message per link that points
        at neither an API page nor a prose page.
    """
    root = api_root.resolve()
    count = 0
    problems: list[str] = []
    for page in root.rglob("*.html"):
        html = page.read_text(encoding="utf-8")

        def fix(match: re.Match[str], page: Path = page) -> str:
            nonlocal count
            href = match.group(1)
            external = ("http://", "https://", "mailto:", "#", "/")
            if not href or href.startswith(external):
                return match.group(0)
            path, sep, anchor = href.partition("#")
            if not path:
                return match.group(0)
            target = (page.parent / path).resolve()
            if not target.is_relative_to(root):
                return match.group(0)
            if target.is_file() or (target / "index.html").is_file():
                return match.group(0)
            rel = target.relative_to(root).as_posix()
            for source in (f"{rel}.md", f"{rel}.ipynb", f"{rel}/index.md"):
                if source in slugs:
                    count += 1
                    return f'href="{base_url}/{slugs[source]}{sep}{anchor}"'
            problems.append(f"{page.relative_to(root)}: no page for -> {href}")
            return match.group(0)

        new = _HREF.sub(fix, html)
        if new != html:
            page.write_text(new, encoding="utf-8")
    return count, problems


def assemble(base_url: str) -> list[str]:
    """Combine both builds into ``public/`` and repair cross-references.

    Args:
        base_url: Deployment path prefix, e.g. ``/geotoolz``.

    Returns:
        The list of link problems found in the assembled site.

    Raises:
        SystemExit: If either generator's output is missing.
    """
    if not MYST_OUT.is_dir():
        sys.exit(f"missing MyST output at {MYST_OUT}")
    if not MKDOCS_OUT.is_dir():
        sys.exit(f"missing Zensical output at {MKDOCS_OUT}")

    if PUBLIC.exists():
        shutil.rmtree(PUBLIC)
    shutil.copytree(MYST_OUT, PUBLIC)
    shutil.copytree(MKDOCS_OUT, PUBLIC / API_SUBDIR)
    for name in ROOT_FILES:
        shutil.copy2(MYST_DIR / name, PUBLIC / name)
    # The reference has no page of its own at its root (docs/index.md is the
    # prose home); send its root, which the theme's logo links to, to the
    # overview.
    home = PUBLIC / API_SUBDIR / "index.html"
    if not home.exists():
        home.write_text(
            f'<!doctype html><meta charset="utf-8"><title>API reference</title>'
            f'<meta http-equiv="refresh" content="0; url={API_HOME}">'
            f'<a href="{API_HOME}">API reference</a>\n',
            encoding="utf-8",
        )
    linked, unlinked = rewrite_prose_links(
        PUBLIC / API_SUBDIR, prose_slugs(MYST_OUT), base_url
    )
    logger.info("pointed {} API links into prose at the MyST pages", linked)

    inventory = parse_inventory((MKDOCS_OUT / "objects.inv").read_bytes())
    mapping = anchor_case_map(inventory)
    replacement = f"{base_url}/{API_SUBDIR}"

    fixed_anchors = 0
    rewritten = 0
    for page in PUBLIC.rglob("*.html"):
        if page.is_relative_to(PUBLIC / API_SUBDIR):
            continue
        html = page.read_text(encoding="utf-8")
        html, anchors = restore_anchor_case(html, mapping)
        html, origins = rewrite_api_origin(html, API_ORIGIN, replacement)
        if anchors or origins:
            page.write_text(html, encoding="utf-8")
        fixed_anchors += anchors
        rewritten += origins

    logger.info("repaired {} lowercased API anchors", fixed_anchors)
    logger.info("rewrote {} API links to {}", rewritten, replacement)

    problems: list[str] = []
    for page in PUBLIC.rglob("*.html"):
        if page.is_relative_to(PUBLIC / API_SUBDIR):
            continue
        text = page.read_text(encoding="utf-8", errors="replace")
        name = page.relative_to(PUBLIC)
        if API_ORIGIN in text:
            problems.append(f"{name}: build-time API origin leaked into output")
        for url in nav_base_url_problems(text):
            problems.append(
                f"{name}: site-nav URL {url!r} is root-relative; the theme will "
                f"prepend BASE_URL to it again on hydration. Use an absolute URL."
            )
    return unlinked + problems + verify_links(PUBLIC, base_url)


app = cyclopts.App(help=__doc__.splitlines()[0], result_action="return_value")


@app.default
def main(*, check: bool = False) -> int:
    """Run the documentation build.

    Args:
        check: Validate sources without rendering the themed MyST site.

    Returns:
        ``0`` on success, ``1`` if the assembled site has broken links.
    """
    base_url = os.environ.get("BASE_URL", "").rstrip("/")

    unbuilt = source_link_problems(MYST_DIR)
    if unbuilt:
        logger.error("{} prose link(s) into unbuilt folders:", len(unbuilt))
        for problem in unbuilt:
            logger.error("  - {}", problem)
        return 1

    build_api()
    build_prose(html=not check)
    if check:
        logger.success("sources validated (themed HTML not rendered)")
        return 0

    problems = assemble(base_url)
    if problems:
        logger.error("{} broken link(s) in the assembled site:", len(problems))
        for problem in problems:
            logger.error("  - {}", problem)
        return 1
    logger.success("assembled site is link-clean -> {}", PUBLIC)
    return 0


if __name__ == "__main__":
    raise SystemExit(app())
