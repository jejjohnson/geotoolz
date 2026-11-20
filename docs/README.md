# Documentation build

geotoolz's documentation is built by **two tools**, deployed as one site with a section per package.

| Half | Tool | Source | Deployed at |
|---|---|---|---|
| Prose — overviews, concepts, how-tos, tutorials, design pages, notebooks | [mystmd](https://mystmd.org) | `docs/myst.yml` (its `toc`) and every page under `docs/` that is not an API page | `/` |
| API reference | [Zensical](https://zensical.org) + mkdocstrings | `mkdocs.yml`; `docs/<pkg>/api/`, `docs/<pkg>/api.md` and the overview `docs/api/index.md` | `/reference/` (overview at `/reference/api/`) |

Build both and assemble them with:

```bash
make docs          # build both halves into public/ and check every link
make docs-serve    # build, then serve public/ at http://127.0.0.1:8000
make docs-api      # build only the API reference into site/
```

mystmd is a Node CLI: install it with `npm install -g mystmd`. `.github/workflows/docs.yml` builds and checks the site on every pull request whose changes reach it (`scripts/ci_select.py`), and `.github/workflows/pages.yml` deploys `public/` from `main` with GitHub Actions.

## The table of contents

`docs/myst.yml` lists every page, one section per package as before. A page that is not in its `toc` is not built, so add each new page and notebook there. mystmd gives every page a flat URL from its file name (`catalog/how-to/staging.md` is served at `/staging/`, with `_` turned into `-`). Every package has an `index.md` and a `concepts.md`, and the xarray design pages repeat names too, and mystmd has no way to set a slug, so the second page of a name is served at `<name>-1/`, the third at `<name>-2/`, in `toc` order; the nav shows each page's own title. Those numbers move when the `toc` changes, so never write a suffixed URL by hand: link to the file (`../concepts.md`), and let the build resolve it. The nav shows the `title` given to the page's entry in the `toc`, and the page keeps its first `# ` heading as its title.

MkDocs-Material syntax renders as plain text in MyST, without a warning: use `:::{note}` (with `:class: dropdown` for a collapsible) rather than `!!! note` or `??? note`, `::::{tab-set}` with `:::{tab-item}` rather than `=== "…"`, `::::{grid}` with `:::{card}` rather than a `grid cards` div, and an HTML `<img width=…>` rather than `{ width=… }`. A link to a heading on another page needs an explicit `(label)=` above that heading. The API pages are still MkDocs Markdown, read by Zensical.

The API half keeps its own `nav` in `mkdocs.yml`.

## The API pages stay beside each package's prose

The API pages live in `docs/<pkg>/api/` (and `docs/<pkg>/api.md` for geocloud, geoproducts and geoarray), where `scripts/capabilities.py`, `scripts/family_parity.py` and the docs-path tests read and write them. Zensical therefore reads `docs/` itself, and `exclude_docs` in `mkdocs.yml` leaves out everything but those pages; `docs/myst.yml` excludes the same pages, so each page is built by exactly one tool. Add a new API folder to both lists. Because the reference mirrors the source tree, an API page links to a prose page by its file, as before (`[Model states](../how-to/model-states.md)`): Zensical renders that link as the URL the page would have had, and `rewrite_prose_links` in `scripts/build_docs.py` replaces it with the URL mystmd gave the file. The reference's own root, `/reference/`, redirects to the overview.

Zensical does not run MkDocs hooks, but it renders with Python-Markdown, so the xarray lane's mkdocstrings options come from `docs/hooks/xarray_options.py`, a Markdown extension that adds them to every `::: xrops.…` and `::: xrproducts.…` directive. `mkdocs.yml` names it, and `scripts/build_docs.py` and `make docs-api` put `docs/hooks` on `PYTHONPATH`.

The old site's redirects (`mkdocs-redirects`) are gone: every URL changed with the move to flat MyST URLs, and the links in this repository point at the new ones.

## Why the split

mystmd is much better at prose: real cross-references, first-class notebook execution, exports to PDF/LaTeX, and MyST's directive syntax. What it does not have is an autodoc equivalent — there is no mature way to render Python docstrings into a MyST site today.

MkDocs + mkdocstrings already does that well, and it publishes a Sphinx-compatible `objects.inv`, which is exactly what mystmd needs to cross-reference *into* it. So each tool does the half it is good at.

The API half is built by [Zensical](https://zensical.org), the successor to Material for MkDocs (which reached end of life on 2026-11-05). It reads `mkdocs.yml` as is and runs mkdocstrings natively, so the config, the API pages and the inventory are unchanged; only the build command is `zensical build` instead of `mkdocs build`. Zensical does not run arbitrary MkDocs plugins or hooks (unsupported ones are silently ignored), so check <https://zensical.org/docs/compatibility/mkdocs/plugins/> before adding one.

## Site-nav URLs must be absolute

The theme re-renders the site nav from the config it embeds for hydration, and prepends `BASE_URL` to any nav URL starting with `/`. A nav entry that already carries the deployment prefix is therefore doubled — `/geotoolz/geotoolz/reference/` — and 404s.

This bites only *after* hydration: the server-rendered HTML is correct, so `curl` and `verify_links` both see a healthy link. Keep `site.nav` URLs absolute (mystmd rejects `/reference/` and `reference/` anyway) and never rewrite them at build time. `nav_base_url_problems` in `scripts/build_docs.py` fails the build if one becomes root-relative.

The trade-off is that a local preview's nav button points at the deployed site. In-page `xref:` links are unaffected — the theme does not re-prefix those, so they are rewritten to `{BASE_URL}/reference/...` as normal.

## If the theme download is blocked

`myst build --html` fetches the site template as a zip from GitHub. Behind a corporate proxy or a restrictive egress policy that request can fail with a 403 while ordinary git access still works. Clone the template and point at it locally instead:

```bash
git clone --depth 1 https://github.com/myst-templates/book-theme.git /tmp/book-theme
```

Then set `site.template` in `docs/myst.yml` to `/tmp/book-theme` for that build. Everything else is unchanged.

Note that the rendered pages still load KaTeX, Font Awesome, and jupyter-matplotlib stylesheets from CDNs at view time. Without network access in the *browser*, maths renders doubled — KaTeX ships an accessibility MathML copy that its stylesheet is responsible for hiding. That is a viewing artefact, not a build problem.

## Cross-references from prose into the API

In any MyST page, link to an API object with the `xref:` protocol and the name as exported from the top-level package:

```markdown
[`NDVI`](xref:api#geoops.NDVI)
[`RegridLike`](xref:api#xrops.interpolate.RegridLike)
```

A target that does not exist in the inventory fails `myst build --strict`, so broken API links are caught on the pull request rather than in production.

## Never link to a file mystmd does not build

mystmd builds only the pages in its `toc`. A relative prose link to any other Markdown file, such as an API page under `docs/api/` (`[Core](../operators/api/core.md)`), does not fail: mystmd publishes the file as a raw download, the link resolves, and the reader gets a Markdown file. A link to a folder (`[examples/](examples/)`) does not resolve at all. Link to an API object with `xref:api#<name>`, to a whole API page by its deployed URL (`https://jejjohnson.github.io/geotoolz/reference/operators/api/core/`), and to a prose page only once it is in the `toc`. `source_link_problems` in `scripts/build_docs.py` fails the build on any of these; the API pages it recognises are the `project.exclude` patterns of `docs/myst.yml`.

## Known upstream issue: the `$` anchor abbreviation

A Sphinx inventory may record an object's anchor as the literal `$`, meaning "the anchor is the object's own name". mystmd 1.11.0 **lowercases the name when it expands that abbreviation**, which breaks links into mkdocstrings' case-sensitive anchors:

| Inventory entry | mystmd emits | Correct? |
|---|---|---|
| `pkg.Name` -> `page/#pkg.Name` | `#pkg.Name` | yes |
| `pkg.mod.Name` -> `page/#$` | `#pkg.mod.name` | **no** |

The failure is silent: the link resolves, the page loads, and the browser simply cannot find the anchor.

**In practice this does not bite** when you link to a name as its package exports it: mkdocstrings gives every name an API page lists with `::: package.Name` an *explicit* anchor. So the natural way to write a link is also the safe one:

```markdown
[`NDVI`](xref:api#geoops.NDVI)            <!-- safe: explicit anchor -->
[`Name`](xref:api#pkg.mod.Name)          <!-- risky if its anchor is `$` -->
```

Prefer the top-level name. Two safety nets back that up in `scripts/build_docs.py`:

- `restore_anchor_case` repairs the lowercased anchors after the build, using the inventory as the source of truth. Delete it once mystmd fixes the expansion upstream.
- `verify_links` checks that **every** internal link in the assembled site resolves to a file that exists and, when it carries a fragment, to an anchor that is really in that file. It sees both generators' output at once, so it catches this class of bug regardless of cause — keep it either way.

Both operate on the **static** HTML. Anything the theme re-renders on hydration — the site nav above being the case that actually bit us — is invisible to them, which is why that one needs its own check.
