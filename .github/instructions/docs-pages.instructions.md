---
applyTo: "docs/**/*.md,README.md,packages/*/README.md"
---

# Documentation Pages — Standards

The READMEs set the bar: a reader learns **why** a package exists, sees
**one picture** of how it works, and runs a **typed, shape-annotated**
example within a minute. Every docs page meets the same bar. Notebooks
follow [`docs-examples.instructions.md`](docs-examples.instructions.md).

## One job per page

Each page is one of five kinds. Pick the kind first; a page that tries to
be two is split.

| Kind | Answers | Lives at | Shape |
|---|---|---|---|
| **Landing** | What is this, should I use it, how do I start? | `docs/<pkg>/index.md` (the stack's: `docs/index.md`) | the package README, adapted (below) |
| **Concepts** | How does it think? | `docs/<pkg>/concepts.md` | the mental model, the contract, the vocabulary — one diagram, small examples |
| **How-to** | How do I do *X*? | `docs/<pkg>/how-to/<task>.md` (`recipes/` in older sections) | goal → minimal runnable example → variations → pitfalls |
| **Tutorial** | Show me end to end on real data | `docs/<pkg>/notebooks/*.ipynb` | executed notebook (see the notebook standard) |
| **Reference** | What exactly does `f` take? | `docs/<pkg>/api/*.md` | mkdocstrings `:::` directives, one heading per public namespace, a one-line intro each |

Design records (`decisions.md`, `design/*.md`) are kept for history and
linked from Concepts; they are not entry points and are not rewritten to
this standard.

## The landing page

Mirror the package README, in this order:

1. `# <dist name>` and a one-sentence **pitch** (a `>` blockquote).
2. **Where it comes from** — the problem in the reader's words, then the
   idea that solves it. Two short paragraphs, no API names in the first.
3. **One rendered diagram** (below).
4. **Install** — the `pip install` lines and an **extras table**
   (*Extra · Pulls in · Needed for*).
5. **Quickstart** — one typed example that runs as written (below).
6. **What's inside** — a table of the public namespaces, each with a
   one-line purpose and a link to its how-to or reference page.
7. **Advanced** (optional) — the example that shows why the design pays
   off, usually across packages.
8. **Next steps** — links to Concepts, the how-tos, the tutorial, the API.

## Code examples

Every Python fence is checked by `scripts/check_docs.py` (CI): it must
parse, and every name it takes from the stack must exist.

- **Typed bindings.** Annotate every name an example binds:
  `ndvi: GeoTensor = gz.NDVI(red=0, nir=1)(scene)`. Operators and
  pipelines get their class (`pipe: gz.Sequential = …`).
- **Shapes and dtypes.** Every array-valued line ends with a comment
  giving its shape and dtype: `# (2, 4500, 4000) uint16`. A line that
  transforms an array gives the transition instead:
  `# (2, h, w) uint16 → (h, w) float32`. Keep these comments in one
  aligned column per fence. Use symbolic sizes (`(C, H, W)`,
  `(T, 1, h, w)`) when they vary — name them once in a `# Shapes: C bands,
  H × W pixels` first comment when a fence uses several — and say what the
  fill is when it matters (`· NaN = no data`).
- **Real names only.** Call the public API as it is on `main` (the checker
  enforces it). A planned feature is never shown as working code.
- **Self-contained.** A fence runs on its own after its imports: no
  names from an earlier fence, no undefined `scene` / `sas` / `token`.
  Build tiny synthetic inputs (`np.random.default_rng(0)…`) or name the
  real public data (`s3://sentinel-cogs/…`, `noaa-goes19`). Secrets come
  from the environment (`os.environ["…"]`).
- **Imports first**, stack aliases as everywhere else: `import geotoolz
  as gz`, `import geopatcher as gp`, `import geocatalog as gc`, `from
  geocloud import files`, `from geoproducts import goes`.
- **Notebook-only syntax is labelled.** Top-level `await` gets a comment
  `# in a notebook / async function`.
- **Placeholders are explicit.** A fence that cannot run as written (a
  `<your-bucket>` sketch) is preceded by `<!-- docs-check: skip -->` and
  says so in prose.
- `console` / `bash` fences for shell; `toml` / `yaml` for config.

## Diagrams

- **Rendered, not drawn inline.** Architecture and flow pictures are
  HTML sources in `docs/assets/diagrams/` rendered to PNG with
  `render.py` (the `docs-diagram` skill). Embed the PNG with alt text that
  states what the picture shows.
- One diagram per landing page; Concepts may have more. Reuse a README
  diagram rather than drawing a near copy.
- Mermaid is kept only for small decision trees ("is this the right
  tool?") that change with the code.

## One home per fact

Redundancy is the main way docs rot: two copies drift, and the reader
cannot tell which is right.

- **Each fact, table and example has one home.** The extras table lives on
  the landing page; the boundary-policy table on the reference page; the
  "define an operator" walkthrough in its how-to. Everywhere else links to
  it in one sentence ("see [Boundary policy](…)").
- **The README and the landing page are the same story**, not two drafts:
  the landing page may go deeper, but its pitch, diagram, quickstart and
  extras table match the README's. Change both in one PR.
- **The stack-level facts** (the five packages, how they depend on each
  other, the seams) live on the site home and "How the packages interlock";
  package pages link there rather than re-explaining the stack.
- **Concepts explain, how-tos do.** A how-to links to the concept it relies
  on instead of re-teaching it; a concepts page shows a small example and
  links to the how-to for the full recipe.
- **A page that only repeats another is removed** (with a redirect), not
  kept "for completeness".

## Readability

- **Lead with the point.** The first sentence of a page and of each
  section says what the reader gets or can do — never "This page
  describes…".
- **Short.** Sentences under 25 words, paragraphs of at most 3 sentences,
  pages that fit their one job. Split a page that needs a table of
  contents to navigate.
- **Scannable.** Headings are tasks or nouns the reader searches for
  (`Download a file`, `Boundary policy`), not "Overview of the X
  subsystem". Bullets for parallel items, numbered lists for steps,
  tables for options and comparisons.
- **Plain words.** No "robust", "seamless", "leverage", "powerful",
  "simply" or "just". Numbers carry units.

## Words

- Short sentences, active voice, the reader as "you".
- Lead with what the reader can do, not with what the code is.
- Name things exactly as the code does, with their public path the first
  time (`geocloud.files.download`), then the short name.
- Facts that drift (package counts, version floors, extra contents) are
  stated once — on the landing page — and linked, not repeated.
- Tables for comparisons and option lists; bullets for steps; prose for
  reasons.

## Checklist for a page

- [ ] One kind; the landing / how-to shape above
- [ ] Every fence typed, shape-annotated and self-contained
- [ ] No fact, table or example repeated from another page — link instead
- [ ] Leads with the point; sentences < 25 words, paragraphs ≤ 3 sentences
- [ ] `uv run python scripts/check_docs.py` passes
- [ ] Diagrams are rendered PNGs with alt text
- [ ] `uv run --group docs mkdocs build --strict` passes
- [ ] In `mkdocs.yml` nav, under the section's Landing / Concepts / How-to /
      Tutorials / Reference grouping
