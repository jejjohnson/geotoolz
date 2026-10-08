---
name: docs-diagram
description: Create or update an architecture / flow diagram for the geotoolz docs or READMEs — HTML source rendered to PNG with docs/assets/diagrams/render.py. Use when asked for a diagram, figure, flowchart or architecture picture in the docs, or when a change makes an existing diagram stale.
---

# Docs diagrams

Diagrams live in `docs/assets/diagrams/`: each `<name>.html` is the source,
`<name>.png` (rendered at 2×) is what docs and READMEs embed. Never edit a
PNG by hand and never draw one in another tool.

## Write the HTML

- Start from the closest existing diagram (`stack-layers.html`,
  `catalog-flow.html`, `patcher-axes.html`, `products-architecture.html`)
  and keep its structure: a `#diagram` root, `_base.css` for the palette and
  type, boxes laid out with CSS.
- Draw edges with `_connect.js`:
  `wire("a", "b", { from: "right", to: "left", label: "GeoTensor" })`
  (`dashed: true` for optional relations, `fromShift` / `toShift` to
  separate parallel wires), then call `ready()`.
- Use real public names (`geocatalog.patch.field_for`,
  `spatial.window.Hann`); when an API is renamed, grep the diagrams too.
- Keep it a summary: one box per concept, a few common methods — not every
  class.

## Render and check

```bash
uv run --no-project --with playwright python docs/assets/diagrams/render.py <name>.html
```

(Set `CHROMIUM_PATH` to use an existing Chromium instead of Playwright's.)
Open the PNG and check for clipped text, overlapping labels and wires
crossing boxes; adjust the HTML and re-render until it reads cleanly.
Commit the HTML and the PNG together, and embed the PNG with descriptive
`alt` text.
