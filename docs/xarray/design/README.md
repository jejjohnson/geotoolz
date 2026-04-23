---
status: draft
version: 0.1.0
---

# xr_toolz Design Doc

**A composable operator library for geoprocessing Earth System Data Cubes.**

!!! note "Adaptation note"
    These documents are the original `geo_toolz` design docs, imported
    verbatim as the architectural source of truth for `xr_toolz`. The
    vision, three-layer stack (L0 primitives → L1 operators → L2 graph),
    operator contract, decisions, and roadmap all carry over. The one
    concrete difference is package layout: `xr_toolz` organises
    submodules by Earth-science domain — `geo` (generic), `ocn`, `atm`
    (with `atm.gas.ch4`), `rs`, `ice` — rather than by feature. Anything
    domain-agnostic (validation, regrid, detrend, metrics, spectral,
    inference, …) lives under `xr_toolz.geo`; only true physics lives in
    the other domain submodules. Read occurrences of `geo_toolz.<topic>`
    in these docs as `xr_toolz.geo.<topic>` for domain-agnostic topics,
    and as `xr_toolz.<domain>.<topic>` for the physics chapters in
    `kinematics`.

## Structure

```
geo_toolz/
├── README.md              # This file
├── vision.md              # Motivation, user stories, design principles, identity
├── architecture.md        # Three-layer stack, Operator model, Graph API, inference, dependencies
├── boundaries.md          # Ownership, ecosystem, scope, testing strategy, roadmap
├── api/
│   ├── README.md          # Submodule inventory, notation, import conventions
│   ├── primitives.md      # Layer 0 — pure functions by submodule
│   ├── components.md      # Layer 1 — Operator classes by submodule
│   └── models.md          # Layer 2 — Graph API, ModelOp, inference
├── examples/
│   ├── README.md          # Index and reading order
│   ├── primitives.md      # Layer 0 — pure function pipelines, pipe syntax
│   ├── components.md      # Layer 1 — Sequential, operator composition, Hydra
│   ├── models.md          # Layer 2 — Graph API, inference, model comparison
│   └── integration.md     # Layer 3 — sklearn, xrpatcher, xarray_sklearn, ekalmX
└── decisions.md           # Design decisions with rationale
```

## Reading Order

1. **[vision.md](vision.md)** — understand the why
2. **[architecture.md](architecture.md)** — understand the three-layer stack
3. **[boundaries.md](boundaries.md)** — understand the scope
4. **[api/README.md](api/README.md)** — scan the surface
5. **[api/primitives.md](api/primitives.md)** → **[components.md](api/components.md)** → **[models.md](api/models.md)** — drill into detail
6. **[examples/primitives.md](examples/primitives.md)** → **[components.md](examples/components.md)** → **[models.md](examples/models.md)** → **[integration.md](examples/integration.md)** — see it in action
7. **[decisions.md](decisions.md)** — understand the tradeoffs
