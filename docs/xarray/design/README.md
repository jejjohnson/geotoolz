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

!!! note "Validation expansion"
    The validation additions extend the existing design docs. Existing
    examples in primitives, components, models, and integration should
    remain unchanged. New validation examples live in additional files.

## Structure

```
geo_toolz/
├── README.md              # This file
├── vision.md              # Motivation, user stories, design principles, identity
├── architecture.md        # Three-layer stack, Operator model, Graph API, inference, dependencies
├── validation.md          # Validation philosophy: scale, data, physical, process, phenomena
├── boundaries.md          # Ownership, ecosystem, scope, testing strategy, roadmap
├── api/
│   ├── README.md          # Submodule inventory, notation, import conventions
│   ├── primitives.md      # Layer 0 — pure functions by submodule
│   ├── components.md      # Layer 1 — Operator classes by submodule
│   ├── models.md          # Layer 2 — Graph API, ModelOp, inference
│   └── validation.md      # Validation API map by scientific question
├── examples/
│   ├── README.md          # Index and reading order
│   ├── primitives.md      # Layer 0 — pure function pipelines, pipe syntax
│   ├── components.md      # Layer 1 — Sequential, operator composition, Hydra
│   ├── models.md          # Layer 2 — Graph API, inference, model comparison
│   ├── integration.md     # Layer 3 — sklearn, xrpatcher, xarray_sklearn, ekalmX
│   ├── validation.md      # Field, spectral, lead-time, structural, process examples
│   ├── lagrangian.md      # Particle and drifter-style transport diagnostics
│   ├── budgets.md         # Heat/salt/volume/energy budget residual examples
│   └── phenomena.md       # Event/object detection and verification examples
├── decisions.md           # Design decisions D1–D10 (existing tradeoffs)
└── validation-decisions.md # Design decisions D11–D15 (validation framework)
```

## Reading Order

1. **[vision.md](vision.md)** — understand the why
2. **[architecture.md](architecture.md)** — understand the three-layer stack
3. **[validation.md](validation.md)** — understand the expanded validation philosophy
4. **[boundaries.md](boundaries.md)** — understand the scope
5. **[api/README.md](api/README.md)** — scan the surface
6. **[api/primitives.md](api/primitives.md)** → **[components.md](api/components.md)** → **[models.md](api/models.md)** → **[validation.md](api/validation.md)** — drill into detail
7. **[examples/primitives.md](examples/primitives.md)** → **[components.md](examples/components.md)** → **[models.md](examples/models.md)** → **[integration.md](examples/integration.md)** → **[validation.md](examples/validation.md)** — see it in action
8. **[decisions.md](decisions.md)** — D1–D10 architectural tradeoffs
9. **[validation-decisions.md](validation-decisions.md)** — D11–D15 validation framework decisions
