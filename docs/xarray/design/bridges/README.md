---
status: draft
version: 0.1.0
---

# xrtoolz bridge modules — design docs

Three helper subpackages that connect labeled xarray data (`DataArray` /
`Dataset` / `DataTree`) to the three Python toolchains xrtoolz pipelines
keep reaching for:

| Module             | Bridges                       | Purpose                                                                                                                          |
| ------------------ | ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `xrtoolz.einx`     | [einx]                        | Named-tensor algebra (einsum, rearrange, reduce, …) where the named axes *are* the DataArray dims.                               |
| `xrtoolz.linalg`   | [gaussx] / [lineax] / [matfree] | Structured linear operators, solvers, logdet, sampling — all in DataArray-space, with physical dims as the row/column labels.    |
| `xrtoolz.prob`     | [numpyro] / [pyrox]           | Probability/statistics on DataArrays: distributions parameterized by named tensors, sampling, log-probability, and posterior IO. |

[einx]: https://github.com/fferflo/einx
[gaussx]: https://github.com/jejjohnson/gaussx
[lineax]: https://github.com/patrick-kidger/lineax
[matfree]: https://github.com/pnkraemer/matfree
[numpyro]: https://num.pyro.ai/
[pyrox]: https://github.com/jejjohnson/pyrox

The three modules are siblings of the existing domain packages
(`xrtoolz.geo`, `xrtoolz.ocn`, `xrtoolz.atm`, `xrtoolz.rs`,
`xrtoolz.ice`) and follow the same Layer-0 / Layer-1 split documented in
`CLAUDE.md` and `docs/design/`.

## Structure

```
design_docs/xrtoolz/
├── README.md          # This file — index + cross-cutting principles
├── overview.md        # Shared motivation, contracts, and dependency model
├── einx/
│   ├── vision.md      # Why bridge einx; user stories
│   ├── architecture.md
│   ├── api.md
│   └── decisions.md
├── linalg/            # gaussx + lineax bridge
│   ├── vision.md
│   ├── architecture.md
│   ├── api.md
│   └── decisions.md
└── prob/              # numpyro + pyrox bridge
    ├── vision.md
    ├── architecture.md
    ├── api.md
    └── decisions.md
```

## Reading order

1. **[overview.md](overview.md)** — the shared model: DataArray as the
   carrier, named dims as the type, lazy backend imports, Layer-0
   functions + thin `Operator` wrappers.
2. **[einx/vision.md](einx/vision.md)** — start here for the lightest
   bridge, since `linalg` and `prob` lean on the same dim-bookkeeping
   patterns.
3. **[linalg/vision.md](linalg/vision.md)** — gaussx-flavoured structured
   linear algebra with physical dims as row/column labels.
4. **[prob/vision.md](prob/vision.md)** — distributions whose parameters
   are DataArrays.
5. Inside each subpackage: `architecture.md` → `api.md` →
   `decisions.md`.

## Cross-cutting principles

These hold for all three modules; see `overview.md` for the reasoning.

1. **DataArray-first.** Layer-0 functions take and return `DataArray`s.
   `Dataset` is supported via small adapters (`pack` / `unpack`), and
   `DataTree` is inherited for free from `xrtoolz.Operator`'s leaf-wise
   dispatch.
2. **Named dims are the type.** The dim *names* on the inputs determine
   which axis a backend operates on; sizes are inferred. No positional
   axis indices in the public API.
3. **Coordinate semantics.** Each function documents which coords it
   preserves, which it drops, and how broadcasting interacts with
   labeled coords. The defaults match xarray's: aligned-by-coord on
   shared dims, broadcast on disjoint dims.
4. **Layer-0 functions + Layer-1 operators.** Every public function has
   a thin `Operator` subclass with the same name (e.g. `einsum` →
   `Einsum`) so the same logic runs in `Sequential` / `Graph`. No logic
   is duplicated; the `Operator` calls the function in `_apply`.
5. **Backend deps are lazy and opt-in.** Following the existing
   `xrtoolz.inference` pattern (D4): backends are imported inside
   functions, never at module load. Each module ships as a separate
   extra — install one at a time with ``pip install xrtoolz[einx]``,
   ``pip install xrtoolz[linalg]``, or ``pip install xrtoolz[prob]``,
   or combine them in a single command (``pip install
   "xrtoolz[einx,linalg,prob]"``).
6. **No reimplementation.** einx owns named-tensor semantics, gaussx
   owns linear-algebra dispatch, numpyro owns inference. These bridges
   only translate between labeled and array carriers.

## Status

- **`einx`** — **implemented** (`src/xrtoolz/einx/`). einx is a core
  dependency (reverses decision D9). The four pattern verbs
  (`einsum` / `rearrange` / `reduce` / `repeat`), the `matmul` /
  `outer` / `batch_matmul` conveniences, `pack_dataset` /
  `unpack_dataset`, and the Layer-1 operators are live. Some signatures
  shifted from the original draft (notably: `rearrange` is positional on
  the input's current dim order, and `reduce` takes ``op=`` keyword-only).
- **`linalg`**, **`prob`** — still **draft / pre-implementation**.

The docs describe the proposed API surface; concrete signatures may
shift once each implementation lands.
