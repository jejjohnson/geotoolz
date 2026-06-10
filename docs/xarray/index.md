# xrtoolz

> Composable operators for geoprocessing Earth System Data Cubes.

`xrtoolz` gives every preprocessing, physics, inference, and evaluation
step a single shape — an **`Operator`**: a callable that maps
`xarray` in to `xarray` out, carries its configuration, and reports its
shape signature. Operators chain linearly with `Sequential` or wire into a
DAG with the functional `Graph` API — *the same operator works in both*.

The carrier-agnostic composition core lives in
[`pipekit`](https://github.com/jejjohnson/pipekit); `xrtoolz` adds the
xarray-specific operator families on top, organised by Earth-science
domain.

## Why xrtoolz

- **One abstraction, three layers.** Pure functions (Layer 0) → `Operator`
  wrappers (Layer 1) → `Graph` DAGs (Layer 2). Drop down a layer whenever
  you need the raw function; stay up high to compose.
- **Domain-organised.** Generic geoprocessing in [`geo`](api/geo/coords.md);
  true physics in [`ocn`](api/ocn.md) / `atm` / `rs`; numerics in
  [`calc`](api/calc.md) / [`transforms`](api/transforms.md) /
  [`interpolate`](api/interpolate.md); evaluation in
  [`metrics`](api/metrics.md) / [`budgets`](api/budgets.md).
- **Typed to the core.** Operators speak `xr.Dataset` / `xr.DataArray`; the
  private numpy kernels behind them are
  [jaxtyped](design/conventions/array-typing.md) so array shapes are part of
  the signature.
- **Bring your own model.** [`ModelOp`](api/inference.md) wraps a trained
  sklearn / JAX / framework-agnostic model as an operator — the ML backend
  is imported lazily, so you only install what you use.

## Install

```bash
uv add xrtoolz          # recommended (resolves pipekit from its source)
```

`xrtoolz` depends on [`pipekit`](https://github.com/jejjohnson/pipekit),
which is pre-PyPI and resolved via `[tool.uv.sources]`. A plain
`pip install` will fail until `pipekit` reaches PyPI — use `uv`:

```bash
uv pip install "git+https://github.com/jejjohnson/xrtoolz@main"
```

## Quickstart

Compute geostrophic surface currents from sea-surface height, then map them
— as a single pipeline:

```python
import xrtoolz
from xrtoolz import Sequential
from xrtoolz.geo import RemoveMean
from xrtoolz.ocn.operators import Streamfunction, GeostrophicVelocities
from xrtoolz.viz.validation import SpatialMapPanel

pipeline = Sequential(
    RemoveMean(var="ssh"),       # de-mean the SSH field
    Streamfunction(),            # ψ = g·η / f
    GeostrophicVelocities(),     # (u_g, v_g) from ψ
)

currents = pipeline(ds)          # xr.Dataset in → xr.Dataset out
fig = SpatialMapPanel(var="u")(currents)
```

The same three operators can be wired as a DAG with `Graph` when you need
branching or shared intermediates — see
[Composition](api/composition.md).

## Where to go next

<div class="grid cards" markdown>

- :material-cube-outline: **[Core Concepts](api/composition.md)** —
  `Operator`, `Sequential`, `Graph`, and the combinators.
- :material-book-open-variant: **[Tutorials](notebooks/operators_pipeline_demo.ipynb)** —
  executable notebooks for pipelines, ocean kinematics, gridding, and
  validation.
- :material-api: **[API Reference](api/index.md)** — every public symbol,
  grouped by subject.
- :material-drawing: **[Design](design/README.md)** — architecture,
  boundaries, conventions, and roadmap.

</div>
