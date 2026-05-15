# geotoolz

> Composable Operator library for remote sensing on top of `georeader.GeoTensor`.

`geotoolz` lets you express remote-sensing pipelines as a `Sequential` of
small composable Operators, or a `Graph` of named ops for branching /
multi-input fusion. Operators run eagerly on `GeoTensor`s, compose with the
`|` pipe operator, round-trip their configs for Hydra-zen integration, and
support inline observation (`Tap`, `Snapshot`, `ShapeTrace`) and control
flow (`Branch`, `Switch`, `Fanout`) without breaking the chain.

**Status:** pre-alpha (`0.0.1`). The **composition core** has landed —
`Operator`, `Sequential`, `Graph`, `ModelOp`, plus the v0.1 idiom library
(observers, control flow, building blocks). Domain operators (radiometry,
indices, cloud masking, sampling, inference) land in subsequent releases.

## Installation

Not yet on PyPI. Install from source:

```bash
git clone https://github.com/jejjohnson/geotoolz.git
cd geotoolz
make install
```

For the optional Hydra-zen integration:

```bash
uv pip install -e '.[hydra]'
```

## Quickstart

A composition core in five lines:

```python
import geotoolz as gz

# Compose with the pipe operator
pipe = gz.Tap(print) | gz.Identity()
pipe("hello, world")     # prints "hello, world", returns "hello, world"

# Or build a Sequential explicitly
pipe = gz.Sequential([gz.Tap(print), gz.Identity()])

# Branch on a predicate, fan out into multiple named outputs, etc.
gz.Fanout({"upper": gz.Lambda(str.upper), "lower": gz.Lambda(str.lower)})("Hi")
# {"upper": "HI", "lower": "hi"}
```

A more realistic shape (preview — domain ops not yet implemented):

```python
import geotoolz as gz

ndvi = gz.Sequential([
    gz.cloud.MaskClouds(qa_band="QA60", bits=[10, 11]),     # v0.2+
    gz.indices.NDVI(red_idx=2, nir_idx=3),                  # v0.2+
])
result = ndvi(gt)                                            # GeoTensor in, GeoTensor out
```

## What's available today

The v0.1 core composition layer ships:

- **`Operator`** — base class with dual-mode `__call__` (eager vs graph)
- **`Sequential`** — linear composition with terminal-op validation
- **`Graph`** / **`Input`** / **`Node`** — symbolic multi-input / multi-output graphs
- **`Fanout`** — sugar for one-input / many-output `Graph`s
- **`ModelOp`** — framework-agnostic inference wrapper (`torch` / `sklearn` / any callable)
- **`Tap`** / **`Snapshot`** / **`ShapeTrace`** — identity ops with side effects
- **`Branch`** / **`Switch`** — control flow
- **`Identity`** / **`Const`** / **`Lambda`** / **`Sink`** — small composable building blocks

Read the [Concepts] page for the model behind these primitives. Three
tutorial notebooks cover the surface from different angles — the
[Composition core notebook] walks through every primitive against
scalars; the [Pipeline idioms] notebook is a recipe gallery of observer
/ control-flow / QC patterns with build-your-own implementations for the
v0.2+ named ops; the [Deployment shapes] notebook tours 13 deployment
patterns (notebook, ETL, FastAPI, tile server, regulatory artifact,
orchestrator, …). The [Core API reference] documents each operator in
detail.

[Concepts]: concepts.md
[Composition core notebook]: notebooks/composition_core.ipynb
[Pipeline idioms]: notebooks/pipeline_idioms.ipynb
[Deployment shapes]: notebooks/deployment_shapes.ipynb
[Core API reference]: api/core.md

## Links

- [Concepts](concepts.md)
- [API Reference — Core](api/core.md)
- [Composition core notebook](notebooks/composition_core.ipynb)
- [Pipeline idioms](notebooks/pipeline_idioms.ipynb)
- [Deployment shapes](notebooks/deployment_shapes.ipynb)
- [Changelog](https://github.com/jejjohnson/geotoolz/blob/main/CHANGELOG.md)
- [GitHub](https://github.com/jejjohnson/geotoolz)
