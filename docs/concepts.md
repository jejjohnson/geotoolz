# Concepts

This page explains the model behind the composition core — what an `Operator`
is, how `Sequential` and `Graph` differ, when to reach for which primitive,
and the small discipline that keeps everything composable.

If you'd rather see code, skip to the [Composition core notebook][notebook]
which walks through every primitive end-to-end. This page is the *why*.

[notebook]: notebooks/composition_core.ipynb

## The composition algebra at a glance

```text
                ┌───────────────────────────────────────┐
                │           Your pipeline               │
                │  Sequential([op_a, op_b, op_c])       │
                │       or                              │
                │   op_a | op_b | op_c                  │
                │       or                              │
                │  Graph(inputs=..., outputs=...)       │
                └─────────────────┬─────────────────────┘
                                  │
                                  ▼
                ┌───────────────────────────────────────┐
                │           Operator (base)             │
                │   __call__: dual-mode dispatch        │
                │   _apply : subclass implements        │
                │   get_config: round-trip dict         │
                │   __or__ : compose with `|`           │
                └─────────────────┬─────────────────────┘
                                  │
                                  ▼
                ┌───────────────────────────────────────┐
                │            Carrier                    │
                │  (Any in core; GeoTensor in domain)   │
                └───────────────────────────────────────┘
```

The composition core is **carrier-agnostic** — the algebra works for
`GeoTensor`s (production), numpy arrays (lightweight tests), scalars, or
anything else. Domain operators (`NDVI`, `MaskClouds`, …) narrow to
`GeoTensor` at their own signatures; the core stays generic.

## What's an `Operator`?

An `Operator` is a thing you call. Subclasses implement `_apply`; the base
class handles two responsibilities you inherit for free:

1. **Dual-mode `__call__`** — running on a value invokes `_apply` (eager
   mode); running on an `Input` or `Node` records a `Node` in a `Graph`
   (graph mode). One method, two behaviours, dispatched on argument type.
2. **Config round-trip** — `get_config()` returns a JSON-serialisable dict
   of constructor args. Used for `__repr__`, pickling sanity, and the
   optional Hydra-zen integration.

The smallest possible operator:

```python
from geotoolz import Operator

class Add(Operator):
    def __init__(self, n: int) -> None:
        self.n = n

    def _apply(self, x: int) -> int:
        return x + self.n

    def get_config(self) -> dict:
        return {"n": self.n}

op = Add(5)
op(10)              # 15 — eager
repr(op)            # "Add(n=5)" — uses get_config()
```

That's the whole contract. Implement `_apply`, implement `get_config()`,
inherit everything else.

## `Sequential` — linear composition

`Sequential` threads the output of each operator into the next:

```python
from geotoolz import Sequential

pipe = Sequential([Add(1), Add(10), Add(100)])
pipe(0)             # 111
```

The same thing via the `|` operator (inherited from `Operator`):

```python
pipe = Add(1) | Add(10) | Add(100)    # Sequential([Add(1), Add(10), Add(100)])
```

The `|` operator **flattens nested `Sequential`s** — `a | (b | c)` and
`(a | b) | c` both produce a single three-element `Sequential`. No
nested wrappers, no surprises.

## Dual-mode `__call__` — eager vs graph

The same operator works in two modes:

```python
import geotoolz as gz

# Eager: pass a value, get a value back
Add(5)(10)                    # 15

# Graph mode: pass an Input/Node, get a Node back
x = gz.Input("x")
node = Add(5)(x)              # Node(operator=Add(5), parents=(x,))
```

The decision is automatic — `__call__` checks the argument type and routes
appropriately. Subclasses only ever implement `_apply`; the dispatch
happens once, in the base class.

## `Graph` — symbolic multi-input / multi-output composition

When your pipeline has branches, fan-out, or multiple inputs, `Sequential`
isn't enough. `Graph` builds a DAG by *calling operators on placeholders*:

```python
import geotoolz as gz

img = gz.Input("image")
ref = gz.Input("reference")

ndvi = NDVI(red_idx=2, nir_idx=3)(img)               # Node
rmse = RMSE(axis=(-2, -1))(ndvi, ref)                # multi-input Node

g = gz.Graph(
    inputs={"image": img, "reference": ref},
    outputs={"ndvi": ndvi, "rmse": rmse},
)

result = g(image=img_gt, reference=ref_gt)
# {"ndvi": GeoTensor, "rmse": scalar}
```

`Graph` topologically sorts the nodes, evaluates each exactly once, and
returns a dict keyed by output name. Cycle detection and unreachable-input
detection happen at construction time, not at `_apply` time.

`Graph` is itself an `Operator`, so it composes — you can put a `Graph`
inside a `Sequential`, or wrap one in `Fanout`.

## The v0.1 idiom library

Beyond the bare composition primitives (`Sequential`, `Graph`), the core
ships a small library of "operators you reach for constantly" — observers,
control flow, and tiny building blocks. The big idea: **the `Operator`
interface is general enough to express things that aren't just
transforms** — side effects, branching, defaults, escape hatches all
become first-class composable units that round-trip the same as any
transform.

### Identity-with-side-effect (observers)

| Operator | What it does |
|---|---|
| `Tap` | Calls `fn(gt)` and passes input through unchanged. Great for inline logging. |
| `Snapshot` | A *controller* — produces snapshot-taking operators via `snap.at(key)`. After the pipeline runs, intermediates are available as `snap[key]`. |
| `ShapeTrace` | Prints `shape`, `dtype`, `crs` at each step. `mode="diff_only"` skips redundant lines. |

```python
import geotoolz as gz

snap = gz.Snapshot()
pipe = gz.Sequential([
    Add(1), gz.Tap(print),               # observe
    snap.at("intermediate"),             # capture
    Add(10),
    snap.at("final"),
])
pipe(0)
snap["intermediate"]                     # 1
snap["final"]                            # 11
```

### Control flow

| Operator | What it does |
|---|---|
| `Branch` | `if predicate(x): if_true(x) else if_false(x)`. Default `if_false=Identity()`. |
| `Switch` | Multi-way dispatch on `key(x)`. Default `default=Identity()`. |

```python
gz.Branch(
    predicate=lambda x: x > 0,
    if_true=Add(100),
    if_false=Identity(),                 # no-op for negative inputs
)
```

### Composition

| Operator | What it does |
|---|---|
| `Fanout` | One input → dict of outputs. Sugar over a single-input `Graph`. |

```python
gz.Fanout({
    "doubled": gz.Lambda(lambda x: x * 2),
    "squared": gz.Lambda(lambda x: x * x),
})(5)                                    # {"doubled": 10, "squared": 25}
```

### Small but load-bearing building blocks

| Operator | What it does |
|---|---|
| `Identity` | Explicit no-op. Use in `Branch.if_false`, `Switch.default`, anywhere a slot needs an `Operator`. |
| `Const` | Return a fixed value regardless of input. Useful for test fixtures. |
| `Lambda` | Inline-callable escape hatch. Flagged `forbid_in_yaml = True` (closures don't round-trip). |
| `Sink` | Side-effect *terminal write that returns the input*. Composes (unlike a write op that returns `None`). |

```python
# Checkpoint an intermediate, keep going
gz.Sequential([
    expensive_step,
    gz.Sink(lambda gt: save_to_disk(gt, "checkpoint.tif")),
    next_step,                           # still receives the GeoTensor
])
```

### `ModelOp` — framework-agnostic inference

```python
op = gz.ModelOp(my_sklearn_classifier, method="predict")
predictions = op(features)               # arrays in, arrays out

# Or with a torch model and batched inference
op = gz.ModelOp(my_torch_unet, batch_size=8)
preds = op(chips)                        # chunks along axis 0, concatenates
```

`ModelOp` never imports a framework — it calls `model(arr)` or
`getattr(model, method)(arr)` directly. Use whatever fits.

## The `Carrier` type

```python
from geotoolz import Carrier
Carrier                                  # typing.Any (in v0.1)
```

`Carrier` is a deliberate type alias for `Any`. The composition core is
*carrier-agnostic* — the same algebra runs on `GeoTensor` (production),
ndarrays (tests), scalars, or anything else. When domain operators land,
they'll narrow to `Carrier` annotations that effectively mean "GeoTensor"
without ever forcing the core layer to import `georeader`.

Identity-preserving operators (`Identity`, `Tap`, `Snapshot`, `ShapeTrace`,
`Sink`) go one step further — their `_apply` uses an internal `TypeVar`
so a static type checker can narrow `op(x: T) -> T`. The carrier survives.

## Two small disciplines

### Round-trip discipline (`forbid_in_yaml`)

Operators that hold runtime closures (`Tap`, `Lambda`, `Branch`, `Switch`,
`Sink`, `ModelOp`) carry `forbid_in_yaml = True`. Their `get_config()` is a
debug repr, not a faithful YAML round-trip. The flag is documented for
future YAML loader enforcement — production pipelines that need to be
auditable should avoid closure-bearing operators, or accept that the YAML
artifact won't fully reproduce them.

### Terminal-operator validation (`_terminal`)

Some operators legitimately return `None` or otherwise break the carrier
chain (`WriteCOG`, viz operators). Mark them with `_terminal = True`;
`Sequential` then rejects them in any position except the last:

```python
class WriteCOG(Operator):
    _terminal = True
    def _apply(self, gt):
        save_to_disk(gt, self.path)
        # returns None — would break the next op in a Sequential

gz.Sequential([WriteCOG("/a"), Add(1)])  # TypeError
gz.Sequential([Add(1), WriteCOG("/a")])  # ok — terminal at end
```

`Sink` is **not** terminal — it performs a side effect *and* returns the
input. That's why `Sink` composes and `WriteCOG` doesn't.

## What's not in the core

Several operators that the algebra supports are deliberately deferred to
later releases. The [Pipeline idioms notebook][idioms] shows how to
build minimal versions of each using just the v0.1 primitives, until the
named ops ship:

- `Profile` / `TimeIt`, `Histogram`, `Spy` / `Hook`, `Diff` — observer
  family extensions; build on `Tap` once needed
- `Try` / `Fallback`, `Coalesce`, `Retry` — exception-handling control
  flow; needs careful design before shipping
- `ApplyToBands`, `Cache` / `Memoize`, `Provenance`, `Mode` — composition
  / stateful operators; bigger surface, separate PRs
- `Subsample` — shape-changing transform; belongs with `radiometry`
- `AssertX` / `Quarantine` — lives in `geotoolz.qc`, not `core`

And the entire **domain operator surface** — radiometry, indices, cloud
masking, atmospheric correction, compositing, pansharpening, SAR,
hyperspectral, sampling, inference, sensor presets — is in the v0.2+
roadmap. The composition algebra is ready; the operators come next.

## Where next

- **Hands-on:** the [Composition core notebook][notebook] runs every
  primitive end-to-end against scalars (no GeoTensor setup required).
- **Recipe book:** the [Pipeline idioms notebook][idioms] is a gallery
  of observer / control-flow / composition / QC patterns, with
  build-your-own implementations for the v0.2+ named ops.
- **Deployment patterns:** the [Deployment shapes notebook][shapes]
  tours 13 deployment patterns (notebook, ETL, FastAPI, tile server,
  regulatory artifact, orchestrator, …).
- **Reference:** the [Core API reference][api] documents each operator
  with its constructor signature and config keys.

[idioms]: notebooks/pipeline_idioms.ipynb
[shapes]: notebooks/deployment_shapes.ipynb
[api]: api/core.md
