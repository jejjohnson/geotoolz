# Define an operator

The minimal viable `Operator` is a keyword-only constructor plus one
method. This recipe walks through writing one from scratch — what each
piece is for, the conventions geotoolz follows, and the small
disciplines that keep operators composable.

## The contract

```python
from pipekit import Operator


class MyOp(Operator):
    def __init__(self, *, knob: float = 1.0) -> None:
        self.knob = knob            # stored under the argument's own name

    def _apply(self, gt):           # the work
        ...
```

That's it. Subclassing `Operator` gives you for free:

- `__call__` with dual-mode dispatch (eager on a value, graph-mode on
  an `Input` / `Node`).
- `__or__` so `op_a | op_b` builds a `Sequential`.
- `get_config()`, derived from the `__init__` signature (see Step 3),
  and a `__repr__` built from it.

## Step 1 — keyword-only constructor

```python
def __init__(self, *, scale: float = 1e-4, clip: tuple[float, float] | None = None) -> None:
    self.scale = scale
    self.clip = clip
```

**Why keyword-only.** YAML / Hydra-zen configs serialise by name, and a
keyword-only signature makes the mapping from config to constructor
unambiguous. Every geotoolz operator follows this rule — including the
ones that wrap an estimator or model (`ModelOp(model=net)`).

**Use the shared vocabulary.** One concept has one parameter name across
the library: a single band is named for what it is (`red`, `nir`,
`qa_band`, … — an integer position or a band name, never `nir_idx`),
several bands are `bands`, a value written into pixels is `fill_value`,
a neighbourhood side length is `window`, an RNG seed is `seed`, a
denominator stabiliser is `eps`. Reusing the names keeps configs
readable and lets your operator sit next to the built-ins.

## Step 2 — `_apply` does the work

```python
import numpy as np
from geotoolz._src.wrap import wrap_like


def _apply(self, gt):
    out = np.asarray(gt, dtype=np.float32) * self.scale
    if self.clip is not None:
        lo, hi = self.clip
        out = np.clip(out, lo, hi)
    return wrap_like(gt, out)
```

**Conventions.**

- **Accept `GeoTensor` and plain arrays.** Read the values with
  `np.asarray(gt)`; `GeoTensor` is an `np.ndarray` subclass, so the same
  code runs on both carriers.
- **Rewrap the result with `geotoolz._src.wrap.wrap_like(gt, out)`** so
  `transform`, `crs`, `attrs` and `fill_value_default` propagate from the
  input (a plain-array input comes back as a plain array). Don't
  construct a new `GeoTensor` by hand unless you really need to.
- **Declare a fill value that matches the output.** georeader treats
  `fill_value_default` as nodata (`validmask()` is `values != fill`),
  even when it is `0`, so an inherited fill is wrong once the output's
  dtype or meaning changes. Pass it explicitly —
  `wrap_like(gt, out, fill_value_default=...)` (or
  `geotoolz._src.valid.wrap_filled`, which also writes the fill into the
  input's nodata pixels): `False` for boolean masks, `0` for label /
  count maps, `NaN` for new float quantities (indices, scores, features),
  and `geotoolz._src.valid.carried_fill(gt, out.dtype)` for outputs that
  carry the input's values. If `0` is real data in your input, give it
  `fill_value_default=None` (or `NaN`) rather than georeader's default `0`.
- **Preserve trailing spatial dims.** If your op collapses the channel
  axis (e.g. NDVI), make sure the output's last two dims still agree
  with the input's `(H, W)` so `wrap_like` accepts it (pass
  `transform=` for outputs on a new grid).
- **Pure function inside.** Don't mutate the input array. If you need a
  scratch buffer, copy first.

## Step 3 — `get_config()` comes for free

`Operator` inherits pipekit's `ConfigMixin`, which derives `get_config()`
from the `__init__` signature: for every parameter it reads the instance
attribute of the same name. Storing each argument under its own name
(Step 1) is therefore all it takes:

```python
op = Scale(scale=2e-4, clip=(0.0, 1.0))
op.get_config()                  # {"scale": 0.0002, "clip": (0.0, 1.0)}
Scale(**op.get_config())         # an equivalent operator
```

**The discipline.** `MyOp(**op.get_config())` must produce an
equivalent operator. Keep runtime-only state out of it with
`__config_exclude__ = ("cache",)`. If your constructor accepts a
callable, an open file handle, or anything else that can't survive
JSON, set `forbid_in_yaml = True`: YAML loaders and `from_state` then
refuse to rebuild the operator, and its config is a debug repr only.

```python
class Tap(Operator):
    forbid_in_yaml = True

    def __init__(self, *, fn) -> None:
        self.fn = fn

    def _apply(self, x):
        self.fn(x)
        return x
```

Override `get_config()` by hand only when the config genuinely differs
from the constructor arguments (pipekit's `Sequential`, whose config is
the list of nested operators, is the canonical case).

## Step 4 (optional) — validate arguments with Pydantic

When the constructor has more than a few knobs or needs validation,
validate them with a Pydantic model at construction time — and still
store the validated values under the argument names, so `get_config()`
stays automatic:

```python
from pydantic import BaseModel, Field


class ScaleCfg(BaseModel):
    scale: float = Field(1e-4, gt=0, description="Multiplicative scale")
    clip: tuple[float, float] | None = None


class Scale(Operator):
    def __init__(self, *, scale: float = 1e-4, clip: tuple[float, float] | None = None) -> None:
        cfg = ScaleCfg(scale=scale, clip=clip)
        self.scale, self.clip = cfg.scale, cfg.clip

    def _apply(self, gt):
        out = np.asarray(gt, dtype=np.float32) * self.scale
        if self.clip is not None:
            out = out.clip(*self.clip)
        return wrap_like(gt, out)
```

The typed model lives at the *config* boundary, not the carrier
boundary — inputs/outputs are still `GeoTensor`s. Validation runs once
at `__init__`, not on every `_apply`.

## Terminal operators

If your op returns `None` (writes to disk, displays, etc.), mark it
terminal so `Sequential` rejects it in any position except the last:

```python
from georeader.save import save_cog


class SaveCOG(Operator):
    _terminal = True

    def __init__(self, *, path: str) -> None:
        self.path = path

    def _apply(self, gt):
        save_cog(gt, self.path)     # returns None
```

(The library's own writers — `gz.WriteGeoTIFF`, `gz.WriteCOG`,
`gz.WriteZarr` — subclass `geotoolz.io.SinkOperator`, which sets
`_terminal = True` for you.)

If you want side effects mid-chain *and* to keep the carrier flowing,
use `Sink(fn)` instead — it runs `fn(gt)` and returns the input
unchanged.

## Test it on scalars first

The core algebra is carrier-agnostic. You can write the dispatch /
composition tests against scalars and only swap in `GeoTensor`s once
the math is right:

```python
class Add(Operator):
    def __init__(self, *, n: int) -> None:
        self.n = n

    def _apply(self, x):
        return x + self.n


assert (Add(n=1) | Add(n=2))(0) == 3
assert Add(n=1).get_config() == {"n": 1}
```

That's the same shape your `GeoTensor`-typed operator will use; you
just get faster fixtures.

## Worked example — `NDVI`

```python
import numpy as np
from pipekit import Operator
from geotoolz._src.wrap import wrap_like


class NDVI(Operator):
    """(NIR - Red) / (NIR + Red + eps); collapses the band axis, keeps (H, W)."""

    def __init__(self, *, nir: int = 3, red: int = 2, eps: float = 1e-10) -> None:
        self.nir, self.red, self.eps = nir, red, eps

    def _apply(self, gt):
        a = np.asarray(gt, dtype=np.float32)
        nir, red = a[self.nir], a[self.red]
        return wrap_like(gt, (nir - red) / (nir + red + self.eps), fill_value_default=np.nan)
```

That's a complete, round-trippable operator in ~10 lines. The built-in
`gz.NDVI` has the same shape, plus band-name resolution
(`gz.NDVI(nir="B08", red="B04")`), nodata judging and `(T, C, H, W)`
stacks.

## See also

- [Concepts](../concepts.md) — the model behind the `Operator` base
  class.
- [Composition core notebook](https://github.com/jejjohnson/research_notebook/blob/main/projects/geostack/notebooks/01_composition_core.ipynb) —
  every primitive against scalars, end-to-end (lives in the
  research_notebook geostack project).
- [Branching pipelines](branching-pipelines.md) — when one operator
  isn't enough.
