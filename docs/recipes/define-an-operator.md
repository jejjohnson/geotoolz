# Define an operator

The minimal viable `Operator` is two methods. This recipe walks through
writing one from scratch — what each method is for, the conventions
geotoolz follows, and the small disciplines that keep operators
composable.

## The contract

```python
from pipekit import Operator


class MyOp(Operator):
    def __init__(self, *, knob: float = 1.0) -> None:
        self.knob = knob

    def _apply(self, gt):           # the work
        ...

    def get_config(self):            # JSON-serialisable args
        return {"knob": self.knob}
```

That's it. Subclassing `Operator` gives you for free:

- `__call__` with dual-mode dispatch (eager on a value, graph-mode on
  an `Input` / `Node`).
- `__or__` so `op_a | op_b` builds a `Sequential`.
- `__repr__` derived from `get_config()`.

## Step 1 — keyword-only constructor

```python
def __init__(self, *, scale: float = 1e-4, clip: tuple[float, float] | None = None) -> None:
    self.scale = scale
    self.clip = clip
```

**Why keyword-only.** YAML / Hydra-zen configs serialise by name. A
positional constructor argument that doesn't appear in `get_config()`
is impossible to round-trip; keyword-only makes the mapping unambiguous.

## Step 2 — `_apply` does the work

```python
import numpy as np

def _apply(self, gt):
    out = np.asarray(gt) * self.scale
    if self.clip is not None:
        lo, hi = self.clip
        out = np.clip(out, lo, hi)
    return gt.array_as_geotensor(out)
```

**Conventions.**

- **Inputs are typed `GeoTensor`** for domain operators. The core
  algebra is carrier-agnostic, but RS operators narrow to `GeoTensor`
  at their own signature.
- **Wrap the result back via `gt.array_as_geotensor(out)`** so
  `transform`, `crs`, and `fill_value_default` propagate. Don't
  construct a new `GeoTensor` by hand unless you really need to.
- **Preserve trailing spatial dims.** If your op collapses the channel
  axis (e.g. NDVI), make sure the output's last two dims still agree
  with the input's `(H, W)` so `array_as_geotensor` accepts it.
- **Pure function inside.** Don't mutate the input array. If you need a
  scratch buffer, copy first.

## Step 3 — `get_config()` round-trips constructor args

```python
def get_config(self) -> dict:
    return {"scale": self.scale, "clip": self.clip}
```

**The discipline.** `MyOp(**op.get_config())` must produce an
equivalent operator. If your constructor accepts a callable, an open
file handle, or anything else that can't survive JSON, set
`forbid_in_yaml = True` and let `get_config()` return a debug repr.

```python
class Tap(Operator):
    forbid_in_yaml = True
    def __init__(self, fn) -> None:
        self.fn = fn
    def _apply(self, x):
        self.fn(x); return x
    def get_config(self):
        return {"fn": f"<callable {self.fn!r}>"}
```

## Step 4 (optional) — typed config via Pydantic

When the constructor has more than a few knobs or needs validation,
push them into a Pydantic model:

```python
from pydantic import BaseModel, Field


class ScaleCfg(BaseModel):
    scale: float = Field(1e-4, gt=0, description="Multiplicative scale")
    clip: tuple[float, float] | None = None


class Scale(Operator):
    def __init__(self, **kwargs) -> None:
        self.cfg = ScaleCfg(**kwargs)

    def _apply(self, gt):
        out = gt.values * self.cfg.scale
        if self.cfg.clip is not None:
            out = out.clip(*self.cfg.clip)
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict:
        return self.cfg.model_dump()
```

The typed model lives at the *config* boundary, not the carrier
boundary — inputs/outputs are still `GeoTensor`s. Validation runs once
at `__init__`, not on every `_apply`.

## Terminal operators

If your op returns `None` (writes to disk, displays, etc.), mark it
terminal so `Sequential` rejects it in any position except the last:

```python
class WriteCOG(Operator):
    _terminal = True

    def __init__(self, *, path: str) -> None:
        self.path = path

    def _apply(self, gt):
        gt.to_cog(self.path)        # returns None

    def get_config(self):
        return {"path": self.path}
```

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
    def get_config(self):
        return {"n": self.n}

assert (Add(n=1) | Add(n=2))(0) == 3
```

That's the same shape your `GeoTensor`-typed operator will use; you
just get faster fixtures.

## Worked example — `NDVI`

```python
import numpy as np
from pipekit import Operator


class NDVI(Operator):
    """(NIR - Red) / (NIR + Red + eps), preserves (H, W)."""

    def __init__(self, *, nir_idx: int = 3, red_idx: int = 2, eps: float = 1e-10) -> None:
        self.nir_idx, self.red_idx, self.eps = nir_idx, red_idx, eps

    def _apply(self, gt):
        a = np.asarray(gt)
        nir, red = a[self.nir_idx], a[self.red_idx]
        return gt.array_as_geotensor((nir - red) / (nir + red + self.eps))

    def get_config(self) -> dict:
        return {"nir_idx": self.nir_idx, "red_idx": self.red_idx, "eps": self.eps}
```

That's a complete, round-trippable operator in ~12 lines. The same
shape covers every domain operator in `geotoolz`.

## See also

- [Concepts](../concepts.md) — the model behind the `Operator` base
  class.
- [Composition core notebook](../notebooks/composition_core.ipynb) —
  every primitive against scalars, end-to-end.
- [Branching pipelines](branching-pipelines.md) — when one operator
  isn't enough.
