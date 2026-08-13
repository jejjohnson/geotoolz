# xrtoolz-core — the xarray Operator base

```bash
# Pre-PyPI, install from the workspace via git (with its pipekit base);
# once published this becomes a plain `pip install xrtoolz-core`.
uv pip install \
  "xrtoolz-core @ git+https://github.com/jejjohnson/xrtoolz@main#subdirectory=packages/xrtoolz-core" \
  "pipekit @ git+https://github.com/jejjohnson/pipekit@main#subdirectory=packages/pipekit"
```

`xrcore` is ~330 lines with one job: make `pipekit.Operator` fluent in
xarray's containers. It ships two names:

- **`xrcore.Operator`** — the base class every xarray-facing operator
  in the stack subclasses.
- **`xrcore.Signature`** — the dict-keyed shape descriptor threaded
  through pipelines for keras-style `summary()` tables.

## The three dispatch modes

`pipekit.Operator` knows two call modes: eager `_apply` and symbolic
`Node` construction. Earth-science data also flows through
`xarray.DataTree` (multi-group / multi-resolution hierarchies), so
`xrcore.Operator` widens `__call__` with one additional branch:

1. **Symbolic** — any `Node` argument routes to pipekit's graph
   construction, recording the operator and its parents.
2. **DataTree** — any `DataTree` argument maps `_apply` over every
   leaf via `xr.map_over_datasets`, reassembling a tree with the
   input's structure.
3. **Eager** — plain `Dataset` / `DataArray` arguments hit `_apply`
   directly.

The consequence: every operator that inherits from this class — every
xrtoolz diagnostic, every combinator, every `Sequential` or `Graph`
built from them — gains DataTree support for free, with a single
`_apply` implementation.

```python
import numpy as np
import xarray as xr
from xrcore import Operator

class Scale(Operator):
    def __init__(self, factor: float) -> None:
        self.factor = factor

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        return ds * self.factor

ds = xr.Dataset({"ssh": ("x", np.arange(3.0))})
tree = xr.DataTree.from_dict({"coarse": ds, "fine": ds * 2})

Scale(10.0)(ds)    # Dataset in → Dataset out
Scale(10.0)(tree)  # DataTree in → DataTree out, each leaf scaled
```

Multi-input operators work too: `xarray` enforces the structural-match
requirement between DataTree arguments, and mixing a `DataTree` with a
plain `Dataset` raises rather than broadcasting silently.

## Signature

`Signature` captures *what shape an operator expects or produces*
without holding data — a frozen mapping of dim name → size (`None`
marks a symbolically unknown size) plus an optional dtype tag.
Operators implement `compute_output_signature` so `Sequential.summary()`
and `Graph.summary()` can render structural tables without executing
the pipeline.

```python
from xrcore import Signature

sig = Signature({"time": 365, "lat": 181, "lon": 360}, dtype="float32")
sig.format()                      # '(time=365, lat=181, lon=360); dtype=float32'
sig.replace_dims({"time": None})  # size unknown after this op
```

## API reference

- [`Operator` and composition](../api/composition.md)

The full design rationale (dispatch order, empty-leaf handling, the
multi-input structural-match rule) lives in
[Design → xarray-native primitives](../design/xarray-native-primitives.md).
