# OB-1.1 — `xrpatcher` integration (`PatchDataset`, `PatchedInference`)

**Source survey item:** [oceanbench-survey.md §A.4.1](oceanbench-survey.md)
**Status:** proposed
**Maps to upstream:** [`jejjohnson/xrpatcher`](https://github.com/jejjohnson/xrpatcher)
(MIT, ~470 LOC, last pushed 2026-03). Stand-alone evolution of the
`oceanbench/_src/datasets/base.py:XRDABatcher` referenced in the
oceanbench survey.

---

## 1. Motivation

ML-based ocean diagnostics (super-resolution, infilling, sub-mesoscale
emulation, neural SSH mapping) almost always run on **bounded spatial
tiles** rather than the full domain — both because GPU memory caps
inference at modest patch sizes and because trained CNNs typically have
fixed receptive fields. The standard pattern is:

```text
1. Slice the input Dataset into overlapping (lat, lon) patches.
2. Run the model on each patch.
3. Stitch the outputs back into a global grid, blending overlaps.
```

xrtoolz currently ships `ModelOp` for **per-pixel** sklearn-style
models (`(N, F)` flatten → `model.predict(x)` → reshape) but has no
tile-wise inference path. The
[`jejjohnson/xrpatcher`](https://github.com/jejjohnson/xrpatcher)
library — single class `XRDAPatcher`, MIT, the user's own repo — already
solves the patch-and-reconstruct problem cleanly:

- Generates patches by `(patch_size, stride)` per dim.
- `__getitem__(i)` returns one `xr.DataArray` patch.
- `reconstruct(items, dims_labels=None, weight=None)` stitches a list
  of equal-shaped patches back into a single DataArray with weighted
  overlap blending.
- Adds runtime caching (`cache=True` / `preload=True`),
  domain-limit pre-selection (`domain_limits=`), and
  full-scan validation (`check_full_scan=`).

This issue takes `xrpatcher` as a hard dependency and adds two thin
Layer-1 Operators (`PatchDataset`, `PatchedInference`) plus a
re-export of `XRDAPatcher`, so patcher-driven tile inference composes
with `Sequential` and the rest of xrtoolz.

## 2. User stories

### 2.1 Tile-wise inference of a trained CNN (primary)

> *I have a trained PyTorch SSH-super-resolution model and want to run
> it on a `(time, lat, lon)` Dataset, tile-by-tile, with overlap
> blending.*

```python
import xarray as xr
from xrtoolz.core import Sequential
from xrtoolz.patcher import PatchDataset, PatchedInference

ds = xr.open_dataset("ssh_lowres.nc")             # (time, lat, lon)

def my_model(da):
    """patch in (lat, lon) → patch out (lat, lon), via torch."""
    arr = torch.from_numpy(da.values)
    with torch.no_grad():
        out = trained_cnn(arr.unsqueeze(0).unsqueeze(0))[0, 0].numpy()
    return out                                     # numpy array, same shape

pipeline = Sequential([
    PatchDataset(
        patches={"lat": 64, "lon": 64},
        strides={"lat": 32, "lon": 32},            # 50% overlap
        var="ssh",
    ),
    PatchedInference(model=my_model),
])
ssh_hires = pipeline(ds)
```

### 2.2 Inspect or reuse the patcher

> *I want to look at the patcher's coordinate plan before running
> inference, and reuse it for two different model variants.*

```python
from xrtoolz.patcher import PatchDataset, PatchedInference

patcher = PatchDataset(
    patches={"lat": 64, "lon": 64},
    strides={"lat": 32, "lon": 32},
    var="ssh",
)(ds)

print(len(patcher))                                 # number of patches
coords_list = patcher.get_coords()                  # one coord-Dataset per patch

ssh_v1 = PatchedInference(model=model_v1)(patcher)
ssh_v2 = PatchedInference(model=model_v2)(patcher)
```

### 2.3 Cached patcher for repeat inference

```python
patcher = PatchDataset(
    patches={"lat": 64, "lon": 64},
    cache=True, preload=True,                       # patches loaded once + held
)(ds)
```

### 2.4 Direct DataArray input (no Dataset narrowing)

```python
patcher = PatchDataset(patches={"lat": 64, "lon": 64})(ds["ssh"])
```

`PatchDataset` accepts both `xr.Dataset` (auto-narrows via `var=` when
multi-var, errors if `var=None` and >1 data_var) and `xr.DataArray`
(passed through to `XRDAPatcher` directly).

### 2.5 Custom overlap weighting

> *I want a centred-Gaussian weight to suppress patch-edge artifacts
> when stitching.*

```python
import numpy as np
patch_size = (64, 64)
yy, xx = np.mgrid[:patch_size[0], :patch_size[1]]
cy, cx = (patch_size[0]-1)/2, (patch_size[1]-1)/2
sigma = patch_size[0] / 4
weight = np.exp(-((yy-cy)**2 + (xx-cx)**2) / (2*sigma**2))

inference = PatchedInference(
    model=my_model,
    reconstruct_kwargs={"weight": weight},
)
```

## 3. What we already have / what's missing

| Capability | Current | This proposal |
|---|---|---|
| Per-pixel sklearn-style ModelOp | [`inference/modelop.py:30`](../../src/xrtoolz/inference/modelop.py) — `(N,F)` flatten/reshape | unchanged (different paradigm) |
| `Operator` / `Sequential` plumbing | [`core/`](../../src/xrtoolz/core/) | reuse |
| `XRDAPatcher` (patch + reconstruct) | — | re-export from `xrpatcher` (hard dep) |
| Layer-1 Operator: build patcher | — | **add** `PatchDataset` |
| Layer-1 Operator: run model per patch + reconstruct | — | **add** `PatchedInference` |
| Random-shuffle training sampler | — | deferred (revisit later) |

## 4. Design

### 4.1 Why hard-dep `xrpatcher`, not vendor

- **`xrpatcher` is MIT, ~470 LOC, single dep (`tqdm`)**, recent activity (2026-03).
- The user owns both repos — release coordination is trivial.
- Other libraries can use `xrpatcher` independently. Vendoring would
  duplicate code and force version skew over time.
- `xrpatcher`'s API is small and stable enough to pin against (e.g.
  `xrpatcher>=0.x,<0.(x+1)`).

We do not vendor or soft-import — `xrpatcher` becomes a top-level
dependency in `pyproject.toml`.

### 4.2 Why two Operators, not a `ModelOp.patcher=` retrofit

The existing
[`ModelOp`](../../src/xrtoolz/inference/modelop.py#L30) is designed
around the per-pixel paradigm:

```text
(time, lat, lon, feature) → reshape to (time*lat*lon, feature)
                          → model.predict(x) → reshape back
```

Tile-based CNNs work the **opposite** way: they take spatial patches
`(C, H, W)` and return same-shape spatial outputs. Adding `patcher=`
to `ModelOp` would warp the existing reshape semantics and confuse the
SklearnModelOp / JaxModelOp subclasses that depend on them.

Cleaner: leave `ModelOp` untouched (no breaking change), add a sibling
abstraction in a new `xrtoolz.patcher` submodule. Users who want
per-pixel models on tiles can still compose:
`PatchedInference(model=ModelOp(rf_model))`.

### 4.3 `PatchDataset` Operator

```python
# src/xrtoolz/patcher/_src/operators.py
class PatchDataset(Operator):
    """Operator that constructs an XRDAPatcher from a Dataset/DataArray."""

    def __init__(
        self, *,
        patches: dict[str, int],
        strides: dict[str, int] | None = None,
        domain_limits: dict | None = None,
        check_full_scan: bool = False,
        cache: bool = False,
        preload: bool = False,
        var: str | None = None,
    ): ...

    def __call__(self, ds: xr.Dataset | xr.DataArray) -> XRDAPatcher: ...

    def get_config(self) -> dict[str, Any]: ...
    def __repr__(self) -> str: ...
```

**Behavior:**

1. If input is `xr.DataArray` → forward directly.
2. If input is `xr.Dataset`:
   - If `var is not None`: narrow to `ds[var]`.
   - If `var is None` and `len(ds.data_vars) == 1`: narrow to the
     single data_var.
   - Otherwise: raise `ValueError` listing available vars and the
     required `var=` kwarg.
3. Construct `XRDAPatcher(da, patches, strides, domain_limits,
   check_full_scan, cache, preload)`.
4. Return the patcher.

`get_config()` round-trips all constructor kwargs (including `var`).
The patcher itself is *not* part of config — it's the result, not
state.

### 4.4 `PatchedInference` Operator

```python
class PatchedInference(Operator):
    """Tile-wise inference: run a callable per patch, reconstruct via the patcher.

    Composes naturally with PatchDataset in a Sequential.
    """

    def __init__(
        self, *,
        model: Callable[[xr.DataArray], xr.DataArray | np.ndarray],
        reconstruct_kwargs: dict[str, Any] | None = None,
        progress: bool = True,
    ): ...

    def __call__(self, patcher: XRDAPatcher) -> xr.DataArray: ...

    def get_config(self) -> dict[str, Any]: ...
    def __repr__(self) -> str: ...
```

**Behavior:**

```python
def __call__(self, patcher: XRDAPatcher) -> xr.DataArray:
    iterator = tqdm(patcher) if self.progress else iter(patcher)
    outputs = [self.model(patch) for patch in iterator]
    return patcher.reconstruct(outputs, **(self.reconstruct_kwargs or {}))
```

**Design notes:**

- `model` is any callable `(xr.DataArray) -> xr.DataArray | np.ndarray`.
  No assumptions about backend (PyTorch, JAX, numpy, sklearn). The
  user wraps backend-specific tensor conversion inside the callable.
- `reconstruct_kwargs` forwarded to `XRDAPatcher.reconstruct` —
  supports `dims_labels=`, `weight=` (e.g. centred Gaussian for
  edge-artifact suppression).
- `progress=True` wraps the iteration with tqdm; cheap to disable.
- `get_config()` emits `{"model": "<callable>"}` flag indicating
  non-roundtrippable when the model isn't a serializable Operator.
  When `model` *is* an `Operator`, recurse into its `get_config()`.

### 4.5 Re-export wiring

```python
# src/xrtoolz/patcher/__init__.py
from xrpatcher import XRDAPatcher
from xrtoolz.patcher._src.operators import PatchDataset, PatchedInference

__all__ = ["XRDAPatcher", "PatchDataset", "PatchedInference"]
```

`XRDAPatcher` is re-exported from xrtoolz so users have a canonical
import path:

```python
from xrtoolz.patcher import XRDAPatcher       # canonical
from xrpatcher import XRDAPatcher              # also works (direct)
```

### 4.6 Composition examples

```python
# Single-tile inference + reconstruct
Sequential([
    PatchDataset(patches={"lat": 64, "lon": 64}, var="ssh"),
    PatchedInference(model=cnn),
])(ds)

# Inference of a per-pixel ModelOp wrapped over tiles (silly but works)
Sequential([
    PatchDataset(patches={"lat": 64, "lon": 64}),
    PatchedInference(model=lambda da: ModelOp(rf, feature_dim="time")(da)),
])(ds)

# Pre-process → patch → infer → post-process
Sequential([
    BandpassWavelength(...),                           # ODC-1.1
    PatchDataset(patches={"lat": 64, "lon": 64}),
    PatchedInference(model=cnn,
                     reconstruct_kwargs={"weight": gaussian_weight}),
    SpatialMapPanel(var="ssh", projection="gulf_stream"),
])(ds)
```

## 5. Library leverage

| Need | Library |
|---|---|
| Patch generation + `__getitem__` + `reconstruct` | `xrpatcher.XRDAPatcher` (hard dep) |
| Operator / Sequential plumbing | `xrtoolz.core` (reuse) |
| Progress bar | `tqdm` (already a transitive dep) |

**One new top-level dependency**: `xrpatcher`. Pinned to a stable
minor version range (e.g. `xrpatcher>=0.x,<0.(x+1)`) to insulate from
upstream API drift. xrpatcher's own dep is just `tqdm`.

## 6. Public API surface

```python
# Underlying class — re-export
xrtoolz.patcher.XRDAPatcher              # = xrpatcher.XRDAPatcher

# Layer-1 Operators
xrtoolz.patcher.PatchDataset(
    *, patches, strides=None, domain_limits=None,
    check_full_scan=False, cache=False, preload=False, var=None,
)
xrtoolz.patcher.PatchedInference(
    *, model, reconstruct_kwargs=None, progress=True,
)
```

## 7. Tests

| Test | Asserts |
|---|---|
| `PatchDataset` on DataArray returns `XRDAPatcher` with correct `__len__` | exact |
| `PatchDataset` on single-var Dataset auto-narrows | works without `var=` |
| `PatchDataset` on multi-var Dataset without `var=` | raises informative `ValueError` listing data_vars |
| `PatchDataset` with `var="ssh"` on multi-var Dataset | narrows correctly |
| `PatchDataset` with `domain_limits` | source pre-`sel`'d before patching |
| `PatchDataset` with `check_full_scan=True` on bad config | raises `IncompleteScanConfiguration` from xrpatcher |
| `PatchDataset` with `cache=True, preload=True` | patches cached after first iteration; second pass reuses |
| `PatchedInference` identity model + uniform weight | reconstruct matches input exactly |
| `PatchedInference` with overlap (stride < patch) | overlap regions averaged correctly |
| `PatchedInference` with custom Gaussian weight | weighted reconstruction matches manual weighted-average |
| `PatchedInference` model returning ndarray vs DataArray | both shapes handled |
| `PatchedInference` `progress=False` | no tqdm bar |
| Sequential composition (`PatchDataset` → `PatchedInference`) | end-to-end identity round-trip |
| `PatchDataset.get_config()` round-trip | reconstructed Operator produces identical `XRDAPatcher` |
| `xrpatcher` import | `from xrtoolz.patcher import XRDAPatcher` works |

Target: ~15 cases.

## 8. Out of scope

- **`RandomPatchSampler`** for ML training data loaders — deferred.
  xrpatcher iterates deterministically; shuffled sampling is a
  separate concern. Revisit when ML training pipelines land in xrtoolz.
- **`ModelOp.patcher=` retrofit** — declined; per-pixel and tile
  paradigms shouldn't share an API. `PatchedInference(model=ModelOp(...))`
  composition is available if anyone needs it.
- **Vendoring `xrpatcher`** — declined; hard dep is the right shape.
- **Backend-specific tile inference helpers** (`PatchedInference` with
  built-in PyTorch/JAX hooks) — keep the abstraction backend-agnostic;
  users wrap their own tensor conversion in the `model` callable.
- **Patcher visualization helpers** — could add a `PatcherCoordsPanel`
  later that overlays the patch grid on a `SpatialMapPanel`. Not
  blocking.

## 9. Effort

≈80 LOC implementation + ≈100 LOC tests. Single PR.

| Slice | LOC |
|---|---|
| `PatchDataset` Operator | 30 |
| `PatchedInference` Operator | 20 |
| `XRDAPatcher` re-export wiring | 5 |
| `pyproject.toml` dep + version pin | 2 |
| Tests | ~100 |
| Docs / re-exports | 15 |

## 10. Risks / open questions

1. **`xrpatcher` version pinning.** Pin to a minor range (e.g.
   `xrpatcher>=0.x,<0.(x+1)`) to insulate from upstream API drift.
   Document the pinning strategy in CONTRIBUTING / pyproject.toml
   comment.
2. **`var=` semantics on `PatchDataset`.**
   - `xr.DataArray` → forwarded.
   - Single-var `xr.Dataset` → auto-narrows.
   - Multi-var `xr.Dataset`, `var=None` → raise informative
     `ValueError`.
   - Multi-var `xr.Dataset`, `var="ssh"` → narrow.
3. **`get_config` round-trip when `model` is a Python lambda.** Emit
   `{"model": "<callable>"}` flag; non-roundtrippable. When `model` is
   an `Operator`, recurse — fully roundtrippable.
4. **Where the new code lives.** New top-level submodule
   `xrtoolz.patcher` (chosen for visibility). Alternative under
   `xrtoolz.geo.inference` rejected — patcher is a generic concept,
   not inference-specific.
5. **Tile-edge artifacts.** Default `weight=None` (uniform) gives
   simple averaging in overlap regions. Users who care about edge
   artifacts pass a Gaussian or windowed weight via
   `reconstruct_kwargs={"weight": ...}`. Document the recipe in the
   docstring.
6. **Memory pressure with `cache=True, preload=True`.** Pre-loaded
   patches are held in `_cache` dict; for very large patchers this
   can OOM. Document; recommend `cache=True, preload=False` for the
   common case where patches are dask-backed.
7. **`tqdm` discoverability.** `progress=True` wraps with tqdm; if
   tqdm is unavailable, fall back to plain iter with a one-time
   warning. (xrpatcher already imports tqdm at the module level, so
   this is currently a hard transitive dep — should be safe.)
