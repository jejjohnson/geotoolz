---
status: complete
version: 0.2.0
---

# Xarray-Native Primitives — Two-Layer Public Contract

> **Implementation status.** PR α (Core + DataTree dispatch) is
> implemented in `src/xrtoolz/_operator.py` and re-exported as
> `xrtoolz.Operator`. The architecture differs slightly from the
> original sketch: since the composition core lives in carrier-agnostic
> `pipekit`, the DataTree branch lives on an xarray-aware
> `xrtoolz.Operator` subclass rather than on `pipekit.Operator`
> itself. `pipekit.Sequential` and `pipekit.Graph` thread the
> resulting `DataTree`s through without changes — they are
> carrier-agnostic.
>
> **PR β is now complete.** Every Layer-0 primitive in
> `metrics/_src/{pixel,spectral,composite,structural,segmented_psd,distributional,probabilistic}.py`
> and `interpolate/_src/{smooth,resample}.py` takes positional
> `DataArray` inputs and keyword-only configuration; the Dataset
> selection / per-variable loop moved into the matching Layer-1
> `Operator._apply`. `metrics/_src/spectral.evaluate_by_frequency_band`
> intentionally stays Dataset-flavoured (it composes an inner
> Operator), and `metrics/_src/dm.dm_test` stays array-positional
> (no Dataset to flip). Operator constructor signatures are
> unchanged.
>
> **PR γ is now complete.** The remaining Dataset-flavoured Layer-0
> primitives are flipped to DataArray-positional with keyword-only
> configuration:
>
> - `transforms/_src/coord_remap.py`: `remap_axis(da, *, source_dim,
>   target_coords, ...)` and `to_phase(da, *, time_dim, period, n_bins)`
>   are now DataArray-in / DataArray-out; the per-variable Dataset loop
>   and non-numeric guard moved into `RemapAxis._apply` and
>   `ToPhase._apply` in `interpolate/operators.py`. The vertical presets
>   (`ToSigma`, `FromSigma`, `ToIsopycnal`, `ToPressureLevels`,
>   `ToHeight`) inherit `RemapAxis._apply` and pick up the loop for
>   free.
> - `transforms/_src/encoders/coord_time.py`: `time_rescale`,
>   `time_unrescale`, `encode_time_ordinal` are DataArray-in /
>   DataArray-out; `encode_time_cyclical` is DataArray-in /
>   Dataset-out (multi-output primitive shape per the design doc
>   "structured multi-output" pattern). The operators
>   (`TimeRescale`, `TimeUnrescale`, `EncodeTimeCyclical`,
>   `EncodeTimeOrdinal`) select `ds[self.time]` before calling the
>   primitive and merge the result back into the Dataset.
> - `transforms/_src/fourier.py`: `rotary_spectrum(u, v, *, dim,
>   avg_dims)` takes the two velocity DataArrays positionally; a new
>   Layer-1 `RotarySpectrum(u, v, dim)` operator wraps it.
> - `geo/_src/along_track.py`: `bandpass_wavelength(da, *, dim, ..., lon,
>   lat, ...)` is DataArray-positional, with `lon` and `lat` now passed
>   as `DataArray | None` (used for spacing inference) rather than as
>   variable-name strings. The `BandpassWavelength` operator keeps the
>   `lon: str` / `lat: str` constructor and does the Dataset selection
>   (including per-variable loop, non-numeric pass-through, and the
>   misspelled-`dim` guard) in `_apply`.
>
> Deliberate non-flips in PR γ:
>
> - The other Fourier primitives in `transforms/_src/fourier.py`
>   (`power_spectrum`, `cross_spectrum`, `coherence`, `stft`,
>   `ke_spectral_flux`, `enstrophy_spectral_flux`, `integral_scale`,
>   `fit_spectral_slope`, `compensated_spectrum`,
>   `drop_negative_frequencies`) were already DataArray-positional and
>   needed no change.
> - The DCT / DST and wavelet primitives in
>   `transforms/_src/{dct,wavelet}.py` were already DataArray-first
>   and are unchanged.
> - The basis encoders in `transforms/_src/encoders/basis.py`
>   (`cyclical_encode`, `fourier_features`, `random_fourier_features`,
>   `positional_encoding`) consume raw `ndarray` for backward
>   compatibility with their Operator wrappers and are out of scope —
>   the Operator layer already does the DataArray plumbing.
> - The transforms `morphology` / `decompose` / `sklearn_op` modules
>   are stateful-estimator or coord/attr utilities (not pure Layer-0
>   primitives) and don't fit the flip contract.
>
> Operator constructor signatures across all three PRs are
> unchanged.

A follow-on refactor to D11 (revised). Flips every primitive in the
package to an xarray-only public signature, keeping user-authored
numpy implementations as private helpers inside the same module.
Lifts DataTree support to the Operator base class so it falls out for
every existing subclass with no per-operator code.

## Motivation

After PR #200, the package has two public tiers (Layer 0 xarray,
Layer 1 Operator) but Layer 0 signatures are still inconsistent:

- `metrics.rmse(ds_pred, ds_ref, variable, dims)` — Dataset + selector kwargs
- `interpolate.gaussian_smooth(ds, dim, sigma)` — Dataset, loops over every
  numeric variable internally
- `geo._src.wavelet1d.cwt1d(da, dim, ...)` — DataArray

Three different shapes for the same conceptual position in the stack.
Users calling primitives directly hit a different ergonomic story per
module, and the "Dataset + variable selector" idiom forces every
primitive to grow `variable_var: str` plumbing that the Operator
layer is already paid to handle.

Separately, DataTree is becoming the canonical xarray container for
multi-group / multi-resolution datasets, and the library has no story
for it.

## Contract

| Layer | Accepts | Returns | Selectors |
|---|---|---|---|
| **Primitive** (private `_src/`) | `DataArray` (one or more positional) | `DataArray`, or `Dataset` when the result is a structured multi-field output (e.g. `geostrophic_velocity(ssh, lat)` → `Dataset(u_g, v_g)`) | None — the caller passes the field |
| **Operator** (public Layer 1) | `Dataset`, or `DataTree` (auto-mapped), or two of either for multi-input | Same container shape as input; reductions may narrow to `DataArray` / scalar; terminal viz returns `matplotlib.Figure` / `Axes` (D10) | Constructor kwargs (`variable=`, `u_var=`, `v_var=`, …) |

Hard rules:

- **No** `np.ndarray` in any public signature.
- **No** `Dataset` in primitive *input* signatures. Multi-field input is
  multiple positional `DataArray`s.
- **Private numpy helpers OK** when the math is genuinely easier as
  `(ndarray, axis)` (e.g. NaN-aware reductions, FFT shape juggling).
  They live as underscore-prefixed functions in the same primitive's
  `_src/` module — no separate `_*_kernels.py` sibling files. Bridge
  with `xr.apply_ufunc`.

## Primitive shapes — common patterns

### 1. Single-input, shape-preserving along a dim

Smoothers, filters, normalizations.

```python
def gaussian_smooth(da: DataArray, *, dim: str, sigma: float) -> DataArray:
    """Same shape; smoothed along `dim`."""
    def _kernel(arr):
        return scipy.ndimage.gaussian_filter1d(arr, sigma, axis=-1)
    return xr.apply_ufunc(
        _kernel, da,
        input_core_dims=[[dim]], output_core_dims=[[dim]],
        vectorize=True, keep_attrs=True,
    )
```

### 2. Single-input, reducing along a dim

Custom reductions where xarray's built-in `.mean(dim=...)` etc. is
insufficient (e.g. NaN policies, weighted reductions, robust stats).

```python
def variance_skipna(da: DataArray, *, dim: str | Sequence[str]) -> DataArray:
    """Drops `dim`."""
    core = [dim] if isinstance(dim, str) else list(dim)
    def _kernel(arr):
        return np.nanvar(arr, axis=tuple(range(-len(core), 0)))
    return xr.apply_ufunc(_kernel, da, input_core_dims=[core])
```

### 3. Single-input, dim-replacing (coord remap, regrid, refine)

```python
def remap_axis(
    da: DataArray, *,
    source_dim: str,
    target_coords: DataArray,
    method: str = "linear",
) -> DataArray:
    """`source_dim` replaced by the dim/length of `target_coords`."""
    ...
```

### 4. Single-input, spectral (dim → freq_dim)

```python
def power_spectrum(da: DataArray, *, dim: str) -> DataArray:
    """Returns DataArray with `dim` replaced by `freq_<dim>`."""
    ...
```

### 5. Single-input, structured multi-output (Dataset out)

When one input field naturally produces several output fields —
gradients, geostrophy from SSH, decompositions.

```python
def gradient(da: DataArray, *, dims: tuple[str, ...]) -> Dataset:
    """One partial-derivative field per requested dim."""
    return Dataset({f"d{da.name}_d{d}": da.differentiate(d) for d in dims})

def geostrophic_velocity(ssh: DataArray, lat: DataArray) -> Dataset:
    """SSH → (u_g, v_g)."""
    return Dataset({"u_g": ..., "v_g": ...})
```

### 6. Multi-input (paired), reducing (metrics)

```python
def rmse(pred: DataArray, ref: DataArray, *, dim: str | Sequence[str]) -> DataArray:
    diff_sq = (pred - ref) ** 2
    return np.sqrt(_nanmean_via_apply_ufunc(diff_sq, dim))
```

### 7. Multi-input (independent fields), derived elementwise

```python
def kinetic_energy(u: DataArray, v: DataArray) -> DataArray:
    return 0.5 * (u**2 + v**2)
```

### 8. Multi-input with auxiliary coords/fields

```python
def relative_vorticity(
    u: DataArray, v: DataArray, lat: DataArray, *,
    lon_dim: str = "lon", lat_dim: str = "lat",
) -> DataArray:
    """Curl on a sphere. lat is broadcast over u/v."""
    ...
```

## Operator shapes — common patterns

Operators wrap primitives. Constructor stores config + variable
selectors; `_apply` selects fields from the Dataset, calls primitive,
reassembles into Dataset (or returns DataArray / Figure for the
reducing / terminal cases).

### 1. Shape-preserving op on one variable

```python
class GaussianSmooth(Operator):
    def __init__(self, variable: str, *, dim: str, sigma: float):
        self.variable, self.dim, self.sigma = variable, dim, sigma
    def _apply(self, ds: Dataset) -> Dataset:
        smoothed = gaussian_smooth(ds[self.variable], dim=self.dim, sigma=self.sigma)
        return ds.assign({self.variable: smoothed})
```

### 2. Multi-input metric (reduces to DataArray)

```python
class RMSE(Operator):
    def __init__(self, variable: str, *, dim: str | Sequence[str]):
        self.variable, self.dim = variable, dim
    def _apply(self, ds_pred: Dataset, ds_ref: Dataset) -> DataArray:
        return rmse(ds_pred[self.variable], ds_ref[self.variable], dim=self.dim)
```

### 3. Derived field from multiple input variables (kinematics)

```python
class KineticEnergy(Operator):
    def __init__(self, *, u_var: str = "u", v_var: str = "v", output_var: str = "ke"):
        self.u_var, self.v_var, self.output_var = u_var, v_var, output_var
    def _apply(self, ds: Dataset) -> Dataset:
        ke = kinetic_energy(ds[self.u_var], ds[self.v_var])
        return ds.assign({self.output_var: ke})
```

### 4. Structured multi-output

```python
class GeostrophicVelocity(Operator):
    def __init__(self, *, ssh_var: str = "ssh", lat_var: str = "lat"):
        self.ssh_var, self.lat_var = ssh_var, lat_var
    def _apply(self, ds: Dataset) -> Dataset:
        derived = geostrophic_velocity(ds[self.ssh_var], ds[self.lat_var])
        return xr.merge([ds, derived])
```

### 5. Operator-only modules (no primitive layer)

`validation`, `subset`, `masks`, `crs` — coord/attr work; no array
math, no primitive layer:

```python
class SubsetBBox(Operator):
    def __init__(self, *, lon_bnds: tuple[float, float], lat_bnds: tuple[float, float]):
        ...
    def _apply(self, ds: Dataset) -> Dataset:
        return ds.sel(lon=slice(*self.lon_bnds), lat=slice(*self.lat_bnds))
```

## DataTree polymorphism — `Operator.__call__` dispatch

`Operator.__call__` grows one branch. Existing `Node`-detection
(symbolic graph) stays; new branch maps over a DataTree:

```python
def __call__(self, *args, **kwargs):
    if any(isinstance(a, Node) for a in args):
        return Node(operator=self, parents=args)
    if any(isinstance(a, DataTree) for a in args):
        return xr.map_over_subtree(self._apply)(*args, **kwargs)
    return self._apply(*args, **kwargs)
```

Consequence: every operator gets DataTree support for free.

```python
op = GaussianSmooth("ssh", dim="time", sigma=3)
op(ds)                       # → Dataset
op(dt)                       # → DataTree, each leaf smoothed independently

metric = RMSE("ssh", dim="time")
metric(ds_pred, ds_ref)      # → DataArray
metric(dt_pred, dt_ref)      # → DataTree of DataArrays, leaf-by-leaf

pipeline = Sequential([Regrid(...), GaussianSmooth("ssh", dim="time", sigma=3)])
pipeline(dt)                 # → DataTree, full pipeline applied per leaf
```

Multi-input ops require both inputs to be DataTrees with matching
structure when used in tree mode — `xr.map_over_subtree` enforces
this.

## What goes away

- Every `variable: str` / `var_ref: str` / `*_var: str` selector kwarg
  on a Layer 0 function. Moves to the Operator constructor.
- Every "loop over `ds.data_vars`, apply to each numeric var" helper
  inside a primitive (e.g. `_apply_along_dim` in
  `interpolate/_src/smooth.py`). The Operator handles the loop now.
- The `_*_kernels.py` files introduced by #200. The numpy code gets
  inlined into the primitive that uses it as an underscore-prefixed
  module-level helper. A numpy helper shared by multiple primitives
  stays as a module-level underscore function in one of them and is
  imported.

## What stays untouched

- Every numpy implementation — algorithms, NaN handling, edge cases.
  Preserved verbatim, just folded into the primitive.
- The Operator / Sequential / Graph composition story.
- All public Operator subclass names and constructor signatures.
- Test files — only their imports / argument shapes shift.

## Module migration map

| Module | Today | After |
|---|---|---|
| `core/operator.py` | Dispatches `Node` only | Adds `DataTree` branch (one new `if`) |
| `core/sequential.py`, `core/graph.py` | Type-check Dataset returns at non-terminal steps | Type-check `Dataset | DataTree` at non-terminal steps |
| `metrics/_src/pixel.py` | `rmse(ds_pred, ds_ref, variable, dims)` | `rmse(da_pred, da_ref, *, dim)` |
| `metrics/_src/segmented_psd.py` | `along_track_psd_score(ds, var_ref, var_pred, ...)` | `along_track_psd_score(ref, pred, ...)` |
| `metrics/_src/{spectral,multiscale,distributional,structural,physical,probabilistic,residuals,forecast,masked,composite,dm,leaderboard,object,lagrangian}.py` | Mixed Dataset+selectors | DataArray-positional |
| `interpolate/_src/smooth.py` | `gaussian_smooth(ds, dim, sigma)` (loops vars) | `gaussian_smooth(da, dim=, sigma=)`; operator does the Dataset loop |
| `interpolate/_src/{grid_to_grid,gap_fill,resample,coord_remap,binning,points_to_grid,grid_to_points,knn,downscale}.py` | Mixed Dataset/DataArray | DataArray (multi-DataArray where appropriate) |
| `transforms/_src/{fourier,decompose,coord_remap,morphology,encoders,*}.py` | Mostly DataArray already | Audit & normalize |
| `geo/_src/{validation,subset,masks,crs}.py` | Operator-only (Dataset coord/attr work) | **Unchanged** — operator-only modules per the contract |
| `geo/_src/wavelet1d.py` and other single-field geo primitives | DataArray already | Unchanged |
| `kinematics/_src/*` (when written, D9) | N/A | Multi-DataArray positional (`kinetic_energy(u, v)`, `geostrophic_velocity(ssh, lat)`) |
| Every `<module>/operators.py` | Wraps Layer 0, selects from Dataset | Same role — selects vars and hands DataArrays to the primitive. Most diffs are 1–2 lines per class |

## Phasing — three sequential PRs

The full migration is too big for one PR. Suggested order:

### PR α — Core + DataTree dispatch

Adds DataTree polymorphism with no primitive changes. Operators still
call current Dataset-shaped Layer 0; they just gain free DataTree
support via the base-class dispatch.

- `core/operator.py`: +~15 LOC (one new `if`).
- `core/sequential.py`, `core/graph.py`: ~5 LOC each (broaden the
  non-terminal type check to `Dataset | DataTree`).
- New `tests/test_core_datatree.py`: ~80 LOC (round-trip a smoother
  and a metric through a 2-leaf DataTree).
- Doc update: this design doc + a paragraph in `architecture.md`.

**Total: ~100–150 LOC.** Low risk — surgical dispatch addition, no
signature changes.

### PR β — Metrics + interpolate primitive flip

The two biggest modules. Inlines kernels, flips signatures to
DataArray-positional, moves Dataset selection into the operators.

- metrics: ~16 `_src/*.py` files + their operator wrappers + tests.
- interpolate: ~10 `_src/*.py` files + their operator wrappers + tests.
- Removes the `_*_kernels.py` files introduced by #200 by folding
  their numpy back into the primitive.

**Total: ~1500–2500 LOC changed** (large share of that is deletions
of plumbing — `_apply_along_dim`, selector kwargs, kernel files).
Medium risk: many signature flips, tests need re-pointing. Numerical
behavior unchanged.

### PR γ — Transforms + geo single-field + cleanup

- transforms: ~6 `_src/*.py` files + operators.
- Single-field geo primitives (`wavelet1d`, encoder helpers): audit.
- Docstring + design-doc sweep.

**Total: ~500–900 LOC.** Low–medium risk — smaller surface than β.

## Code change estimate — totals

| PR | Files touched | LOC delta (gross) | Risk |
|---|---|---|---|
| α — Core + DataTree | 3–4 src + 1 new test | **~100–150** | Low |
| β — Metrics + interpolate | ~50 files | **~1500–2500** | Medium |
| γ — Transforms + geo + cleanup | ~15 files | **~500–900** | Low–medium |
| **Total** | **~65 files** | **~2100–3550 LOC** over 3 PRs | — |

For comparison, PR #200 was ~1200 LOC. Each individual PR in this
plan is in the same ballpark as recent work on the repo.

A meaningful share of the diff is **deletions** — the
`_apply_along_dim` Dataset-looping helpers in each `_src/*.py` go
away (operators handle the loop now), and the `_*_kernels.py` files
from #200 collapse into their primitive.

## Breaking changes

This is a breaking change for anyone calling Layer 0 functions
directly:

- `metrics.rmse(ds_pred, ds_ref, variable="ssh", dims="time")` →
  `metrics.rmse(ds_pred["ssh"], ds_ref["ssh"], dim="time")`
- `interpolate.gaussian_smooth(ds, dim="time", sigma=2.0)`
  (Dataset-wide) → `interpolate.gaussian_smooth(ds["x"], dim="time",
  sigma=2.0)` (single var); use the `GaussianSmooth()` operator if
  you want the Dataset-wide loop.

Operator-level API (`RMSE(...)`, `GaussianSmooth(...)`,
`Sequential([...])`) is unchanged — anyone going through Operators
sees no break.

Per current project status ("noone is using my lib except me"), the
breaking-change cost is contained. No deprecation shims planned,
consistent with prior pre-1.0 churn (e.g. F3 module consolidation in
#98, public array tier removal in #200).

## Relationship to existing decisions

- **Revises D11** (which already stepped down from three tiers to
  two). D11 stated the *public surface* is two-tier; this doc
  tightens what each tier *accepts and returns* so the two layers are
  contract-clean across every module.
- **Compatible with D1, D2, D3, D6, D10** — Operator-as-callable,
  split-object stateful pattern, dual-mode `__call__`, dict-in/out
  Graph, viz-as-Operator: all unchanged.
- **D4** (no framework dep for `ModelOp`) and **D5** (numpy/scipy for
  compute) are unchanged — user-authored numpy implementations remain
  the compute core; the refactor only changes how they're plumbed.
- **D9** (kinematics submodule) — kinematics is greenfield and will
  be written in the new style from day one. No migration cost for
  this module.
- **D12** (`interpolate` unified resampling module) — unchanged
  structure; only the primitive signatures inside `interpolate/_src/`
  shift.

## Open questions for sign-off

1. Confirm the contract (top table + hard rules) matches expectations.
2. Confirm the operator pattern of "augment the Dataset in place"
   (`ds.assign({var: result})` or `xr.merge([ds, derived])`) is the
   intended idiom for derived-field operators, rather than "operator
   returns only the new variable, caller merges."
3. Confirm phasing α → β → γ in that order.
4. Confirm DataTree dispatch goes in `Operator.__call__` (Option 1
   from prior discussion) rather than an explicit `MapOverTree`
   combinator.
