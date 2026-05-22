# ODC-2.2 — Laplacian (Gauss–Seidel) gap-fill + verify 3D regrid path

**Source survey item:** [ocean-data-challenges-survey.md §2.2](ocean-data-challenges-survey.md)
**Status:** proposed
**Maps to upstream:** `src/mod_regrid.py` from `2023_SSH_mapping_train_eNATL60_test_NATL60-`.

---

## 1. Motivation

The upstream `rec_regrid(ds_source, ds_target, field='ssh')` does two
things:

1. **3-D `(lon, lat, time)` regridding** via `pyinterp.Grid3D` +
   trivariate interpolation onto the target mesh.
2. **NaN extrapolation** via `pyinterp.fill.gauss_seidel` — iterative
   Laplacian relaxation that fills masked cells consistent with the
   surrounding finite values (harmonic infill).

Of those two:

- **(1) is already done** by our existing
  [`regrid_like(ds, target, dims=("lat","lon","time"), method="linear")`](../../src/xrtoolz/interpolate/_src/grid_to_grid.py).
  `xr.Dataset.interp` natively handles a `time` coord including
  `datetime64`. No new function needed — just a docstring nudge and a
  3-D regrid integration test for confidence.
- **(2) is missing.** Our `fillnan_*` family is convex-hull bounded
  (`fillnan_spatial` → `griddata`), globally smooth (`fillnan_rbf` →
  RBF), or temporal-only (`fillnan_temporal` → `interpolate_na`). None
  does the **iterative Laplacian relaxation** that fills NaN cells with
  the harmonic interpolant of the surrounding values. The module
  docstring acknowledges this gap:

  > *"these deliberately avoid heavy C++ dependencies (`pyinterp`,
  > `xesmf`); for Gauss-Seidel or ESMF-conservative regridding, use
  > those libraries directly."*

This issue closes the gap with a pure-numpy `fillnan_laplacian` —
Gauss–Seidel (optionally SOR) relaxation, ~30 LOC, no `pyinterp`. The
existing 3-D regrid path is verified by test rather than re-implemented.

## 2. User stories

### 2.1 Fill NaN holes in a regridded SSH field (primary)

> *I have an SSH map regridded onto a target grid. Land borders + sparse
> coverage left holes. I want them filled by harmonic infilling, like
> `pyinterp.fill.gauss_seidel`, without adding pyinterp as a dep.*

```python
import xarray as xr
from xrtoolz.interpolate import regrid_like, fillnan_laplacian

ds_target = regrid_like(
    ds_source, ds_target,
    dims=("lat", "lon", "time"),
    method="linear",
)
ds_filled = fillnan_laplacian(
    ds_target["ssh"],
    max_iter=1000, tol=1e-4, relaxation=1.0,
)
```

### 2.2 SOR for faster convergence on dense holes

```python
ds_filled = fillnan_laplacian(
    ds_target["ssh"],
    relaxation=1.5,    # SOR; ~2× fewer iterations on typical fields
    max_iter=500,
)
```

### 2.3 As a Layer-1 Operator inside a Sequential

```python
from xrtoolz.interpolate import RegridLike, FillNaNLaplacian
from xrtoolz.core import Sequential

regrid_and_fill = Sequential([
    RegridLike(target=ds_target, dims=("lat", "lon", "time")),
    FillNaNLaplacian(max_iter=1000, tol=1e-4),
])
```

## 3. What we already have / what's missing

| Capability | Current | This proposal |
|---|---|---|
| 3-D `(lon, lat, time)` regrid | [`regrid_like`](../../src/xrtoolz/interpolate/_src/grid_to_grid.py) — `xr.Dataset.interp` handles datetime64 | verify via test, document |
| Convex-hull `griddata` infill | [`fillnan_spatial`](../../src/xrtoolz/interpolate/_src/gap_fill.py) | unchanged |
| RBF infill | [`fillnan_rbf`](../../src/xrtoolz/interpolate/_src/gap_fill.py) | unchanged |
| Temporal `interpolate_na` | [`fillnan_temporal`](../../src/xrtoolz/interpolate/_src/gap_fill.py) | unchanged |
| Iterative Laplacian relaxation | — | **add** `fillnan_laplacian` |
| Operator wrapper | — | **add** `FillNaNLaplacian` |

## 4. Design

### 4.1 Algorithm

Given a 2-D field `u(i, j)` with a NaN mask `M`, find the harmonic
extension that satisfies `∇²u = 0` on `M` with Dirichlet BCs from the
finite cells. Discretise the 5-point Laplacian and apply Gauss–Seidel
relaxation in-place (with optional SOR over-relaxation):

```text
For each iteration:
  for each (i, j) in M (in raster order):
    u_avg = 0.25 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])
    u[i, j] = (1 - ω) * u[i, j] + ω * u_avg
  if max(|u - u_old|) < tol: break
```

True Gauss–Seidel (in-place updates) converges ~2× faster than Jacobi
(snapshot updates). SOR with ω ≈ 1.5 typically halves Gauss–Seidel
iteration count again on smooth fields.

**Initial guess**: `nanmean(u)` over the finite support — biased but
stable and converges to the harmonic solution regardless.

**Boundary condition**: Neumann (reflective). Domain edges get
zero-gradient mirrors — appropriate for ocean fields and avoids the
zero-Dirichlet artifacts that biased pyinterp output near domain
borders. Optionally `boundary="wrap"` for cylindrical longitude.

**Implementation note**: vectorise the sweep using array shifts
(`np.roll` + edge fixups) rather than Python-level `for (i, j)` loops.
This trades strict raster order for "checkerboard Jacobi-like in-place
on red/black sweep" behaviour, with virtually identical convergence on
smooth fields. For maximum-fidelity Gauss–Seidel users can pass
`mode="strict"` (Cython-free Python loop, slower).

### 4.2 Tier B — xarray primitive

```python
# src/xrtoolz/interpolate/_src/gap_fill.py — new function alongside fillnan_*
def fillnan_laplacian(
    da: xr.DataArray, *,
    max_iter: int = 1000,
    tol: float = 1e-4,
    relaxation: float = 1.0,            # 1.0 = Gauss-Seidel; 1.5 ≈ optimal SOR
    boundary: str = "reflect",          # "reflect" | "wrap"
    lon: str = "lon",
    lat: str = "lat",
) -> xr.DataArray:
    """Fill NaN cells via iterative harmonic relaxation (Gauss-Seidel / SOR).

    Solves ∇²u = 0 on the masked region with Dirichlet BCs from the
    finite cells. Stops when max(|u_new − u_old|) < ``tol`` or after
    ``max_iter`` iterations, whichever comes first.

    Operates slice-by-slice along any leading dims via xr.apply_ufunc.
    All-NaN and all-finite slices pass through unchanged.

    Parameters
    ----------
    da
        DataArray with at least ``lon`` and ``lat`` dims.
    max_iter
        Iteration cap. Default 1000.
    tol
        Convergence threshold on `max(|Δu|)` per iteration. Default 1e-4.
    relaxation
        Successive over-relaxation factor ω. ω=1.0 → vanilla
        Gauss–Seidel; 1.0<ω<2.0 → SOR (faster but can diverge for
        pathological masks). Default 1.0.
    boundary
        ``"reflect"`` (Neumann, default) or ``"wrap"`` (cylindrical).
    lon, lat
        Coordinate names.

    Returns
    -------
    xr.DataArray
        Same-shape array with NaN cells filled.
    """
```

~35 LOC.

### 4.3 Layer-1 Operator

```python
# src/xrtoolz/interpolate/operators.py
class FillNaNLaplacian(Operator):
    """Iterative Laplacian gap-fill operator."""

    def __init__(self, *,
                 max_iter: int = 1000,
                 tol: float = 1e-4,
                 relaxation: float = 1.0,
                 boundary: str = "reflect",
                 lon: str = "lon",
                 lat: str = "lat"): ...

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        return ds.map(
            lambda da: fillnan_laplacian(da, **self._kw)
            if {self.lat, self.lon} <= set(da.dims) else da
        )

    def get_config(self) -> dict[str, Any]: ...
```

Standard pattern. Variables that don't carry both `lat` and `lon`
pass through.

### 4.4 3-D regrid: verified, not re-implemented

`regrid_like(dims=("lat","lon","time"))` already does what the upstream
`pyinterp.trivariate` step does. The required test fixtures:

```python
def test_regrid_like_3d_with_datetime64_time():
    src = make_synthetic_field(dims=("time", "lat", "lon"))
    tgt = make_target_grid(diff_in_all_three_dims=True, datetime64=True)
    out = regrid_like(src, tgt, dims=("lat", "lon", "time"), method="linear")
    assert_allclose(out, expected_analytic_on_target)
```

Plus a one-line docstring update on `regrid_like` noting the 3-D
pattern explicitly.

### 4.5 Module docstring nudge

Soften the [`gap_fill.py`](../../src/xrtoolz/interpolate/_src/gap_fill.py)
header — replace:

> *"for Gauss-Seidel or ESMF-conservative regridding, use those
> libraries directly"*

with:

> *"for ESMF-conservative regridding use `xesmf` directly. For
> Gauss-Seidel-style harmonic infill see :func:`fillnan_laplacian`."*

## 5. Library leverage

| Need | Library |
|---|---|
| Array shifts / boundary handling | `numpy.roll` + edge slices |
| Slice-by-slice over leading dims | `xarray.apply_ufunc(vectorize=True)` (matches existing `fillnan_*` pattern) |
| Datetime64 interp on time axis | `xr.Dataset.interp` (built-in) |

No new dependencies. The implementation is pure numpy.

## 6. Public API surface

```python
xrtoolz.interpolate.fillnan_laplacian(da, *, max_iter, tol, relaxation,
                                       boundary, lon, lat)
xrtoolz.interpolate.FillNaNLaplacian(...)
```

Re-exported from `xrtoolz.interpolate.__init__`.

## 7. Tests

| Test | Asserts |
|---|---|
| Analytic harmonic field with a hole | Infilled values match analytic within tol |
| All-NaN slice | Returns all-NaN unchanged |
| All-finite slice | No change |
| Convergence: SOR ω=1.5 | Fewer iterations than ω=1.0 on the same field |
| `max_iter` cap | Honoured even when tol not met |
| `tol` early stop | Iteration count drops as tol loosens |
| Neumann boundary | Corner NaN filled via 2-neighbor reflection |
| `boundary="wrap"` | Longitude-edge NaN filled across 0/360 seam |
| Multi-leading-dim DataArray `(time, lat, lon)` | Iterates per slice independently |
| Operator round-trip via `get_config` | Reconstructed Operator produces identical output |
| 3-D `regrid_like(dims=("lat","lon","time"))` with datetime64 | Reproduces analytic on a different target grid |

Target: ~11 cases.

## 8. Out of scope

- **`fillnan_biharmonic`** (skimage `inpaint_biharmonic`) — biharmonic
  ≠ harmonic; different physics; not what the upstream uses. Add later
  if a user asks.
- **`regrid_with_fill` convenience** — users compose
  `Sequential([RegridLike(...), FillNaNLaplacian(...)])`. No new
  abstraction.
- **Tier A array-kernel split** — existing `fillnan_*` family is
  xarray-only via `apply_ufunc`. Match the convention; don't fragment.
- **`pyinterp` backend dispatch** — declined; pure numpy is sufficient
  and matches the explicit "no heavy C++ deps" stance of the gap_fill
  module.
- **3-D Laplacian fill** (over `time` too) — uncommon for SSH; current
  proposal is per-2-D slice. Add in a follow-up if needed.
- **ESMF conservative regridding** — separate concern;
  `xesmf` if a user needs it.

## 9. Effort

≈55 LOC implementation + ≈80 LOC tests. Single PR.

| Slice | LOC |
|---|---|
| `fillnan_laplacian` | 35 |
| `FillNaNLaplacian` operator | 20 |
| 3-D regrid integration test + docstring nudge | 5 + tests |
| Tests | ~80 |
| Docs / re-exports | 10 |

## 10. Risks / open questions

1. **Naming.** `fillnan_laplacian` vs `fillnan_gauss_seidel` vs
   `fillnan_relaxation`. **Recommend `fillnan_laplacian`** —
   describes the math (harmonic infill via Laplace's equation) and
   matches the existing family style (`fillnan_rbf`, `fillnan_spatial`).
2. **Default `relaxation`.** ω=1.0 (vanilla Gauss–Seidel) is robust;
   ω=1.5 (SOR) is faster but can diverge on pathological masks.
   **Default to 1.0**; document SOR for advanced users.
3. **Default boundary.** Neumann (`"reflect"`) recommended for ocean
   fields; avoids zero-Dirichlet artifacts. Wrap mode available for
   cylindrical longitude.
4. **Strict-Gauss–Seidel vs vectorised red/black sweep.** Vectorised
   default for speed. Optional `mode="strict"` (Python loop) for
   exact reference behaviour — skip in v1 unless needed.
5. **Dask compatibility.** `apply_ufunc(vectorize=True)` works on
   dask-backed inputs but materialises per slice. For multi-time-step
   fields this is fine; for huge spatial fields users may want
   `parallel=True`. Document; add as kwarg if needed.
6. **Initial guess.** `nanmean(u)` is biased but stable. Linear-from-
   `griddata` would converge faster but couples gap_fill modules.
   **Recommend `nanmean`** for v1 simplicity.
