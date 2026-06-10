# sklearn integration — `SklearnOp`, NaN masking, accessor, patch-wise composition

**Status:** proposed
**Scope:** consolidate the sklearn surface across `xrtoolz.utils`,
`xrtoolz.transforms`, and `xrtoolz.inference` and close the remaining
gaps so the bridge becomes a first-class composable layer.

---

## 1. Motivation

`xrtoolz` already ships a working xarray ↔ sklearn bridge:

| Layer | What ships today | Path |
|-------|------------------|------|
| 0 | `XarrayEstimator` — stack → delegate → unstack, attribute proxy, NaN `"propagate"`/`"raise"` | [src/xrtoolz/utils/_src/sklearn_wrap.py](https://github.com/jejjohnson/xrtoolz/blob/main/src/xrtoolz/utils/_src/sklearn_wrap.py) |
| 1 (presets) | `pca_op`, `eof_op`, `ica_op`, `nmf_op`, `kmeans_op` returning fitted-able `XarrayEstimator`s | [src/xrtoolz/transforms/_src/decompose.py](https://github.com/jejjohnson/xrtoolz/blob/main/src/xrtoolz/transforms/_src/decompose.py) |
| 2 (inference) | `SklearnModelOp` — duck-typed wrapper for a *fitted* model, used in `Graph` DAGs | [src/xrtoolz/inference/modelop.py](https://github.com/jejjohnson/xrtoolz/blob/main/src/xrtoolz/inference/modelop.py) |

The gap is on the **composition** side. `XarrayEstimator` is an sklearn
`BaseEstimator`, not an `xrtoolz.Operator` — so a fitted PCA cannot be
dropped into a `Sequential([validate, regrid, pca, classify])` chain
without a thin shim each time. Three other pieces are missing: a
`"mask"` NaN policy for land-masked grids, a `da.sklearn` accessor for
ergonomic one-liners, and an explicit recipe for patch-wise composition
with `xrpatcher`.

This doc proposes the smallest surface that closes those gaps without
disturbing the bridge that already exists.

---

## 2. Gap inventory

| Capability | `xrtoolz` today | gap |
|---|---|---|
| Stack → delegate → unstack | ✓ | — |
| Attribute proxy (`coef_`, `components_`, …) | ✓ | — |
| `Dataset` input → column-concat | ✓ | — |
| `inverse_transform` re-grids to training feature space | ✓ | — |
| NaN policy `"propagate"` / `"raise"` | ✓ | — |
| NaN policy `"mask"` (drop → fit → re-insert) | ✗ | **G1** |
| `da.sklearn` / `ds.sklearn` accessor | ✗ | **G2** |
| `xrpatcher` composition pattern | undocumented | **G3** |
| `Operator` protocol wrapper for `Sequential` chains | ✗ | **G4** |
| `partial_fit` / online learners | ✗ | deferred |
| dask-backed lazy compute | ✗ | deferred |

The xarray ↔ sklearn marshalling code is not duplicated; the work below
is additive.

---

## 3. Proposed work

### G1 — `nan_policy="mask"` on `XarrayEstimator`

Drop sample rows containing any NaN before delegating, then re-insert
NaN rows on the way out so output coordinates align with the input
sample axis. This is the policy that makes the bridge usable on
land-masked ocean grids without forcing the user to impute upstream.

```text
Original:    [s0, s1(NaN), s2, s3(NaN), s4]
Cleaned:     [s0, s2, s4]
sklearn  →   [r0, r2, r4]
Reassembled: [r0, NaN, r2, NaN, r4]    ← input sample coords preserved
```

Implementation note: the mask is computed per call (`_stack`) and used
in `_unstack` to re-insert NaN rows. No state is stored on the
estimator beyond what already exists, which keeps thread-safety
identical to today.

**Where:** extend `NanPolicy` literal + `_check_no_nan` site in
[sklearn_wrap.py](https://github.com/jejjohnson/xrtoolz/blob/main/src/xrtoolz/utils/_src/sklearn_wrap.py); add a
test matrix in `tests/test_sklearn_wrap.py` covering propagate / raise /
mask × DataArray / Dataset × transform / predict / inverse_transform.

### G2 — `da.sklearn` accessor (DataArray + Dataset)

```python
# Equivalent forms
Xt = XarrayEstimator(StandardScaler(), sample_dim="time").fit_transform(X)
Xt = X.sklearn.fit_transform(StandardScaler(), sample_dim="time")
```

The accessor is a thin adapter — it constructs an `XarrayEstimator`
internally and delegates. Registered via
`xr.register_dataarray_accessor("sklearn")` /
`register_dataset_accessor("sklearn")`. Single code path: there is no
parallel implementation.

**Where:** new `src/xrtoolz/utils/_src/sklearn_accessor.py`, exported
from `xrtoolz.utils`. Opt-in import — registering accessors at package
import time is acceptable since the cost is negligible and idempotent.

### G3 — xrpatcher composition pattern (no new code)

Document the pattern that `xrpatcher` (already a hard dep per
[OB-1.1](ob-1.1-xrpatcher-integration.md)) composes with
`XarrayEstimator` through xarray DataArrays only — no adapter needed.
Two canonical shapes:

```python
# (a) Per-patch local statistics — each patch fits its own scaler
patcher = XRDAPatcher(da=ssh, patches={"lat": 64, "lon": 64},
                      strides={"lat": 64, "lon": 64})
scaler  = XarrayEstimator(StandardScaler(), sample_dim="time")
scaled  = [scaler.fit_transform(patcher[i]) for i in range(len(patcher))]
out     = patcher.reconstruct(scaled)

# (b) Single global estimator, patch-wise inference for memory bounding
pca     = XarrayEstimator(PCA(n_components=10), sample_dim="time").fit(ssh)
recon   = patcher.reconstruct(
    [pca.inverse_transform(pca.transform(patcher[i])) for i in range(len(patcher))]
)
```

**Where:** a section in the bridge's user-facing notebook
([docs/notebooks/](../notebooks/)) and a recipe block in
`xrtoolz.utils.__init__` docstring. No new operators required.

### G4 — `SklearnOp(Operator)`

The remaining gap is composition with `Sequential`. Today, fitted
`XarrayEstimator` instances can be called like Operators in spirit
(`obj(x)` via a `transform` shim), but:

- `Sequential` calls `op(ds)`; `XarrayEstimator` exposes `transform(x)`.
- `Sequential` expects `Dataset → Dataset`; `XarrayEstimator` returns
  a `DataArray` for `DataArray` input.
- Operators have `get_config()` / `__repr__()`; sklearn estimators
  expose `get_params()`.

A thin Layer-1 wrapper resolves all three:

```python
class SklearnOp(Operator):
    """Operator-protocol wrapper around any sklearn estimator.

    Defers all marshalling to XarrayEstimator. Used to drop a fitted
    (or fittable) sklearn pipeline into a `Sequential([...])` chain.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        variable: str | None = None,           # which Dataset var to act on
        output_variable: str | None = None,    # default: overwrite `variable`
        sample_dim: Hashable | None = None,
        new_feature_dim: str = "component",
        nan_policy: Literal["propagate", "raise", "mask"] = "propagate",
        method: Literal["transform", "predict", "fit_transform"] = "transform",
    ): ...

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        wrap = self._wrap                      # XarrayEstimator (lazy-cloned)
        out  = getattr(wrap, self.method)(ds[self.variable])
        return ds.assign({self.output_variable or self.variable: out})

    def get_config(self) -> dict[str, Any]: ...
    def __repr__(self) -> str: ...
```

Decisions baked in (matching `D4` in [decisions.md](decisions.md)):

- `SklearnOp` does not import sklearn directly — it delegates via
  `XarrayEstimator`, which already imports `BaseEstimator`/`clone`. The
  Operator layer stays framework-agnostic in spirit.
- `method=` is explicit: an op chain rarely wants `fit_transform`
  (which mutates state every call); tests should warn when used inside
  `Sequential`.
- For multi-variable estimators (Dataset-in), `variable=None` passes
  the whole Dataset; `output_variable` then names the new variable.

**Where:** `src/xrtoolz/transforms/_src/sklearn_op.py`, exported from
`xrtoolz.transforms`.

---

## 4. Out of scope (deferred)

- **Dask-backed lazy compute.** Re-evaluate once `dask-ml` API surface
  stabilises.
- **`partial_fit` / streaming learners.** Useful but adds state-machine
  semantics that don't belong in an Operator. Revisit if a concrete
  user need appears.
- **Cross-validation splitters that respect spatial autocorrelation.**
  Belongs in a downstream `xrtoolz.experiments` module if at all.
- **Per-variable Dataset mode** (one estimator per variable).
  Implementable as `Sequential([SklearnOp(..., variable=v) for v in vars])`
  — no new primitive needed.

---

## 5. Acceptance checklist

- [ ] `XarrayEstimator(nan_policy="mask")` round-trips a NaN-laden grid
      through `StandardScaler` / `PCA` / `KMeans` without leaking NaN
      into sklearn and without losing the input's NaN locations on the
      way out.
- [ ] `da.sklearn.fit_transform(StandardScaler(), sample_dim="time")`
      is bitwise-identical to the explicit `XarrayEstimator` form.
- [ ] `Sequential([Validate(...), SklearnOp(StandardScaler(), ...),
      SklearnOp(fitted_pca, method="transform")])` runs end-to-end on a
      real ocean Dataset and preserves coords/attrs.
- [ ] `SklearnOp(estimator).get_config()` round-trips through
      `Sequential.get_config()` for serialisation parity with other
      operators.
- [ ] Notebook recipe demonstrating `XRDAPatcher` × `XarrayEstimator`
      for per-patch and global-fit-patch-infer patterns.
