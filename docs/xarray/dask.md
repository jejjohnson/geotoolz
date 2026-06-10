# Dask

Every `xrtoolz` operator accepts **dask-backed (chunked)** xarray inputs,
not just eager numpy ones. Where an operator's kernel is naturally
per-chunk it stays **lazy** — the work is deferred into the dask graph and
only runs on `.compute()`. Where the algorithm needs the whole array (a
global neighbour search, a global fit), the operator still accepts dask
input but **materialises internally**, with an identical result.

!!! tip "The one rule: keep *core* dims in a single chunk"
    An operator transforms along its *core* dimensions — the spatial
    `(lat, lon)` for a 2-D gap-filler, the transform dim for an FFT or
    wavelet, the smoothing dim for a filter. **Chunk the batch dims** (e.g.
    `time`) and **keep each core dim whole**:

    ```python
    ds = ds.chunk({"time": 1, "lat": -1, "lon": -1})   # -1 = one chunk
    ```

    Operators that need a whole core slice are built with
    `allow_rechunk=False`, so a *split* core dim raises a clear error rather
    than silently rechunking (which would corrupt a windowed transform).

## A lazy pipeline

The composition layer is dask-transparent — chunk once at the source and the
whole `Sequential` / `Graph` stays lazy until you ask for a result:

```python
import xarray as xr
from xrtoolz import Sequential
from xrtoolz.geo import RemoveMean
from xrtoolz.ocn.operators import Streamfunction, GeostrophicVelocities

ds = xr.open_mfdataset("ssh_*.nc").chunk({"time": 1, "lat": -1, "lon": -1})

pipeline = Sequential(
    RemoveMean("time"),
    Streamfunction(),
    GeostrophicVelocities(),
)

lazy = pipeline(ds)        # nothing computed yet — a dask graph
result = lazy.compute()    # run it (optionally under a dask scheduler/cluster)
```

## Per-operator tiers

Every operator falls into one of three tiers. The
[`tests/test_dask_compat.py`](https://github.com/jejjohnson/xrtoolz/blob/main/tests/test_dask_compat.py)
sweep asserts numpy/dask **result parity** for all of them, and asserts
**laziness** for the lazy tiers.

| Tier | Behaviour | Operators |
|------|-----------|-----------|
| **Lazy** | stays a dask graph | `calc`-driven `ocn` (KE, streamfunction, velocity magnitude); `geo` (`RemoveMean`, climatology, validators, masks); `transforms.PowerSpectrum`; `Coarsen`; `RegridLike`; all pixel/spectral **metrics**; the gap-fillers (`FillNaN{Laplacian,Spatial,RBF,Biharmonic,IDW}`); the smoothers (`GaussianSmooth`, `MovingAverage`, `fir_filter`, `lowpass_filter`); morphology |
| **Eager (dask-safe)** | accepts dask, computes internally, identical result | `calc.gradient`/`laplacian` and the `ocn` diagnostics built on them (`RelativeVorticity`, `Divergence`, `GeostrophicVelocities`, `OkuboWeiss`); binning (`Bin2D`, `Histogram2D`); the sklearn bridge (`XarrayEstimator`, decompositions, `SklearnModelOp`) |
| **N/A** | numpy-array primitives — no xarray, so no dask | `points_to_grid`, `idw_to_grid`, `kde_to_grid` (use the `*ToGrid` operators for the xarray surface) |

!!! note "Why some operators can't be fully lazy"
    `Bin2D`, the IDW/KNN regridders, RBF infilling, and the scikit-learn
    estimators rest on a **global** computation — a KD-tree over every point,
    or a single fit across the whole sample. Those have no correct per-chunk
    decomposition, so the operator gathers what it needs and computes it.
    They remain perfectly usable on dask-backed data; they just aren't a
    deferred graph. Making them distributed would mean reimplementing the
    underlying KD-tree / least-squares solvers, which is out of scope.

## Testing your own pipelines

The `array_backend` / `maybe_chunk` fixtures (in `tests/conftest.py`) let a
single test body run against both backends:

```python
def test_my_pipeline(array_backend, maybe_chunk):
    ds = maybe_chunk(build_dataset(), array_backend, {"time": 1})
    out = my_pipeline(ds)
    if array_backend == "dask":
        assert out["u"].chunks is not None     # stayed lazy
        out = out.compute()
    xr.testing.assert_allclose(out, expected)
```
