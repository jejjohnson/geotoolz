# Spike #516: results

23 Oct 2026, branch `spike/stored-dims-kernelop` (PR #540).
Machine: Linux x86_64, 2 CPU cores, no GPU; Python 3.13.17, JAX 0.10.2 (CPU, x64 on), NumPy 2.4.5, xarray 2026.4.0, dask 2026.7.0, Equinox 0.13.8.
Reproduce from `packages/geotoolz-array`: `uv run pytest spike -o addopts= -q -s` and `uv run python spike/bench_op_o8.py`.

## Verdict

**Option A holds: build W1 as designed, with five amendments (F1–F5).**
One kernel, wrapped once in a `KernelOp`, ran unchanged on raw arrays, NumPy carriers, JAX carriers under `jit`, and dask-backed `DataArray`s.
New transforms never retraced, grids never aligned silently, and gradients flowed through carriers and padded observations.
None of the fallbacks (carriers stopping at the model boundary, `sample` returning raw arrays only) is needed.

## Gates

| Gate | Result | Evidence |
| --- | --- | --- |
| GA-G2 One trace, many scenes | **pass** | 1 trace over 100 random transforms; a new CRS adds 1, a new shape adds 1. The step derives spacing from the traced transform (`geometry=`) |
| GA-G15 Patches | **pass** | 64 patches, 64 transforms: 1 trace through `map_over` (one `jax.vmap` over data and transform rows), and 1 trace for a second scene. Output transforms equal input transforms bit for bit; the stitch equals today's carrier on the whole scene to rtol 1e-12 |
| GA-G16 DA gradient | **pass** | `jit(value_and_grad(J))` with 40 observations padded to 64 and to 128 with NaN coordinates and values: J and ∇ₓJ equal to rtol 1e-14, all finite |
| GA-G17 No JAX on I/O | **pass** | geocatalog, geocloud and geoproducts import, and `geocloud.hdf.read_netcdf` reads a file, with JAX hidden by an import hook; no `jax` module loaded. The file was written by the prototype's `to_xarray`, so geocloud also reads the encoding decided in geostack#77 (transform and EPSG:32613 exact) |
| GA-G18 Shared functions agree | **pass** for `combine`, `stack`, `isel`, `map_over` (exact) and `reduce`, `KernelOp` matched filter (≤ 1e-12) | NumPy, JAX and dask-backed xarray give the same values and the same dims; the xarray results stay lazy until computed. Not covered: `argreduce`, `take_along`, `concat`, `transpose`, `sel` |
| GA-G19 No silent alignment | **pass** | `combine`, `stack` and a `KernelOp` raise on a 10⁻⁷ m offset on NumPy, JAX eager, JAX under `jit` (needs F1) and dask xarray (`join="exact"`). Also: a `DataArray` with no xarray adapter raises naming `pipekit-array[xarray]` (geostack#90), and mixed NumPy/JAX inputs raise naming `to_jax()` |
| GA-G20 Lossless round trip | **pass** | 200 random windows and a 12.5° rotated grid round-trip bit for bit (transform, dims, labels, CRS, data). #463: centres only → edge origin, not identity. #464: edge coordinates under a centre `GeoTransform` raise. rioxarray reads the hand-written `spatial_ref` (transform to 1e-9, EPSG 32613) |
| GA-G21 Patch overhead | **pass** against a budget of today + 25 µs per op, set before measuring | 10k patches, 5-op chain, eager NumPy, three runs: today 276–297 µs per patch, prototype 336–360 µs (+8 to +16 µs per op). The same chain batched (one carrier, `map_over` under `jit`): 6.5–8.2 µs per patch |
| GA-G22 Reproducible costs | not run | #536, next |

## OP-O8 (geostack#79)

4D-Var cost over one window: 256 × 256 state, T = 8 frames, 5 diffusion steps per frame, N = 20,000 observations padded to a fixed N.
(A) the forward model returns a `GeoMapSeries` carrier rebuilt per evaluation, and H(x) is `sample` on the carrier, weights recomputed from its transform; (B) the L1 loop on raw frames, weights precomputed once.

| | A, L3 carrier | B, L1 loop |
| --- | --- | --- |
| traces of `jit(value_and_grad(J))` over three windows (20,000, 14,000, 9,500 real observations) | 1 | 1 |
| compile time | 0.66 s | 0.61 s |
| per-call time, median of 200 interleaved calls | 18.3 ms | 18.1 ms |
| XLA cost analysis: flops | 6.37 M | 3.73 M |
| XLA cost analysis: bytes accessed | 52.6 MB | 52.9 MB |

- **Paired ratio A / B: median 1.007, interquartile 0.958–1.057**, stable over three runs (1.004–1.014).
- J is identical and ∇ₓJ agrees to 7 × 10⁻¹⁵ relative.
- The rule in #79 (A if within 10 % and one trace) picks **A**.

Unpaired timings on this 2-core machine swung between 0.71 and 1.19 across runs, which is why the ratio is taken per interleaved pair.
A's extra flops are the bilinear weights recomputed in the graph; they cost nothing measurable here because the model dominates.
For a cheap model with many observations, caching the weights is an optimisation inside `sample`, not a reason to change the `ForwardModel` contract.
Not measured: GPU.

## Findings: amendments to the design

- **F1. `check_compatible` returns the carriers.**
  Under `jit` a transform is a tracer, so an exact grid check cannot be a Python `if`.
  Today's geoarray skips the transform comparison under a trace (`_require_same_grid`), so a 10⁻⁷ m offset passes silently inside `jit`.
  The prototype's adapter attaches `equinox.error_if` to the data and returns the guarded carriers; the caller must use them.
  Changes jejjohnson/pipekit#100 (adapter interface) and #517 (geoarray's adapter).
- **F2. Outputs keep the first input's dim order.**
  A gufunc puts core dims last, so the per-column score `(r,c)->(r)` with `core={"r": "y"}` comes back as `(x, y)`, which is not a `GeoArray["y x"]`.
  `KernelOp` transposes each output to the first input's dim order, with dims new to the output last, on every backend (xarray included, for parity).
  Changes jejjohnson/pipekit#103.
- **F3. lineax kernels are JAX-only.**
  The matched filter drafted in #516 uses `lineax.linear_solve`, which cannot run on NumPy or dask blocks.
  A dual-backend kernel uses `xp.linalg.solve`; a kernel that needs lineax declares `backends=("jax",)`.
  Changes #533 and the matched-filter kernel in radtrax.
- **F4. `geometry=` needs one transform per call.**
  Derived geometry from a batched carrier (`(P, 6)` transforms) varies along `patch`, and keyword inputs are passed whole, never mapped.
  `KernelOp` raises and names `map_over(op, x, "patch")`, which gives each item its own transform; that path is also the fast one (6.5 µs per patch above).
  Changes jejjohnson/pipekit#103 and `pipekit_kernels.md` ("What a kernel may take").
- **F5. An output-only core dim declares its size.**
  dask must know an output's shape before running, and `background_stats` makes `(c, d)` from `(r, c)`.
  `KernelSpec(sizes={"d": "c"})` says d has c's size.
  Changes jejjohnson/pipekit#103.

Smaller notes:

- **`sample` needs a per-point frame index** to read a `GeoMapSeries` at tracks (each observation on its own frame): `sample(series, xy, valid, tidx)`. Changes #522.
- **`geoarray._src.backend.jax_loaded`** tests `"jax" in sys.modules`, which is true when a tool blocks JAX with `sys.modules["jax"] = None`; `is_tracer` then fails importing `jax.core`. `sys.modules.get("jax") is not None` fixes it. Rare, so not a gate.
- **GA-G15's reference** is today's carrier on the whole scene, not geopatcher's stitch, because the prototype patches do not overlap.

## Consequences for the W1 issues

- jejjohnson/pipekit#98 and #512 build as designed, with F1–F5 folded into pipekit#100, pipekit#103, #517, #522 and #533.
- geostack#79 has its numbers; the rule picks A.
- #536 turns the GA-G21 and per-op timings into the committed benchmark.
