# Spike: stored dims and KernelOp on NumPy, JAX and xarray

Issue: jejjohnson/geotoolz#516 (epic #535). Design: `docs/design/geoarray.md`, `pipekit_kernels.md` and `operators.md` in jejjohnson/geostack.

This directory is a throwaway prototype: it is not shipped, not imported by `geoarray`, and is deleted once the gates are answered.
It tests one question: can a geoarray carrier with stored dims, and a `KernelOp` dispatching through per-carrier adapters, run one kernel unchanged on NumPy, JAX and xarray, without recompiling per scene, without aligning grids silently, at an acceptable cost?

## Layout

| Path | What it prototypes | Ships later in |
| --- | --- | --- |
| `proto/carrier.py` | `GeoArray` with stored `dims`, a `(*batch, 6)` transform leaf and a static rest | geoarray 0.3.0 (#517, #519) |
| `proto/dimarray.py` | `DimArray`: data, dims, labels; a pytree | pipekit-array (pipekit#101) |
| `proto/adapters.py` | the adapter registry and the adapters for `GeoArray`, `DimArray`, `xr.DataArray` | pipekit core + pipekit-array (pipekit#99, #100, #104); geoarray's adapter (#517) |
| `proto/kernelop.py` | `KernelSpec`, `KernelOp`, dispatch per backend | pipekit-array (pipekit#103) |
| `proto/shared.py` | `map_over`, `combine`, `stack`, `reduce` over the adapters | pipekit-array, re-exported by geoarray |
| `proto/sample.py` | `sample`: bilinear H(x) at padded points | geoarray (#522) |
| `proto/kernels.py` | the L1 kernels: normalized difference, matched-filter score and background statistics, the diagnostic gradient | geoops (#533, #539), radtrax |
| `test_gates.py` | one test per gate, GA-G2 and GA-G15 to GA-G21 | — |
| `bench_op_o8.py` | the OP-O8 measurements for geostack#79 | — |

Run from `packages/geotoolz-array`:

```bash
uv run pytest spike -o addopts= -q
uv run python spike/bench_op_o8.py
```

## Gates

| Gate | Passes when | Status |
| --- | --- | --- |
| GA-G2 One trace, many scenes | one trace across 100 random transforms; a new CRS or shape adds exactly one | pending |
| GA-G15 Patches | 64 patches with different transforms in one trace via `map_over`; stitch matches today's | pending |
| GA-G16 DA gradient | `grad(cost)` with padded observations; J and ∇ₓJ unchanged by the pad length and finite with NaN padding | pending |
| GA-G17 No JAX on I/O | geocatalog, geocloud and geoproducts import and read without JAX | pending |
| GA-G18 Shared functions agree | each shared function identical on NumPy, JAX and dask-backed xarray | pending |
| GA-G19 No silent alignment | `combine` and `stack` raise on a 10⁻⁷ m offset on every backend | pending |
| GA-G20 Lossless round trip | `from_xarray(to_xarray(g))` keeps transform, dims and coordinates, rotated grids included | pending |
| GA-G21 Patch overhead | 10k patches through a 5-op chain within a budget of today's carrier | pending |
| GA-G22 Reproducible costs | committed benchmark (#536) | after this spike |
| OP-O8 measurements | trace count, compile time and per-call time of `jit(grad(cost))` for an L3 and an L1 forward model | pending |

Results are posted on #516 and recorded in `RESULTS.md` here.
