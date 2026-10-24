# geotoolz-array (`geoarray`) — agent rules

The georeferenced array every package of the stack will share. The root
[`AGENTS.md`](../../AGENTS.md) applies too; this file adds what is specific
to `geoarray`. It is the bottom of the workspace: it depends on numpy,
pyproj, affine, pipekit core (pure Python) and pipekit-array only, and
never imports another workspace package or rasterio. georeader is imported by one module only, the `[georeader]`
adapter `_src/georeader.py`, and lazily (ruff's `TID251` rejects it
anywhere else).

## What lives here

- **`GeoArray`** (`_src/array.py`) — a frozen dataclass that *holds* a
  NumPy or JAX array (composition: JAX arrays cannot be subclassed) with
  its stored dims (ending in `y x`), coordinates, transform, CRS, fill
  value and attrs.
- **Coordinates** (`_src/coords.py`, public as `geoarray.coords`) —
  `SpectralAxis`, `VariableAxis`, `VerticalAxis`, `TimeAxis` and the grid
  coordinates; frozen and hashable by value, sliced with their dim.
- **The named-array adapter** (`_src/adapter.py`) — `GeoArrayAdapter`,
  registered with pipekit at import, so pipekit-array's dim-level
  functions and `KernelOp` run on carriers. geoarray re-exports those
  functions (`geoarray.reduce is pipekit_array.reduce`); never write a
  geoarray-local copy, add an adapter fast path instead.
- **`Grid`, `PixelWindow`** (`_src/grid.py`) — a pixel grid without data,
  and a rectangle of pixels (named apart from geopatcher's window
  *functions*, in rasterio's `Window` field order); all affine maths
  lives here.
- **`FrozenAttrs`** (`_src/attrs.py`) — hashable metadata, and the
  per-band rules (`PER_BAND_KEYS`, `select_bands`, `concat_bands`) that
  keep every per-band key one entry per band.
- **Indexing and nodata** (`GeoArray.__getitem__`, `isel`, `squeeze`,
  `valid_mask`, in `_src/array.py`) — spatial slices move the transform
  through `shift_scale_transform`, the one place slice maths lives.
- **Geometry functions** (`_src/geometry.py`) — `crop`, `read_window`,
  `write_window`, `pad`, `pixel_coords`, `footprint`, and the joins
  `combine` / `stack` / `concatenate` (`GridMismatchError`).
- **The backend dispatch** (`_src/backend.py`) — the one place that tells
  NumPy from JAX and imports JAX on demand; `_src/options.py` holds
  `set_options` (``warn_on_transfer``).
- **`RasterSource`, `AsyncRasterSource`** (`_src/source.py`) — the
  protocols every lazy reader of the stack implements (grid, shape, dtype,
  fill, attrs; `read_window` / `load`, or `aread_window` / `aload`). A
  `GeoArray` is a `RasterSource`. geoarray ships no reader.
- **The georeader adapter** (`_src/georeader.py`, public as
  `geoarray.interop.georeader`) — `from_georeader` / `to_georeader` for
  carriers, `source_from_georeader` / `source_to_georeader` for lazy
  readers; the one module that imports georeader (lazily, `[georeader]`
  extra), mapping public attributes only.

The design is
[geostack's `docs/design/01_geoarray.md`](https://github.com/jejjohnson/geostack/blob/main/docs/design/01_geoarray.md);
its workstream letters (A type, B geometry, C interop) and gate IDs (GA1,
GB2, …) name the tests that enforce it.

## The membership test

geoarray holds operations that **read only the georeference**:

```
re-index      isel, crop, read_window, pad, the joins
re-describe   coords, dims, pixel_coords, footprint
read at       to_pixel, interp_at   (points in, values out)
```

Anything that **writes a new grid** (warp, resample, reproject, mosaic,
CRS kernels) belongs in `geoops.geom`. `interp_at` blends neighbours, but
its output lives on the points it was given, not on a new grid, so it is
"read at", not resampling. The nouns `sample` (an ML batch item) and
`member` (an ensemble draw) are kept for dim names, never function names.

## The pytree split

| Field | Pytree role | Changing it |
|---|---|---|
| `data` | leaf | never recompiles |
| `transform` (`(*batch, 6)` float64) | leaf | never recompiles; differentiable |
| `dims`, `coords`, `crs`, `fill_value`, `attrs` | static | recompiles |

- **Stored dims are the source of truth.** Never read a layout from
  `ndim`; code that builds a carrier passes `dims=` (library code that
  still has only the 0.2 layout passes `DIMS[-ndim:]` explicitly, so the
  `DeprecationWarning` reaches users only). The first `transform.ndim - 1`
  dims are batch dims.
- **A bare `jax.vmap` maps the leading dim only**: `_unflatten` matches the
  static dims to the leaves (`pipekit_array.dimarray.vmapped_dims`). Map a
  named dim with `geoarray.map_over`.
- **Static fields must be hashable and compare equal when equal.** `attrs`
  go through `FrozenAttrs`; every NaN (fill or attr) is normalised to the
  one `math.nan` object, because `nan != nan` would otherwise make equal
  carriers recompile.
- **`_unflatten` never validates.** JAX unflattens with placeholder leaves
  while tracing; keep it a plain field assignment.
- **A JAX transform is float64**, so JAX carriers need `jax_enable_x64`
  (`X64Error` otherwise). Never round a transform to float32 silently.
- **Under a trace, host-side reads raise `GeoArrayTraceError`** (`grid`,
  `to_numpy`), naming the alternative. Never let JAX's generic
  `TracerArrayConversionError` reach the user from geoarray code.
- **Grid checks never skip under a trace.** Traced transforms are compared
  at run time by a `jax.debug.callback` that raises `GridMismatchError`
  (JAX re-raises it as `JaxRuntimeError`). Not `equinox.error_if` on the
  data: XLA prunes that check when nothing needs those values, as in
  `grad` with respect to another carrier (#541).

## Conventions

- **Positionally, a 4-D carrier never loses its band axis alone**: an
  integer band index (`g[:, k]`) or a squeeze that would turn
  `(time, band, y, x)` into `(time, y, x)` raises, because 0.2 code (the
  geoops kernels, `axis=-3`) reads `(T, H, W)` as `(C, H, W)`. By name
  (`isel`, `sel`) the stored dims say what is left, so it is allowed.
- **NumPy and JAX mix eagerly (design decision C1).** Mixed inputs promote
  to JAX (`_promote` in `_src/geometry.py`); output stays where the
  computation ran; `__array__` copies JAX data to the host eagerly and
  raises `GeoArrayTraceError` under a trace.
- JAX is optional. Import it through `_src/backend.py` (`import_jax`,
  `is_jax_array`, `is_tracer`); a NumPy-only install never loads it.
  The pytree is registered at import whenever JAX is installed.
- Transforms are (a, b, c, d, e, f), the `affine.Affine` / rasterio order,
  never GDAL's (c, a, b, f, d, e).
- georeader is LGPL-3.0 and geotoolz is MIT: write geoarray from the
  behaviour and its tests, never by copying georeader code. georeader may
  appear in tests as an oracle and behind an optional extra.

## Tests

- Run from this directory: `uv run pytest tests/test_array.py -v`.
- JAX tests use the `x64` fixture in `conftest.py`, which turns 64-bit JAX
  on for the test and restores the previous setting.
- The all-extras CI run fails on any skip, so a test that needs an extra
  uses `pytest.importorskip` and the extra is in the root dev group.
