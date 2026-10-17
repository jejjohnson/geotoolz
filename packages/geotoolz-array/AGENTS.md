# geotoolz-array (`geoarray`) — agent rules

The georeferenced array every package of the stack will share. The root
[`AGENTS.md`](../../AGENTS.md) applies too; this file adds what is specific
to `geoarray`. It is the bottom of the workspace: it depends on numpy,
pyproj and affine only, and never imports another workspace package or
rasterio. georeader is imported by one module only, the `[georeader]`
adapter `_src/georeader.py`, and lazily (ruff's `TID251` rejects it
anywhere else).

## What lives here

- **`GeoArray`** (`_src/array.py`) — a frozen dataclass that *holds* a
  NumPy or JAX array (composition: JAX arrays cannot be subclassed) with
  its transform, CRS, fill value and attrs.
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

geoarray holds only operations that **re-index or re-describe pixels
without computing new values**: the output contains only input values,
fill values or coordinates. Slicing, crops, windows and pixel coordinates
belong here; warping, resampling, CRS kernels and anything that
interpolates belong in `geoops.functional`.

## The pytree split

| Field | Pytree role | Changing it |
|---|---|---|
| `data` | leaf | never recompiles |
| `transform` (`(6,)` float64) | leaf | never recompiles; differentiable |
| `crs`, `fill_value`, `attrs` | static | recompiles |

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

## Conventions

- **A 4-D carrier never loses its band axis alone**: an integer band
  index or a squeeze that would turn `(T, C, H, W)` into `(T, H, W)`
  raises, because `(T, H, W)` reads as `(C, H, W)`.
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
