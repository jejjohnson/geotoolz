# geotoolz-ops (`geoops`) — agent rules

Operator families on `geoarray.GeoArray`, composed with `pipekit`
(`Operator`, `Sequential`, `Graph`). The root [`AGENTS.md`](../../AGENTS.md)
applies too; this file adds what is specific to `geoops`.

geoops is everything that computes. What re-indexes, joins, re-describes
or reads at points is geoarray's ([its membership test](../geotoolz-array/AGENTS.md#the-membership-test)):

```
geoarray: reads the georeference, never computes a field
  re-index     isel, sel, crop, read_window, write_window, pad
  join         stack, concat, combine      one grid; your fn computes
  re-describe  dims, coords, pixel_coords, valid_mask, footprint
  read at      to_pixel, interp_at         points in, values out

geoops: computes
  new values   indices, radiometry, mask, normalize, …   same grid
  new grid     reproject, resample, warp, mosaic, rasterize
```

The test: if the output holds only input values, fills and coordinates,
or values read at points you gave it, it is geoarray. Otherwise it is
geoops.

## Family layout

Every operator family (`radiometry`, `indices`, `compositing`, `segment`, …)
has the same shape, enforced by `tests/test_geotoolz.py::test_family_layout`:

```
geoops/<family>/
├── __init__.py         # re-exports only — no def / class
└── _src/
    ├── array.py        # Tier A: pure numpy primitives, jaxtyping-annotated,
    │                   # no GeoArray / metadata / operator state
    ├── operators.py    # Tier B: every pipekit.Operator of the family; judges
    │                   # nodata, calls Tier A, rewraps via wrap_like
    └── <topic>.py      # optional: constants / lookup tables / non-Operator
                        # helpers (qa/_src/scl.py, radiometry/_src/solar.py, …)
```

- A primitive lives in `array.py` and is exported from the family
  `__init__`; Tier-A names stay out of the top-level `gz.*` namespace. Every
  public Operator class is top-level (including sub-namespaces such as
  `geom.coregister`), except the justified few listed in the
  `geoops/__init__.py` docstring.
- Each public name has one home family — no cross-family re-exports
  (`tests/test_geotoolz.py::test_public_operators_exported`,
  `::test_one_home_per_public_name`).
- A family may call another family's primitive or operator when that *is*
  the maths (`compositing` → `indices.ndvi`, `plume` → `segment`,
  `viz` → `radiometry` stretches) rather than re-implementing it.
- Not a family: `patch_ops.py`, the geopatcher bridge (`[patch]` extra).
  Product readers live in geotoolz-products, never here.
- Removals and renames are outright
  (`tests/test_geotoolz.py::test_removed_modules_are_gone`).

## Shared plumbing — use it, extend it

Anything two or more families need lives in `geoops/_src/`:

| Module | What it gives you |
|---|---|
| `wrap.py` | `wrap_like` / `rewrap_attrs` / `adopt_attrs` — the one way to turn a result back into a `GeoArray` with the input's georeferencing (and the place a georeader `GeoTensor` is refused) |
| `engine.py` | `to_engine` / `from_engine` — the one boundary to georeader's functions (reprojection, mosaics, rasterize / vectorize, windowed reads, solar geometry); no `GeoTensor` crosses into or out of an operator |
| `valid.py` | the nodata model: `invalid_values`, `valid_pixels`, `mask_invalid_to_nan`, `restore_fill`, `wrap_filled` |
| `bands.py` | `resolve_band` / `resolve_bands` (index or name), `band_names`, `resolve_wavelengths`, band-attr bookkeeping |
| `shape.py` | `band_axis`, `band_dim` / `dim_axis` (an operator's `dim=` and its position), `require_ndim`, `map_frames` / `over_frames` (split a time series by its `time` dim) |
| `dtype.py` | `as_float` — promote integer DNs before arithmetic |
| `geo.py` | `require_geoarray`, `reject_geotensor`, `affine_of`, `as_pixel_window`, `require_projected_crs`, `ground_pixel_size`, `require_grid_match`, `pixel_xy` |
| `labels.py` | connected components, holes, skeleton length, region properties |
| `samples.py` | `(c, h, w)` cube ↔ `(n, c)` samples for per-pixel estimators |
| `stretch.py` | the NaN-aware percentile stretch |
| `blending.py` | triangular weights and overlap-add |
| `fitted.py` | `fit_once` / `fit_lock` for fit/transform operators |
| `config.py` | `get_config()` coercion (`jsonable`, `as_tuple`, …) |
| `optional.py` | `import_optional` for extras-gated dependencies |

## Operator parameter vocabulary

Every `pipekit.Operator` constructor is **keyword-only** —
`__init__(self, *, ...)`, with no exception for a wrapped estimator / model /
patcher (`PixelwisePCA(estimator=PCA())`, `ModelOp(model=net)`,
`GridSampler(patcher=p)`). One concept has one parameter name across every
family:

| Concept | Name | Notes |
|---|---|---|
| The band dim | `dim: str` (default `"band"`) | A dim *name*: the position is read from the carrier's stored dims (`geoops._src.shape.dim_axis`; `band_dim` handles the deprecated `axis: int \| None = None`, which maps `-3` to `"band"` for one release). Never a reduction or an orientation. `segment`'s skimage wrappers take `dim: str \| None`, `None` for no channel axis. Tier-A primitives keep a positional `axis`. |
| Reduction axes | `reduce_axes` | Tuple of axes (or `None` = global): `radiometry.PercentileClip`, `viz.StretchToUint8`, the `normalize` primitives, `radiometry.dos1`. |
| Orientation | `direction` | `"column"` / `"row"` (`restore.DestripeColumn`), `"scan"` / `"sample"` (`geom.SegmentStitch`). |
| One band | one `BandRef` name per band (`red`, `nir`, `swir1`, `qa_band`, `band`, …) | Integer position *or* band name; no `*_idx` twins. Tier-A primitives take integer `*_idx` positions. |
| Several bands | `bands` | List of `BandRef` (`spectral.SelectBands`, `viz.Composite`). `io` readers keep rasterio's 1-based file `indexes`. |
| Value written into pixels | `fill_value` | Also the carrier attribute (`GeoArray.fill_value`) and `wrap_like`'s keyword; a strategy string is `strategy` (`restore.ReplaceOutliers`). |
| RNG seed | `seed` | |
| Neighbourhood side length | `window` | Pixels (odd int, or `(h, w)`): despeckle, destripe, `MedianDenoise`, `NLMeans`, `CLAHE`, `AdaptiveWindowBackground`, `SpectralSmoothing`. |
| Neighbourhood half-width / distance | `radius`, `search_radius` | Not a window (radius `r` ≈ window `2r + 1`): `mask.BufferMask` (with `unit`), `restore.GapFillIDW`, `restore.NLMeans`, `viz.AnnotatePoints`. |
| Output size | `size` | Crop / tile / chip size (`augment.RandomCrop`, `geom.Tile`, `geom.SlidingWindow`, `patch_ops` samplers) — not a window. `io.ReadWindow(window=...)` is a rasterio pixel window, not a size. |
| Gaussian scale | `sigma` | `segment.Quickshift(kernel_size=...)` keeps skimage's name: a kernel *width*, not a window. |
| Areas | `min_area_px`, `max_hole_area_px`, `min_area_m2` | Unit suffix is mandatory. |
| Connectivity | `connectivity: 4 \| 8` | Converted internally for skimage / scipy (`1` / `2` is rejected). |
| Wavelengths | `wavelengths` (nm) | Also the `attrs["wavelengths"]` key; qualified variants `source_wavelengths` / `target_wavelengths`. |
| Percentile stretch bounds | `lower` / `upper` | |
| QA selection | `qa_band`, `bits`, `values`, `targets` | `targets` = `SENSOR_QA_REGISTRY` names. |
| Denominator stabiliser | `eps` (default `1e-10`) | |
| scikit-learn wrappers | `Pixelwise*` | `PixelwisePCA`, `PixelwiseKMeans`, … never shadow the sklearn class they wrap. |

`tests/test_operator_contract.py::test_constructors_are_keyword_only` and
`::test_constructor_vocabulary` enforce the style and reject the retired
spellings.

## Tests

- Every operator must pass `tests/test_operator_contract.py`, which
  enforces the pipekit `Operator` and `GeoArray` contracts in the root
  `AGENTS.md`. The test walks every class on its own; register a new
  operator's constructor arguments in `CTOR_KWARGS` / `RUNTIME_CTOR_KWARGS`
  when it needs any.
- Tiers: `slow` and `integration` markers; the fast tier
  (`-m "not slow and not integration"`) runs in CI, the rest via the
  "Extended Tests" workflow.
- Coverage gate: 65 % (`[tool.coverage.report] fail_under`).
- Run from this directory: `uv run pytest tests/test_indices.py -v`.
- jaxtyping annotations need ruff's `F722` ignore, configured in this
  package's `[tool.ruff]`.
