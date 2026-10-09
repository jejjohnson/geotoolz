# Exact grid alignment

This record explains why the alignment checks work the way they do. How to
use them is the [Grid alignment](../how-to/grid-alignment.md) how-to.

## Problem

`GeoSlice` derives `shape` and `transform` by rounding
`length / resolution` half up. That hides subpixel misregistration: a
bbox 40,000 m wide at 30 m resolution silently becomes a 1,333-pixel grid
that is 10 m short of the requested extent. The error surfaces later, as
an off-by-one against a co-registered label tile or in a matchup join.

## Decisions

**Opt-in, not a new default.** `.shape` stays rounding-based (half up on
both axes, not Python's half-to-even `round`) for compatibility with every
existing loader. `aligned_shape()`, `align=` and `is_grid_aligned` are the
explicit strict paths.

**Snap preserves the affine origin.** For a north-up raster the affine
maps pixel `(0, 0)` to `(xmin, ymax)`. Snap holds those fixed and extends
`xmax` rightward and `ymin` downward, so the bounds still cover the
requested AOI and `transform.c` / `transform.f` match the nominal origin.
Snapping to the nearest lattice instead would move the origin; callers who
want that do it themselves and verify with `align="error"`.

**`align` is not part of identity.** Two slices with the same bounds,
interval, resolution and CRS compare and hash equal whatever their mode.
This keeps `set[GeoSlice]`, dict keys and the frozen dataclass's hash
contract intact.

**Warnings use `warnings`, not loguru.** The package calls
`logger.disable("geocatalog")` at import for library hygiene. A loguru
notice would be invisible by default, exactly when a user opted in to be
told about misaligned bounds.

**Tolerance is relative to the pixel.** The default
`step * 10 ** -PIXEL_PRECISION` (a thousandth of a pixel) is as strict at
0.0001° as at 10 m.

**`Literal` is checked at runtime.** A misspelled mode such as
`align="warning"` raises at construction rather than silently meaning
`"off"`.

## Not covered

- **Reprojection.** `to_crs` rescales resolution to preserve shape, so the
  child carries `align="off"`; a strict parent must not make `to_crs`
  raise on its own output.
- **`iter_slices` footprints.** Slices built from arbitrary polygons are
  almost never whole pixels and carry `align="off"`.
- **Time.** xreader's `_divide_evenly` has a `timedelta64` branch that was
  not ported; the interval index is not checked for uniform sampling.

## Attribution

`count_steps` is derived from
[`terrax.xreader.stencils._divide_evenly`](https://github.com/neuralgcm/terrax)
(Apache-2.0, © Google LLC; original author Stephan Hoyer).
Modifications © 2026 J. Emmanuel Johnson, licensed MIT under geocatalog's
overall MIT licence. The Apache header in
`packages/geotoolz-catalog/src/geocatalog/_src/grid.py` carries the
upstream notice.
