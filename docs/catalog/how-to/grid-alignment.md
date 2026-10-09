# Check grid alignment

Make a misaligned `GeoSlice` fail loudly instead of silently gaining or
losing a pixel. By default `GeoSlice.shape` rounds `extent / resolution`,
so a 40 000 m box at 30 m becomes 1 333 pixels, 10 m short. The error
only shows later, as an off-by-one against a label tile.

## Choose a check

| You want | Use |
| --- | --- |
| An exact pixel count, or an error | `GeoSlice.aligned_shape()` |
| Every slice checked when it is built | `GeoSlice(..., align="warn" \| "error" \| "snap")` |
| To know whether two slices share a pixel lattice | `geocatalog.grid.is_grid_aligned(a, b)` |
| An exact step count for any length | `geocatalog.grid.count_steps(length, step)` |

```python
import warnings

import pandas as pd

import geocatalog as gc
from geocatalog.grid import GridAlignmentWarning, count_steps, is_grid_aligned

june: pd.Interval = pd.Interval(pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-30"), closed="both")
loose: gc.GeoSlice = gc.GeoSlice(
    bounds=(0.0, 0.0, 105.0, 100.0), interval=june,                  # x-extent 105 m: 10.5 pixels
    resolution=(10.0, 10.0), crs="EPSG:32629",
)
rounded: tuple[int, int] = loose.shape                               # (10, 11), silently rounded
try:
    loose.aligned_shape()                                            # raises, naming the residual
except ValueError as err:
    print(err)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", GridAlignmentWarning)
    snapped: gc.GeoSlice = gc.GeoSlice(
        bounds=(0.0, 0.0, 105.0, 100.0), interval=june,
        resolution=(10.0, 10.0), crs="EPSG:32629", align="snap",
    )                                                                # bounds → (0, 0, 110, 100)
exact: tuple[int, int] = snapped.aligned_shape()                     # (10, 11)

shifted: gc.GeoSlice = gc.GeoSlice(
    bounds=(5.0, 0.0, 115.0, 100.0), interval=june, resolution=(10.0, 10.0), crs="EPSG:32629"
)
same_lattice: bool = is_grid_aligned(snapped, shifted)               # False: origins differ by 5 m
report: dict = is_grid_aligned(snapped, shifted, explain=True)       # per-axis residuals
steps: int = count_steps(length=100.0, step=10.0, label="x-extent")  # 10
```

## `align=` modes

| Mode | Behaviour |
| --- | --- |
| `"off"` (default) | no check |
| `"warn"` | emit `GridAlignmentWarning`, keep the bounds |
| `"error"` | raise `ValueError` on the first misaligned axis |
| `"snap"` | grow the bounds outward to whole pixels, keeping `xmin` and `ymax`; warn per edit |

A misspelled mode raises at construction. `align` is not part of a
slice's identity, so slices that differ only in mode compare and hash
equal.

The warnings go through Python's `warnings` module, not loguru, so they
show even though geocatalog's logger is off. Filter them as usual:
`warnings.simplefilter("error", GridAlignmentWarning)` in tests, `"ignore"`
in audited pipelines.

## Tolerance

The default tolerance is a thousandth of a pixel:
`step * 10 ** -PIXEL_PRECISION`. It scales with the resolution, so the
check is as strict at 0.0001° as at 10 m. Pass `tol=` (absolute, in CRS
units) to `count_steps` or `is_grid_aligned` to change it.

## Limits

- **Different CRSs.** `is_grid_aligned` returns `False`; reproject one
  side with `GeoSlice.to_crs` first.
- **Reprojected slices.** `to_crs` keeps the output shape, so its result
  is rarely a whole number of pixels; it always carries `align="off"`.
- **Catalog footprints.** `iter_slices` builds slices from arbitrary
  footprints with `align="off"`. Call `aligned_shape()` when you need a
  guarantee.
- **Time.** The interval is not checked for regular sampling.

Why snapping keeps the origin, and where `count_steps` comes from, is in
the [design record](../design/exact-grid-alignment.md).
