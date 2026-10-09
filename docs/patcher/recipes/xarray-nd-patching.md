# xarray N-D patching

Patch an `xarray.DataArray` by dimension name, index the patches like a
list for an ML loader, and rebuild a `DataArray` with its coords. This
covers what `xrpatcher` and `xbatcher` do, with the four axes.

| xrpatcher | geopatcher |
|---|---|
| `XRDAPatcher(da, patches=..., strides=...)` | `SpatialPatcher(geometry=Rectangular(size), sampler=RegularStride(step))` over `geopatcher.fields.XarrayField(da)` |
| `check_full_scan=True` | `spatial.sampler.RegularStride(step, check_full_scan=True)` |
| `patcher[i]`, `len(patcher)`, `cache=` / `preload=` | `geopatcher.run.IndexedPatchView(patcher, field, cache=True, preload=True)` |
| `patcher.reconstruct(outputs)` | `patcher.merge_to_xarray(patches, field)` |

Needs the `[grid]` extra. ADR-005 in the
[design decisions](../decisions.md) explains why the view is a
`Sequence[Patch]` and why its cache lives on the view.

## Index, process and rebuild

```python
import numpy as np
import xarray as xr

import geopatcher as gp
from geopatcher.fields import XarrayField
from geopatcher.run import IndexedPatchView

# Shapes: a 240 × 360 lat/lon grid cut into 30 × 30 patches.
u: xr.DataArray = xr.DataArray(
    np.random.default_rng(0).random((240, 360), dtype=np.float32),  # (240, 360) float32
    dims=("latitude", "longitude"),
    coords={"latitude": np.linspace(89.5, -89.5, 240), "longitude": np.linspace(0, 359, 360)},
)
field: XarrayField = XarrayField(u)

patcher: gp.SpatialPatcher = gp.SpatialPatcher(
    geometry=gp.spatial.geometry.Rectangular(size=(30, 30)),
    sampler=gp.spatial.sampler.RegularStride(step=(30, 30), check_full_scan=True),
    window=gp.spatial.window.Boxcar(),
    aggregation=gp.spatial.aggregation.OverlapAdd(),
)

view: IndexedPatchView = IndexedPatchView(patcher, field, cache=True, preload=True)
n: int = len(view)                                    # 96 = 8 × 12 patches
chip: xr.DataArray = view[0].data                     # (30, 30) float32, dims (latitude, longitude)

outputs: list[gp.Patch] = [view[i].with_data(view[i].data * 2.0) for i in range(n)]  # (30, 30) float32 each
rebuilt: xr.DataArray = patcher.merge_to_xarray(outputs, field)  # (240, 360) float32, coords restored
```

- `view[i]` is a `Patch`: `.data` is the slice and `anchor`, `indices`
  and `weights` come with it.
- `merge_to_xarray` casts back to the source dtype when every value fits;
  call `patcher.merge` for the raw `np.ndarray`.

## Wrap the view for an ML loader

`IndexedPatchView` is a stdlib `Sequence`, so a torch `Dataset` or Grain
`RandomAccessDataSource` is a three-method wrapper.

```python
import numpy as np
import xarray as xr

import geopatcher as gp
from geopatcher.fields import XarrayField
from geopatcher.run import IndexedPatchView


class PatchDataset:
    """Same shape as torch.utils.data.Dataset / grain.RandomAccessDataSource."""

    def __init__(self, view: IndexedPatchView) -> None:
        self.view = view

    def __len__(self) -> int:
        return len(self.view)

    def __getitem__(self, i: int) -> np.ndarray:
        return self.view[i].data.values               # (30, 30) float32


field: XarrayField = XarrayField(
    xr.DataArray(np.zeros((240, 360), dtype=np.float32), dims=("latitude", "longitude"))  # (240, 360) float32
)
patcher: gp.SpatialPatcher = gp.SpatialPatcher(
    geometry=gp.spatial.geometry.Rectangular(size=(30, 30)),
    sampler=gp.spatial.sampler.RegularStride(step=(30, 30)),
    window=gp.spatial.window.Boxcar(),
    aggregation=gp.spatial.aggregation.OverlapAdd(),
)
dataset: PatchDataset = PatchDataset(IndexedPatchView(patcher, field, cache=True))
batch: np.ndarray = np.stack([dataset[i] for i in range(4)])  # (4, 30, 30) float32
```

The view pickles, so multi-worker loaders work under `spawn`,
`forkserver` and `fork`. Each worker gets the anchor list and the
`PatchCache` binding, and starts its own empty in-memory cache. The
patcher and field must pickle too; every built-in `Field` does.

`IndexedPatchView(TemporalPatcher(...), series)` works the same way; pass
`patcher_kwargs={"time_axis": 1, "coord": times}` for a non-default time
axis or a stencil geometry.

## Pitfalls

- **`check_full_scan=True`** raises `IncompleteScanConfiguration` when the
  stride leaves a trailing partial tile — turn it on when silent
  truncation would change your epoch length. It defaults to `False`.
- **Every dim is tiled.** `RegularStride` over a `GridDomain` walks every
  coord dim. For a cube with a time axis, slice time first
  (`da.isel(time=k)`) or use a [SpatioTemporalPatcher](temporal-patching.md#space-time-cubes).
- **The in-memory cache is per process.** For a cache that survives runs,
  pass a [PatchCache](patch-cache.md).
