# xarray N-D patching — the xrpatcher migration path

If you're coming from `xrpatcher` (or `xbatcher`), `geopatcher` already
covers the same indexed-access + reconstruct workflow — just through the
four-axis Patcher abstraction. This recipe walks through the migration in
side-by-side form.

The three pieces you need are already in `geopatcher`:

- `XarrayField` — wraps an `xarray.DataArray` as a Field with a
  `GridDomain` view.
- `SpatialPatcher` + `SpatialRectangular` + `SpatialRegularStride` —
  the four-axis composition produces patches indistinguishable from
  `xrpatcher`'s slices.
- `IndexedPatchView` — random-access wrapper exposing
  `len(view)` / `view[i]` / `for p in view` over the patcher's anchors.

Two new bits of sugar make the side-by-side really one-for-one:

- `SpatialRegularStride(check_full_scan=True)` — raises
  `IncompleteScanConfiguration` when the stride doesn't exactly tile
  the domain. Same robustness win as `xrpatcher`'s eponymous flag.
- `SpatialPatcher.merge_to_xarray(patches, field)` — `merge` + rewrap as
  `xarray.DataArray` with the original coords intact.

See **ADR-005** in `docs/decisions.md` for the design rationale (why
`Sequence[Patch]` rather than `torch.utils.data.Dataset`, why the cache
lives on the view).

## Prerequisites

```bash
pip install 'geopatcher[grid]'   # xarray
```

## Side-by-side

```python
# Before — xrpatcher
import xarray as xr
from xrpatcher import XRDAPatcher

da = xr.tutorial.load_dataset("eraint_uvz").u[..., :240, :360]
patcher = XRDAPatcher(
    da,
    patches={"latitude": 30, "longitude": 30},
    strides={"latitude": 30, "longitude": 30},
    check_full_scan=True,
    cache=True,
    preload=True,
)

print(len(patcher))             # number of patches
patch = patcher[0]              # one DataArray, cached
outs = [model(patcher[i]) for i in range(len(patcher))]
recon = patcher.reconstruct(outs)   # → DataArray with restored coords
```

```python
# After — geopatcher
import xarray as xr
from geopatcher import (
    IndexedPatchView,
    SpatialBoxcar,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRectangular,
    SpatialRegularStride,
)
from geopatcher.fields import XarrayField

da = xr.tutorial.load_dataset("eraint_uvz").u[..., :240, :360]
field = XarrayField(da)

patcher = SpatialPatcher(
    geometry    = SpatialRectangular(size=(30, 30)),
    sampler     = SpatialRegularStride(step=(30, 30), check_full_scan=True),
    window      = SpatialBoxcar(),
    aggregation = SpatialOverlapAdd(),
)

view = IndexedPatchView(patcher, field, cache=True, preload=True)
print(len(view))                # number of patches
patch = view[0]                 # one Patch, cached
outs = [model(view[i].data) for i in range(len(view))]
recon = patcher.merge_to_xarray(outs_as_patches, field)
                                # → DataArray with restored coords
```

The key differences:

- **`view[i].data`** is the slice (a `DataArray`); the `Patch` carrier
  also holds `anchor`, `indices`, and `weights`. xrpatcher exposed the
  raw slice; geopatcher gives you the metadata alongside.
- **Patch size is on the geometry, stride on the sampler.** The
  four-axis split is what unlocks the rest of `geopatcher`'s surface
  (windows, aggregations, hooks, on_error, journal, prefetch) — the
  patcher is still a value object you can swap pieces of.
- **`merge_to_xarray`** wraps `field.with_data(patcher.merge(...))`
  internally; if you want the raw `np.ndarray`, call `patcher.merge`
  directly.

## Random access for torch / Grain

`IndexedPatchView` is a stdlib `Sequence[Patch]` — zero ML-framework
dependencies in `geopatcher` core. Wrap in one line where you need
framework-specific shapes:

```python
import torch

class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, view: "IndexedPatchView") -> None:
        self.view = view

    def __len__(self) -> int:
        return len(self.view)

    def __getitem__(self, i: int) -> torch.Tensor:
        return torch.as_tensor(self.view[i].data.values)
```

```python
import grain.python as grain

class PatchSource(grain.RandomAccessDataSource):
    def __init__(self, view: "IndexedPatchView") -> None:
        self.view = view

    def __len__(self) -> int:
        return len(self.view)

    def __getitem__(self, i: int):
        return self.view[i].data.values
```

Both wrappers benefit from `IndexedPatchView`'s in-memory cache without
any framework changes.

## When to use `check_full_scan=True`

Turn it on when:

- You're training over a region of fixed size and silent truncation
  would change your epoch length without warning.
- You're round-tripping through `merge` and a partial trailing tile
  would shift the reconstruction boundary.

Leave it off when:

- You explicitly want the "stride past the edge, drop the leftover"
  behaviour — sliding-window inference where the last under-full
  position is acceptable.

The default is `False` (matches the existing geopatcher behaviour);
xrpatcher migrators who relied on `check_full_scan=True` should flip
the flag.

## v0.1 limitations

- **2-D and N-D where every dim gets tiled.** `SpatialRegularStride`
  over a `GridDomain` walks every coord dim with `(size, step)`. If your
  cube includes a time axis you don't want to tile, take the spatial
  slice first (`da.isel(time=k)`) or use `TemporalPatcher` for the time
  axis. The time-axis patching path is documented in
  [`recipes/temporal-stencils.md`](temporal-stencils.md).
- **In-memory cache only.** Content-addressed caching across sessions
  is tracked at [#24](https://github.com/jejjohnson/geotoolz/tree/main/packages/geotoolz-patcher/issues/24);
  this PR ships only the index-keyed in-memory variant.
- **No xrpatcher `dims_labels` auto-discovery on reconstruct.**
  `merge_to_xarray` uses the field's coord schema; specify it via the
  field's `with_data` rather than letting the reconstruct guess.

## See also

- [`IndexedPatchView`](../api/reference.md) — the random-access wrapper.
- [`recipes/temporal-stencils.md`](temporal-stencils.md) — time-axis
  counterpart (when you want `TimeStencil('-9h', '3h', '3h')` semantics).
- ADR-005 in [`docs/decisions.md`](../decisions.md) — design rationale.
- `xrpatcher` upstream: https://github.com/jejjohnson/xrpatcher.
