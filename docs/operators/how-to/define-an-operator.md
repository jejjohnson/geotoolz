# Define an operator

Write your own step and it composes with every built-in: in a
`Sequential`, in a `Graph`, and in `get_config()`. This page is the one
walkthrough; the model behind it is in [Concepts](../concepts.md#the-operator).

## Write the operator

Subclass `geotoolz.Operator` (pipekit's), store each keyword-only argument
under its own name, and do the work in `_apply`. Rewrap the result with
`geotoolz.carrier.wrap_like`, or `wrap_filled` when nodata pixels need the
output's fill, then run `geotoolz.testing.check_operator`.

```python
import numpy as np
from georeader.geotensor import GeoTensor
from rasterio.transform import from_origin

import geotoolz as gz
from geotoolz.carrier import carried_fill, wrap_filled
from geotoolz.testing import check_operator


class Scale(gz.Operator):
    """Multiply digital numbers by a scale factor."""

    def __init__(self, *, scale: float = 1e-4) -> None:
        self.scale = scale                                          # same name as the argument

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        out: np.ndarray = np.asarray(gt, dtype=np.float32) * self.scale  # (C, H, W) any → (C, H, W) float32
        # Rewrap like the input and write the output's fill into its nodata pixels.
        return wrap_filled(gt, out, fill_value_default=carried_fill(gt, out.dtype))


rng: np.random.Generator = np.random.default_rng(0)
dn: np.ndarray = rng.integers(1, 4000, size=(4, 32, 32), dtype=np.uint16)  # (4, 32, 32) uint16
dn[:, :2, :2] = 0                                                         # a few nodata pixels
scene: GeoTensor = GeoTensor(
    dn, transform=from_origin(750_000, 4_350_000, 10, 10), crs="EPSG:32610",
    fill_value_default=0, attrs={"band_names": ["B2", "B3", "B4", "B8"]},
)

op: Scale = Scale(scale=1e-4)
reflectance: GeoTensor = check_operator(op, scene)                        # (4, 32, 32) uint16 → (4, 32, 32) float32
assert op.get_config() == {"scale": 1e-4}
assert gz.Operator.from_state(op.state).get_config() == op.get_config()
```

`check_operator` fails naming the first broken rule. It checks:

- the constructor is keyword-only, and `get_config()` is JSON that
  rebuilds an equal operator (or the operator is `forbid_in_yaml`);
- the operator works on graph `Input` nodes;
- a GeoTensor in gives a GeoTensor out on the same grid, and a plain array
  gives a plain array;
- `attrs` is a fresh dict whose per-band keys match the output bands;
- the fill suits the output dtype, the input's nodata stays nodata, and
  the input is left untouched;
- a `(T, C, H, W)` stack matches the per-frame results, or is rejected
  with an error naming the operator.

## Follow the rules

| Rule | Why |
|---|---|
| Keyword-only `__init__(self, *, ...)`, each argument stored under its own name | `get_config()` is derived from it; configs map to arguments by name. |
| Hold configuration only, never a carrier | A second raster is a positional `_apply` argument, so a `Graph` can wire it in. |
| Implement `_apply`, never `__call__` | `__call__` switches between eager runs and graph recording. |
| Read with `np.asarray(gt)`, rewrap with `wrap_like` | Plain arrays and GeoTensors both work; CRS, transform and attrs carry over. |
| Never mutate the input | The same scene may feed other branches. |
| Declare the output fill | `False` for masks, `0` for labels and counts, `NaN` for new float quantities, `carried_fill(gt, dtype)` for carried values. |
| Use the shared parameter names | `red` / `nir` / `qa_band` for one band, `bands`, `fill_value`, `window`, `seed`, `eps`. |

The full parameter vocabulary is in
[`packages/geotoolz/AGENTS.md`](https://github.com/jejjohnson/geotoolz/blob/main/packages/geotoolz/AGENTS.md#operator-parameter-vocabulary).

## Collapse bands and judge nodata

An index reads two bands and returns one value per pixel. Resolve band
references by name or position, leave invalid pixels as `NaN`, and
decorate `_apply` with `over_frames` so a `(T, C, H, W)` stack runs frame
by frame.

```python
import numpy as np
from georeader.geotensor import GeoTensor
from rasterio.transform import from_origin

import geotoolz as gz
from geotoolz.carrier import mask_invalid_to_nan, over_frames, resolve_band, wrap_like
from geotoolz.testing import check_operator


class NormalizedDifference(gz.Operator):
    """(a - b) / (a + b + eps) for two bands; NaN where either is nodata."""

    def __init__(self, *, a: int | str = 3, b: int | str = 2, eps: float = 1e-10) -> None:
        self.a, self.b, self.eps = a, b, eps

    @over_frames
    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        x: np.ndarray = mask_invalid_to_nan(gt)                       # (C, H, W) any → (C, H, W) float · NaN = nodata
        a: np.ndarray = x[resolve_band(gt, self.a)]                   # (H, W) float
        b: np.ndarray = x[resolve_band(gt, self.b)]                   # (H, W) float
        return wrap_like(gt, (a - b) / (a + b + self.eps), fill_value_default=np.nan)


rng: np.random.Generator = np.random.default_rng(0)
dn: np.ndarray = rng.integers(1, 4000, size=(4, 32, 32), dtype=np.uint16)  # (4, 32, 32) uint16
dn[:, :2, :2] = 0                                                         # a few nodata pixels
scene: GeoTensor = GeoTensor(
    dn, transform=from_origin(750_000, 4_350_000, 10, 10), crs="EPSG:32610",
    fill_value_default=0, attrs={"band_names": ["B2", "B3", "B4", "B8"]},
)

ndvi: GeoTensor = check_operator(NormalizedDifference(a="B8", b="B4"), scene)  # (4, 32, 32) uint16 → (32, 32) float · NaN = nodata
assert np.isnan(ndvi.values[:2, :2]).all()
```

The built-in `gz.NDVI` has the same shape. Before writing a step, search
the [capability index](../../capabilities.md): it may already exist.

## Hold a live object

An operator that holds a callable, model or open handle cannot rebuild
from JSON. Set `forbid_in_yaml = True`: its config becomes a debug repr,
and `from_state` refuses to rebuild it.

```python
from collections.abc import Callable

import numpy as np

import geotoolz as gz
from geotoolz.carrier import wrap_like


class ApplyFunction(gz.Operator):
    """Apply an elementwise function to every pixel."""

    forbid_in_yaml = True

    def __init__(self, *, fn: Callable[[np.ndarray], np.ndarray]) -> None:
        self.fn = fn

    def _apply(self, gt: np.ndarray) -> np.ndarray:
        return wrap_like(gt, self.fn(np.asarray(gt)))


x: np.ndarray = np.random.default_rng(0).random((3, 8, 8))  # (3, 8, 8) float64
y: np.ndarray = ApplyFunction(fn=np.sqrt)(x)                  # (3, 8, 8) float64 → (3, 8, 8) float64
```

For a one-off function, pipekit's `gz.Lambda(fn)` does the same without a
class.

## Write a terminal operator

An operator that returns something other than a carrier, such as `None`
after a write, sets `_terminal = True`. `Sequential` then accepts it only
as the last step.

```python
import tempfile
from pathlib import Path

import numpy as np

import geotoolz as gz


class SaveNpy(gz.Operator):
    """Write the array to a .npy file; returns None."""

    _terminal = True

    def __init__(self, *, path: str) -> None:
        self.path = path

    def _apply(self, gt: np.ndarray) -> None:
        np.save(self.path, np.asarray(gt))


path: str = str(Path(tempfile.mkdtemp()) / "ndvi.npy")
x: np.ndarray = np.random.default_rng(0).random((4, 8, 8))                  # (4, 8, 8) float64
pipeline: gz.Sequential = gz.NDVI(red=2, nir=3) | SaveNpy(path=path)        # (4, 8, 8) float64 → None
pipeline(x)
saved: np.ndarray = np.load(path)                                          # (8, 8) float64
```

The built-in writers (`gz.WriteGeoTIFF`, `gz.WriteCOG`, `gz.WriteZarr`)
are terminal the same way. To write mid-chain and keep the carrier
flowing, use `gz.Sink(write_fn)`, which returns its input.

## Learn state with fit and transform

An operator that learns from data never overwrites its constructor
attributes in `_apply`. It follows the
[fitted-operator contract](../concepts.md#fitted-operators):

- `fit(x) -> self` writes trailing-underscore attributes (`mean_`);
- `transform(x)` only reads them;
- `_apply` calls `transform`, after `geotoolz.carrier.fit_once` when it
  learns on the first call, so concurrent first calls fit once.

Fitted attributes are not constructor arguments, so they stay out of
`get_config()` with no extra code.

## Further reading

- [Branching pipelines](branching-pipelines.md): combine your operator
  with others in a `Graph`, `Branch` or `Switch`.
- [Carrier helpers](../api/carrier.md): every helper in
  `geotoolz.carrier` and `geotoolz.testing`.
