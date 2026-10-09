# geotoolz

> **Compose remote-sensing pipelines like you compose functions.**
> Sentinel-2 to cloud-free NDVI in a handful of small operators, with the
> same code shape as your unit tests.

## Where it comes from

Remote-sensing code tends to grow into long functions that mix band
indexing, nodata handling, rescaling and georeferencing. Nothing in them
can be reused or tested alone.

`geotoolz` cuts each step into an **Operator**: a typed callable with a
keyword-only constructor. It takes a georeader `GeoTensor` (or a plain
`ndarray`) and returns the same kind of carrier, with its CRS, transform,
band names and fill value intact.

Chain operators with `|` into a `Sequential`, or wire them into a named
`Graph` when the pipeline branches. Both are operators too, so pipelines
nest, and `get_config()` turns any pipeline into JSON. The composition
core comes from [pipekit](https://github.com/jejjohnson/pipekit); about
275 remote-sensing operators sit on top. How geotoolz meets the other
packages is on [How the packages interlock](../geostack.md).

![Sequential and Graph composition of geotoolz operators, with the output shape and dtype of every step](../assets/diagrams/geotoolz-composition.png)

## Install

```bash
pip install geotoolz                 # every operator family
pip install 'geotoolz[patch]'        # + tile → operate → stitch with geopatcher
```

Pre-PyPI, install from a clone; see the [stack install](../index.md#install).
Backends that only a few operators use are extras. Those operators import
without them and raise an `ImportError` naming the extra when called.

| Extra | Pulls in | Needed for |
|---|---|---|
| `viz` | `matplotlib` | `viz.ApplyColormap` (named matplotlib colormaps) |
| `learn` | `scikit-learn`, `joblib` | `learn.SklearnOp` / `Pixelwise*` imputer NaN strategies and `save_state` / `load_state` / `state_path=`; `plume` DBSCAN clump counting |
| `zarr` | `zarr>=3` | `io.WriteZarr` |
| `hdf5` | `h5py` | `io.ReadHDF` on HDF5 |
| `hdf4` | `pyhdf` | `io.ReadHDF` on HDF4 |
| `netcdf` | `netCDF4` | `io.ReadNetCDF` |
| `vector-cube` | `xvec` (+ `xarray`) | `geom.coregister.RasterToPoints` / `PointsToRaster`, bilinear point sampling |
| `hydra` | `hydra-zen` | YAML `builds()` / `instantiate()` round-trips |
| `patch` | `geotoolz-patcher[pipekit]` | `geotoolz.patch_ops` (tile → map → stitch, label-aware samplers) |

## Quickstart

Cloud-free NDVI from a Sentinel-2 L2A scene, first as a chain, then as a
graph that masks clouds with the SCL band. The scene here is synthetic: 12
reflectance bands plus SCL on a real UTM grid, named through
`attrs["band_names"]` as the geoproducts readers and geocatalog loaders do.

```python
import numpy as np
from georeader.geotensor import GeoTensor
from rasterio.transform import from_origin

import geotoolz as gz

L2A: list[str] = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
rng: np.random.Generator = np.random.default_rng(0)
dn: np.ndarray = rng.integers(200, 4000, size=(12, 64, 64), dtype=np.uint16)  # (12, 64, 64) uint16
scl: np.ndarray = rng.choice([4, 5, 8, 9], size=(1, 64, 64)).astype(np.uint16) # (1, 64, 64) uint16 · 8, 9 = cloud
scene: GeoTensor = GeoTensor(
    np.concatenate([dn, scl]),                                                 # (13, 64, 64) uint16
    transform=from_origin(750_000, 4_350_000, 10, 10),                         # 10 m pixels
    crs="EPSG:32610",                                                          # UTM zone 10N
    fill_value_default=0,
    attrs={"band_names": [*L2A, "SCL"]},
)

# Sequential — a linear chain, built with |
ndvi_pipeline: gz.Sequential = (
    gz.SelectBands(bands=L2A)                                                  # (13, 64, 64) uint16 → (12, 64, 64) uint16
    | gz.DNToReflectance(scale=1e-4)                                           # (12, 64, 64) uint16 → (12, 64, 64) float64
    | gz.NDVI(red="B4", nir="B8")                                              # (12, 64, 64) float64 → (64, 64) float64
)
ndvi: GeoTensor = ndvi_pipeline(scene)                                         # (13, 64, 64) uint16 → (64, 64) float64 · NaN = no data
config: dict = ndvi_pipeline.get_config()  # {"operators": [{"class": "SelectBands", "config": {...}}, ...]}

# Graph — the same steps, plus a cloud mask on a second branch
x: gz.Input = gz.Input("scene")
reflectance: gz.Node = gz.DNToReflectance(scale=1e-4)(gz.SelectBands(bands=L2A)(x))  # (12, 64, 64) float64
ndvi_node: gz.Node = gz.NDVI(red="B4", nir="B8")(reflectance)                        # (64, 64) float64
cloud: gz.Node = gz.S2SCL(targets=["cloud_shadow", "cloud", "cirrus"])(x)           # (64, 64) bool · True = drop
clean: gz.Node = gz.ApplyMask()(ndvi_node, cloud)                                   # (64, 64) float64 · NaN under cloud

graph: gz.Graph = gz.Graph(inputs={"scene": x}, outputs={"ndvi": clean, "cloud": cloud})
out: dict[str, GeoTensor] = graph(scene=scene)  # {"ndvi": (64, 64) float64, "cloud": (64, 64) bool}
assert out["ndvi"].crs == scene.crs and out["ndvi"].transform == scene.transform
```

Both results keep the scene's CRS and transform. NDVI declares `NaN` as
its fill because it is a new float quantity; the mask uses `False`.

## Is this the right tool?

| You want to… | Use |
|---|---|
| run one array operation on one scene | rasterio + numpy directly |
| reuse a remote-sensing step across scenes and scripts | an `Operator` ([define one](how-to/define-an-operator.md)) |
| chain two to six steps | `Sequential` (`a \| b \| c`) |
| branch, fan in, or return several named outputs | `Graph` ([branching pipelines](how-to/branching-pipelines.md)) |
| run a model tile by tile over a large raster | `geotoolz.patch_ops` ([tile → operate → stitch](patch_ops.md)) |
| find and load scenes from STAC or local files | [geocatalog](../catalog/index.md), then pass the `GeoTensor` in |

## What's inside

Every public operator is importable from the top level (`gz.NDVI`) and
from its family (`gz.indices.NDVI`). Each family also exports its pure
numpy primitives (`gz.indices.ndvi`).

| Namespace | What it does | Reference |
|---|---|---|
| core | `Operator`, `Sequential`, `Graph`, `Input`, `Branch`, `Switch`, `Fanout`, `Tap`, `Snapshot`, … (from pipekit) and `ModelOp` | [Core](api/core.md) |
| `geotoolz.carrier`, `geotoolz.testing` | helpers that keep the `GeoTensor` contract, and `check_operator` | [Carrier](api/carrier.md) |
| `radiometry` | DN ↔ radiance ↔ reflectance, sun geometry, brightness temperature, DOS1, stretches | [Radiometry](api/radiometry.md) |
| `indices` | spectral indices: NDVI, EVI, NDWI, NBR, NDSI, … | [Indices](api/indices.md) |
| `qa` | cloud and quality masks from QA bits, SCL classes and sensor presets | [QA](api/qa.md) |
| `mask` | geometry and DEM masks, mask morphology and algebra, `ApplyMask` | [Mask](api/mask.md) |
| `spectral` | band selection, stacking, band math, spectral resampling | [Spectral](api/spectral.md) |
| `geom` (+ `geom.coregister`) | reproject, resample, crop, tile, rasterise, swath geometry, registration | [Geometry](api/geom.md) |
| `compositing` | temporal composites and multi-source fusion | [Compositing](api/compositing.md) |
| `restore` | denoising, despeckling, destriping, gap filling, MNF | [Restore](api/restore.md) |
| `segment` | thresholds, superpixels, watershed and other skimage segmentations | [Segment](api/segment.md) |
| `measure` | connected components, region properties, contours | [Measure](api/measure.md) |
| `feature` | edges, blobs, corners, Hough transforms, local features | [Feature](api/feature.md) |
| `plume` | methane and CO₂ plume retrieval, masks, footprints, mass and flux | [Plume](api/plume.md) |
| `matched_filter` | hyperspectral matched filters and background statistics | [Matched filter](api/matched_filter.md) |
| `augment` | augmentations that respect georeferencing and physical bands | [Augment](api/augment.md) |
| `normalize` | per-band scalers, non-linear scaling, histogram stretch and matching | [Normalize](api/normalize.md) · [how-to](normalization.md) |
| `learn` | scikit-learn estimators as pixel-wise operators | [Learn](api/learn.md) |
| `einx` | einstein-notation tensor operations that keep georeferencing | [Einx](api/einx.md) |
| `viz` | composites, display stretches, colormaps, hillshade, overlays | [Viz](api/viz.md) |
| `io` | window readers, HDF and NetCDF readers, GeoTIFF / COG / Zarr writers | [IO](api/io.md) · [how-to](io.md) |
| `patch_ops` | geopatcher as operators: `GridSampler`, `ApplyToChips`, `MergePatches`, label-aware samplers | [Tile → operate → stitch](patch_ops.md) |

## Next steps

- **[Concepts](concepts.md)**: the Operator contract, `Sequential` and
  `Graph`, control flow, observers, fitted operators, band names, mask
  polarity.
- **How-to guides**: [define an operator](how-to/define-an-operator.md) ·
  [branching pipelines](how-to/branching-pipelines.md) ·
  [tile → operate → stitch](patch_ops.md) ·
  [normalise for training and inference](normalization.md) ·
  [read HDF and NetCDF](io.md).
- **Tutorial**: [operators on Sentinel-2 over Lake Tahoe](notebooks/operators_lake_tahoe.ipynb).
- **Reference**: start at [Core](api/core.md); the
  [capability index](../capabilities.md) lists every public name once.
- **Changelog**: [`packages/geotoolz/CHANGELOG.md`](https://github.com/jejjohnson/geotoolz/blob/main/packages/geotoolz/CHANGELOG.md).
