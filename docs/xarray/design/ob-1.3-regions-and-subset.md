# OB-1.3 — `RegionSpec` registry + regionmask-backed `subset_to_region`

**Source survey item:** [oceanbench-survey.md §B.1.3](oceanbench-survey.md)
**Status:** proposed
**Maps to upstream:** `regions.py:{BoundingBox, RegionSpec, subset_dataset_to_region, custom_region, region_from_dict, load_region_file}` from
[`mercator-ocean/oceanbench/oceanbench/core/regions.py`](https://github.com/mercator-ocean/oceanbench/blob/main/oceanbench/core/regions.py).

---

## 1. Motivation

xrtoolz today has two parallel, incomplete region stories:

1. [`subset_bbox(lon_bnds, lat_bnds)`](https://github.com/jejjohnson/xrtoolz/blob/main/src/xrtoolz/geo/_src/subset.py#L11)
   — boolean mask + `where(drop=True)`. Works for simple bounding
   boxes on rectilinear grids. **No antimeridian wrap-around, no
   0–360 vs −180–180 auto-detect, no support for polygon regions.**
2. [`viz/_src/projections.py:PRESETS`](https://github.com/jejjohnson/xrtoolz/blob/main/src/xrtoolz/viz/_src/projections.py#L24)
   — a hard-coded dict mapping region names (`"gulf_stream"`,
   `"north_atlantic"`, `"mediterranean"`, ...) to cartopy projection
   class names + `(lon_min, lon_max, lat_min, lat_max)` extents.
   **Used only for plotting.**

Mercator's [`regions.py`](https://github.com/mercator-ocean/oceanbench/blob/main/oceanbench/core/regions.py)
ships a complete regional-subset toolkit (`BoundingBox`, `RegionSpec`,
named registry, JSON serialization, antimeridian-aware
`subset_dataset_to_region`).

The natural unification: **a single `RegionSpec` registry in
`xrtoolz.geo.regions` consumed by both subset and viz**. Each entry
carries its bounds (for subsetting) and projection (for plotting).
The cartopy `PRESETS` becomes a derived view of the same registry.

For the geometry primitive itself, we lean on **`regionmask.Regions`**
(already a dependency, used in ODC-1.4 `scores_by_region`).
`regionmask` already handles:

- Rectangles via `shapely.geometry.box`.
- Antimeridian-spanning rectangles via `MultiPolygon` of two boxes.
- Polygons of arbitrary shape.
- 0–360 vs −180–180 auto-detection via `wrap_lon=True`.
- Gridded *and* scattered point datasets via the same
  `Regions.mask(lon, lat)` call.
- Pre-built region sets — `regionmask.defined_regions.*`.

This means we don't reimplement antimeridian logic, polygon
containment, or convention auto-detection. We delegate to regionmask
and ship a thin wrapper that adds named-region metadata + cartopy
projection.

## 2. User stories

### 2.1 Subset to a named region (primary)

```python
import xarray as xr
from xrtoolz.geo import subset_to_region

ds_gs = subset_to_region(ds, "gulf_stream")
```

### 2.2 Subset with a custom rectangle (antimeridian-aware)

```python
from xrtoolz.geo import subset_to_region, custom_region

# A region spanning the dateline (170°E → -170°E)
date_line_region = custom_region(
    id="date_line",
    display_name="Date-line crossing",
    lat_min=-30, lat_max=30,
    lon_min=170, lon_max=-170,           # wrap-around — handled
)
ds_dl = subset_to_region(ds, date_line_region)
```

### 2.3 Subset with a regionmask polygon set

```python
import regionmask
from xrtoolz.geo import subset_to_region

ocean_basins = regionmask.defined_regions.natural_earth_v5_0_0.ocean_basins_50
ds_atl = subset_to_region(ds, ocean_basins["North Atlantic Ocean"])
```

### 2.4 Polygon from GeoJSON (no geopandas needed)

```python
from xrtoolz.geo import polygon_from_geojson, subset_to_region

custom_polygon = polygon_from_geojson("path/to/region.geojson", name="study_area")
ds_subset = subset_to_region(ds, custom_polygon)
```

### 2.5 Plot the same named region

```python
from xrtoolz.viz.validation import SpatialMapPanel

panel = SpatialMapPanel(var="ssh", projection="gulf_stream")  # unchanged API
panel(ds_gs)
```

`"gulf_stream"` resolves through the same registry the subset uses —
no duplicate definition.

### 2.6 As Layer-1 Operators inside a Sequential

```python
from xrtoolz.core import Sequential
from xrtoolz.geo import SubsetToRegion, RenameFromCFStandardNames

pipeline = Sequential([
    RenameFromCFStandardNames(),         # OB-1.2
    SubsetToRegion(region="gulf_stream"),
    # ... downstream metrics
])
```

### 2.7 JSON round-trip

```python
from xrtoolz.geo import region_to_dict, region_from_dict, load_region_file

d = region_to_dict(my_region)
recovered = region_from_dict(d)
from_disk = load_region_file("regions/gulf_stream.json")
```

## 3. What we already have / what's missing

| Capability | Current | This proposal |
|---|---|---|
| Simple bbox subset | [`geo/_src/subset.py:subset_bbox`](https://github.com/jejjohnson/xrtoolz/blob/main/src/xrtoolz/geo/_src/subset.py#L11) | unchanged (kept for compat) |
| Antimeridian wrap-around | — | **add** (via regionmask MultiPolygon) |
| 0–360 vs −180–180 auto-detect | — | **add** (via `regionmask.Regions.mask(wrap_lon=True)`) |
| Gridded vs scattered point handling | partial (`where(drop=True)`) | **add** (uniform via regionmask) |
| Polygon-region subset | — | **add** (regionmask passthrough) |
| Named region registry | viz `PRESETS` only | **add** `xrtoolz.geo.REGIONS` (consumed by both subset + viz) |
| `RegionSpec` / `BoundingBox` dataclasses | — | **add** `RegionSpec` (thin viz-metadata wrapper around `regionmask.Regions`) |
| `bbox_region` builder (rect → Regions) | — | **add** |
| `polygon_from_geojson` (shapely-backed) | — | **add** (~10 LOC) |
| JSON serialization | — | **add** `region_to_dict` / `region_from_dict` / `load_region_file` |
| `SubsetToRegion` Operator | — | **add** |
| Pre-built polygon region sets | partial (regionmask available) | document `regionmask.defined_regions.*` as canonical source |

## 4. Design

### 4.1 The geometry primitive: `regionmask.Regions`

We do **not** introduce a parallel geometry primitive. `regionmask.Regions`
covers everything:

- **Rectangles**: wrap a `shapely.geometry.box(lon_min, lat_min, lon_max, lat_max)`.
- **Antimeridian rectangles**: `shapely.geometry.MultiPolygon` of two
  boxes, split at ±180°.
- **Polygons**: any shapely geometry.
- **Pre-built named sets**: `regionmask.defined_regions.natural_earth_v5_0_0.{land_110, ocean_basins_50}`,
  `regionmask.defined_regions.ar6.ocean`, SREX, etc.
- **GeoJSON ingest**: shapely-backed via `polygon_from_geojson`.

This means our subset code has a **single dispatch path**:

```python
def subset_to_region(ds, region, ...):
    if isinstance(region, str):
        region = resolve_region(region)
    if isinstance(region, RegionSpec):
        region = region.regions                    # unwrap to regionmask.Regions
    # region is now regionmask.Regions
    mask = region.mask(ds[lon], ds[lat]).notnull()
    return ds.where(mask, drop=True)
```

No hand-rolled antimeridian / 0-360 / point-vs-grid branching.

### 4.2 `RegionSpec` — viz-metadata wrapper

```python
# src/xrtoolz/geo/_src/regions.py — new module
@dataclass(frozen=True)
class RegionSpec:
    """Named region pairing geometry with viz metadata.

    The geometry primitive is :class:`regionmask.Regions` — handles
    rectangles, antimeridian-spanning boxes, polygons, and pre-built
    region sets uniformly.
    """
    id: str
    display_name: str
    regions: regionmask.Regions
    projection: str | None = None      # cartopy CRS class name
```

`RegionSpec` is **not** the geometry primitive. It's a thin pairing of
a `regionmask.Regions` (geometry) with metadata (`id`, `display_name`,
`projection`). Users who don't care about names / projections work with
`regionmask.Regions` directly.

### 4.3 Builders

```python
def bbox_region(
    *,
    id: str,
    name: str,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
) -> regionmask.Regions:
    """Construct a regionmask.Regions from a rectangular bbox.

    Handles antimeridian wrap-around (lon_min > lon_max) by emitting
    a MultiPolygon split at the dateline.
    """
    if lon_min <= lon_max:
        poly = shapely.geometry.box(lon_min, lat_min, lon_max, lat_max)
    else:
        poly = shapely.geometry.MultiPolygon([
            shapely.geometry.box(lon_min, lat_min, 180.0, lat_max),
            shapely.geometry.box(-180.0, lat_min, lon_max, lat_max),
        ])
    return regionmask.Regions(outlines=[poly], names=[name], abbrevs=[id], name=id)


def custom_region(
    *,
    id: str,
    display_name: str,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    projection: str | None = "PlateCarree",
) -> RegionSpec:
    """Build a named, validated RegionSpec from a rectangular bbox."""
    _validate_id(id); _validate_bounds(lat_min, lat_max, lon_min, lon_max)
    return RegionSpec(
        id=id,
        display_name=display_name,
        regions=bbox_region(id=id, name=display_name,
                            lat_min=lat_min, lat_max=lat_max,
                            lon_min=lon_min, lon_max=lon_max),
        projection=projection,
    )


def polygon_from_geojson(
    data: dict | str | Path,
    *,
    name: str = "custom",
) -> regionmask.Regions:
    """Construct a Regions from a GeoJSON Feature/Polygon/MultiPolygon.

    Accepts a GeoJSON dict, a path to a .geojson file, or a JSON
    string. Uses shapely.geometry.shape internally; no geopandas dep.
    """
    if isinstance(data, (str, Path)) and (Path(str(data))).exists():
        with open(data) as f:
            data = json.load(f)
    elif isinstance(data, str):
        data = json.loads(data)
    geometry = data.get("geometry", data) if isinstance(data, dict) else data
    poly = shapely.geometry.shape(geometry)
    return regionmask.Regions(outlines=[poly], names=[name],
                              abbrevs=[name], name=name)
```

### 4.4 Built-in registry

Mirrors the existing viz `PRESETS` to maintain backward compatibility:

```python
REGIONS: dict[str, RegionSpec] = {
    "global": RegionSpec(
        id="global", display_name="Global",
        regions=bbox_region(id="global", name="Global",
                            lat_min=-90, lat_max=90, lon_min=-180, lon_max=180),
        projection="Robinson",
    ),
    "north_atlantic": RegionSpec(
        id="north_atlantic", display_name="North Atlantic",
        regions=bbox_region(id="north_atlantic", name="North Atlantic",
                            lat_min=10, lat_max=65, lon_min=-80, lon_max=0),
        projection="PlateCarree",
    ),
    "gulf_stream": RegionSpec(
        id="gulf_stream", display_name="Gulf Stream",
        regions=bbox_region(id="gulf_stream", name="Gulf Stream",
                            lat_min=30, lat_max=45, lon_min=-80, lon_max=-50),
        projection="PlateCarree",
    ),
    "kuroshio": RegionSpec(
        id="kuroshio", display_name="Kuroshio",
        regions=bbox_region(id="kuroshio", name="Kuroshio",
                            lat_min=25, lat_max=45, lon_min=130, lon_max=180),
        projection="PlateCarree",
    ),
    "mediterranean": RegionSpec(
        id="mediterranean", display_name="Mediterranean",
        regions=bbox_region(id="mediterranean", name="Mediterranean",
                            lat_min=30, lat_max=46, lon_min=-6, lon_max=36),
        projection="PlateCarree",
    ),
    "ibi": RegionSpec(
        id="ibi", display_name="IBI",
        regions=bbox_region(id="ibi", name="IBI",
                            lat_min=26.17, lat_max=56.08, lon_min=-19.08, lon_max=5.08),
        projection="PlateCarree",
    ),
}


def resolve_region(region: str | RegionSpec) -> RegionSpec:
    """Look up a region by id, or return a RegionSpec unchanged."""
    if isinstance(region, RegionSpec):
        return region
    if region in REGIONS:
        return REGIONS[region]
    raise KeyError(
        f"unknown region {region!r}; available: {sorted(REGIONS)}. "
        f"Use custom_region(...) for ad-hoc bboxes."
    )
```

### 4.5 The subset op

```python
# src/xrtoolz/geo/_src/subset.py — alongside existing subset_bbox
def subset_to_region(
    ds: xr.Dataset,
    region: str | RegionSpec | regionmask.Regions,
    *,
    lon: str = "lon",
    lat: str = "lat",
    validate: bool = True,
) -> xr.Dataset:
    """Subset a Dataset to a named, custom, or polygon region.

    region:
      - str → registry lookup → RegionSpec (or KeyError)
      - RegionSpec → unwrapped to its regionmask.Regions
      - regionmask.Regions → used directly
    """
    if isinstance(region, str):
        region = resolve_region(region)
    if isinstance(region, RegionSpec):
        region = region.regions

    mask = region.mask(ds[lon], ds[lat]).notnull()
    if validate and not bool(mask.any().item()):
        raise ValueError(
            "Region does not overlap dataset coordinates. "
            "Pass validate=False to allow empty results."
        )
    return ds.where(mask, drop=True)
```

### 4.6 Performance fast-path (deferred)

For extreme-scale grids (global 1/12°+), `regionmask.Regions.mask`
on a single rectangular box has overhead from per-point shapely
containment (even with STRtree pre-filter) vs. raw numpy comparison.
For typical eval grids (Gulf Stream, Med Sea, North Atlantic,
~100k–1M points) this is unmeasurable. **Ship without a fast-path**;
add later if profiling demands:

```python
# Optional fast-path inside subset_to_region:
if _is_simple_bbox(region):
    return _bbox_mask_numpy(ds, region.bounds, lon, lat)
```

Not in v1 scope. Mentioned only so reviewers know the option exists.

### 4.7 JSON serialization

```python
def region_to_dict(region: RegionSpec) -> dict[str, Any]:
    """Serialize a RegionSpec to a JSON-safe dict.

    Geometry encoded as GeoJSON via shapely.geometry.mapping.
    """

def region_from_dict(data: dict[str, Any]) -> RegionSpec:
    """Inverse of region_to_dict.

    Accepts both our native format (with ``regions`` field as GeoJSON)
    and the simpler bbox-only form
    ``{"id": ..., "display_name": ..., "bounds": {"lat_min": ...}}``
    for ergonomic round-trips.
    """

def load_region_file(path: str | Path) -> RegionSpec:
    """Load a RegionSpec from a JSON file."""
```

GeoJSON encoding lets polygon regions round-trip cleanly. Rectangle-only
shorthand stays ergonomic.

### 4.8 Viz `PRESETS` bridge

[`viz/_src/projections.py`](https://github.com/jejjohnson/xrtoolz/blob/main/src/xrtoolz/viz/_src/projections.py)
already has:

```python
PRESETS: dict[str, dict[str, Any]] = {
    "gulf_stream": {"projection": "PlateCarree", "extent": (-80, -50, 30, 45)},
    ...
}
```

Replace with a derived view:

```python
def _build_presets() -> dict[str, dict[str, Any]]:
    from xrtoolz.geo.regions import REGIONS
    out = {}
    for region_id, spec in REGIONS.items():
        # Derive (lon_min, lon_max, lat_min, lat_max) from regions.bounds
        bounds = spec.regions.bounds_global    # (lon_min, lat_min, lon_max, lat_max)
        if spec.projection is not None:
            extent = (bounds[0], bounds[2], bounds[1], bounds[3]) if region_id != "global" else None
            out[region_id] = {"projection": spec.projection, "extent": extent}
    return out

PRESETS = _build_presets()
```

Backward compatible — the existing `_resolve_projection("gulf_stream")`
call site keeps working. Users can still bypass with explicit cartopy
projection objects.

### 4.9 Layer-1 Operator

```python
# src/xrtoolz/geo/operators.py
class SubsetToRegion(Operator):
    def __init__(
        self, *,
        region: str | RegionSpec | regionmask.Regions,
        lon: str = "lon",
        lat: str = "lat",
        validate: bool = True,
    ): ...
    def __call__(self, ds): return subset_to_region(...)
    def get_config(self): ...
```

`region` serializes via:
- `str` → emit as-is.
- `RegionSpec` → `region_to_dict`.
- `regionmask.Regions` → emit a flag plus `outlines` as GeoJSON.

## 5. Library leverage

| Need | Library |
|---|---|
| Region geometry primitive | `regionmask.Regions` (existing dep) |
| Antimeridian wrap-around | `shapely.geometry.MultiPolygon` (transitive via regionmask) |
| 0–360 vs −180–180 auto-detect | `Regions.mask(wrap_lon=True)` (regionmask) |
| Polygon mask machinery | `Regions.mask` (regionmask) |
| GeoJSON parsing | `shapely.geometry.shape` (transitive via regionmask) |
| Pre-built named region sets | `regionmask.defined_regions.*` (regionmask) |
| JSON serialization | stdlib `json`, GeoJSON via `shapely.geometry.mapping` |
| Dataclasses | stdlib `dataclasses` |

**No new top-level deps.** `shapely` and `regionmask` are already in
the dep tree (regionmask was added in ODC-1.4).

## 6. Public API surface

```python
# xrtoolz.geo (re-exports from xrtoolz.geo._src.regions and .subset)
from xrtoolz.geo import (
    # Dataclasses
    RegionSpec,
    # Built-in registry
    REGIONS,
    # Builders
    bbox_region,                   # (...) -> regionmask.Regions
    custom_region,                 # (...) -> RegionSpec
    polygon_from_geojson,          # (...) -> regionmask.Regions
    # Lookup
    resolve_region,
    # Subset
    subset_to_region,
    # JSON
    region_to_dict, region_from_dict, load_region_file,
    # Operator
    SubsetToRegion,
)
```

## 7. Tests

| Test | Asserts |
|---|---|
| `subset_to_region(ds, "gulf_stream")` on rectilinear grid | trims to bounds |
| `subset_to_region` on (point,)-dim scattered Dataset | works without explicit dispatch |
| `custom_region` with `lon_min > lon_max` (antimeridian) | yields MultiPolygon; mask correctly spans dateline |
| `bbox_region` with simple bounds | single Polygon |
| 0–360 dataset + `RegionSpec(lon_min=-80, lon_max=-50)` | regionmask `wrap_lon=True` handles it; correct mask |
| `subset_to_region` with `regionmask.defined_regions.natural_earth_v5_0_0.ocean_basins_50["North Atlantic Ocean"]` | polygon-region subset works |
| `subset_to_region` with `validate=True` and non-overlapping region | raises informative `ValueError` |
| `subset_to_region` with `validate=False` and non-overlapping region | returns empty Dataset |
| `polygon_from_geojson(geojson_dict)` | constructs Regions with one outline |
| `polygon_from_geojson(path)` | reads file, constructs Regions |
| `resolve_region("gulf_stream")` | returns registered RegionSpec |
| `resolve_region(RegionSpec(...))` | passes through |
| `resolve_region("unknown")` | raises informative `KeyError` listing options |
| `region_to_dict` / `region_from_dict` round-trip | identity (incl. polygon regions) |
| `load_region_file` from JSON | reads file, returns RegionSpec |
| Viz `_resolve_projection("gulf_stream")` | still works (no regression) |
| `SubsetToRegion(region="gulf_stream")` Operator round-trip | reconstructed produces identical output |
| `SubsetToRegion(region=custom_region(...))` round-trip | identity |

Target: ~18 cases.

## 8. Out of scope

- **Multi-region subset / set operations** (union / intersection of
  named regions) — `regionmask.Regions` supports multiple outlines
  natively if a user wants to compose; no dedicated helper in v1.
- **Time bounds in `RegionSpec`** — purely spatial for v1; users
  compose with `subset_time`.
- **YAML region files** — JSON only for v1 (no PyYAML dep).
- **Region-aware metric reductions** — covered by ODC-1.4
  `scores_by_region`, which already accepts `regionmask.Regions`.
- **Performance fast-path** for global 1/12°+ grids — defer; add only
  if profiling demands. Mentioned in §4.6 as a known design lever.
- **`geopandas` dep** — explicitly declined; `regionmask` + `shapely`
  cover the needs. Users with shapefiles can `geopandas.read_file()`
  in their own code and pass shapely geoms to `polygon_from_geojson`
  or directly to `regionmask.Regions(outlines=[...])`.
- **Mercator's strict `_validate_display_name` / regex `id` pattern**
  — adopted for `id` (snake-case enforced); `display_name` non-empty
  check only.

## 9. Effort

≈115 LOC implementation + ≈110 LOC tests. Single PR.

| Slice | LOC |
|---|---|
| `xrtoolz.geo.regions` (`RegionSpec`, `REGIONS`, builders, JSON helpers) | 70 |
| `subset_to_region` (single dispatch) | 25 |
| Viz `PRESETS` bridge | 10 |
| `SubsetToRegion` Operator | 20 |
| Tests | ~110 |
| Docs / re-exports | 10 |

## 10. Risks / open questions

1. **regionmask performance vs hand-rolled numpy mask.** For
   simple-rectangle-on-global-1/12° (~16M points), regionmask may be
   ~10× slower than direct numpy comparison due to per-point shapely
   containment (even with STRtree pre-filter). For typical eval grids
   (regional, ~100k–1M points), the difference is unmeasurable. Ship
   without fast-path; add `_is_simple_bbox` short-circuit in
   `subset_to_region` only if profiling demands.
2. **`projection` as a string field.** `RegionSpec.projection` stores
   the cartopy CRS *class name* (`"PlateCarree"`, `"Robinson"`) as a
   string, not a `ccrs.Projection` instance. Keeps `RegionSpec`
   serialization-friendly and avoids forcing a cartopy import in `geo`.
   Resolved to `ccrs.Projection` only when consumed by viz code.
3. **Backward compat with viz `PRESETS`.** `_resolve_projection(name)`
   keeps working. The `PRESETS` dict is now derived from `REGIONS`
   rather than hard-coded — but the lookup behaviour is identical.
4. **`bounds_global` from `regionmask.Regions`.** regionmask exposes
   `Regions.bounds_global` (a 4-tuple per region). The viz bridge
   uses it to derive `(lon_min, lon_max, lat_min, lat_max)` for
   cartopy `set_extent`. Verify this property exists across
   regionmask versions; if not, fall back to
   `Regions.polygons[0].bounds`.
5. **`subset_to_region` with multi-region `regionmask.Regions`.**
   When `region` has >1 outline (e.g. a 5-basin set), the mask is
   "any region matches". Document; users wanting per-region subsets
   loop over `regions[name]` and call once per region, or use
   ODC-1.4 `scores_by_region` which groups instead of subsetting.
6. **Where it lives.** `xrtoolz.geo.regions` (chosen). Alternative
   `xrtoolz.regions` top-level — rejected (regions are a geo
   concern; `xrpatcher` is generic and gets top-level placement,
   regions don't).
7. **`subset_bbox` retention.** Existing `subset_bbox(lon_bnds,
   lat_bnds)` stays for backward compat. Document
   `subset_to_region(custom_region(...))` as the recommended path
   for new code.
