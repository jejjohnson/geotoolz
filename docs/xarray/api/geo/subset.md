# Subsetting, Regions & Masks

Carve a dataset down to a region of interest — by bounding box, time
window, named region, or boolean predicate — and add land / ocean /
country masks. Named regions come from a small registry that also reads
GeoJSON polygons.

## Subsetting

::: xrtoolz.geo.operators.SubsetBBox

::: xrtoolz.geo.operators.SubsetTime

::: xrtoolz.geo.operators.SubsetToRegion

::: xrtoolz.geo.operators.SelectVariables

## Masks

::: xrtoolz.geo.operators.AddLandMask

::: xrtoolz.geo.operators.AddOceanMask

::: xrtoolz.geo.operators.AddCountryMask

::: xrtoolz.geo.operators.ApplyMask

## Functional primitives (Layer 0)

These pure functions back the operators above; each takes `xr.DataArray`/`xr.Dataset` and a `dim:` argument.

::: xrtoolz.geo.subset_bbox

::: xrtoolz.geo.subset_time

::: xrtoolz.geo.subset_to_region

::: xrtoolz.geo.subset_where

::: xrtoolz.geo.select_variables

::: xrtoolz.geo.add_land_mask

::: xrtoolz.geo.add_ocean_mask

::: xrtoolz.geo.add_country_mask

::: xrtoolz.geo.apply_mask

## Region registry

`REGIONS` is the built-in named-region table; `RegionSpec` is the frozen
bounding-box descriptor. The helpers below resolve a region from a name, a
dict, or a GeoJSON polygon.

::: xrtoolz.geo.REGIONS

::: xrtoolz.geo.RegionSpec

::: xrtoolz.geo.bbox_region

::: xrtoolz.geo.custom_region

::: xrtoolz.geo.resolve_region

::: xrtoolz.geo.region_from_dict

::: xrtoolz.geo.region_to_dict

::: xrtoolz.geo.load_region_file

::: xrtoolz.geo.polygon_from_geojson
