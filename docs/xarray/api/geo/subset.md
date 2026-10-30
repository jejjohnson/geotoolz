# Subsetting, Regions & Masks

Carve a dataset down to a region of interest — by bounding box, time
window, named region, or boolean predicate — and add land / ocean /
country masks. Named regions come from a small registry that also reads
GeoJSON polygons.

## Subsetting

::: xrops.geom.operators.SubsetBBox

::: xrops.geom.operators.SubsetTime

::: xrops.geom.operators.SubsetToRegion

::: xrops.geom.operators.SelectVariables

## Masks

::: xrops.mask.operators.AddLandMask

::: xrops.mask.operators.AddOceanMask

::: xrops.mask.operators.AddCountryMask

::: xrops.mask.operators.ApplyMask

## Functional primitives (Layer 0)

These pure functions back the operators above; each takes `xr.DataArray`/`xr.Dataset` and a `dim:` argument.

::: xrops.geom.subset_bbox

::: xrops.geom.subset_time

::: xrops.geom.subset_to_region

::: xrops.geom.subset_where

::: xrops.geom.select_variables

::: xrops.mask.add_land_mask

::: xrops.mask.add_ocean_mask

::: xrops.mask.add_country_mask

::: xrops.mask.apply_mask

## Region registry

`REGIONS` is the built-in named-region table; `RegionSpec` is the frozen
bounding-box descriptor. The helpers below resolve a region from a name, a
dict, or a GeoJSON polygon.

::: xrops.geom.REGIONS

::: xrops.geom.RegionSpec

::: xrops.geom.bbox_region

::: xrops.geom.custom_region

::: xrops.geom.resolve_region

::: xrops.geom.region_from_dict

::: xrops.geom.region_to_dict

::: xrops.geom.load_region_file

::: xrops.geom.polygon_from_geojson
