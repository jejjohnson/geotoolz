# Subsetting, Regions & Masks

Carve a dataset down to a region of interest — by bounding box, time
window, named region, or boolean predicate — and add land / ocean /
country masks. Named regions come from a small registry that also reads
GeoJSON polygons.

## Subsetting

::: xrops.geo.operators.SubsetBBox

::: xrops.geo.operators.SubsetTime

::: xrops.geo.operators.SubsetToRegion

::: xrops.geo.operators.SelectVariables

## Masks

::: xrops.geo.operators.AddLandMask

::: xrops.geo.operators.AddOceanMask

::: xrops.geo.operators.AddCountryMask

::: xrops.geo.operators.ApplyMask

## Functional primitives (Layer 0)

These pure functions back the operators above; each takes `xr.DataArray`/`xr.Dataset` and a `dim:` argument.

::: xrops.geo.subset_bbox

::: xrops.geo.subset_time

::: xrops.geo.subset_to_region

::: xrops.geo.subset_where

::: xrops.geo.select_variables

::: xrops.geo.add_land_mask

::: xrops.geo.add_ocean_mask

::: xrops.geo.add_country_mask

::: xrops.geo.apply_mask

## Region registry

`REGIONS` is the built-in named-region table; `RegionSpec` is the frozen
bounding-box descriptor. The helpers below resolve a region from a name, a
dict, or a GeoJSON polygon.

::: xrops.geo.REGIONS

::: xrops.geo.RegionSpec

::: xrops.geo.bbox_region

::: xrops.geo.custom_region

::: xrops.geo.resolve_region

::: xrops.geo.region_from_dict

::: xrops.geo.region_to_dict

::: xrops.geo.load_region_file

::: xrops.geo.polygon_from_geojson
