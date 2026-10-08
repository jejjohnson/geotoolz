# geotoolz-products

Readers for Earth-observation data products (import name `geoproducts`). Each
reader turns a mission's or provider's product into a georeader `GeoData` /
`GeoTensor`, so the operators (`geotoolz`), the patchers (`geopatcher`) and the
catalog loaders (`geocatalog`) consume it unchanged.

```bash
pip install geotoolz-products                  # readers (georeader only)
pip install 'geotoolz-products[obstore]'       # pooled cloud byte-range reads
pip install 'geotoolz-products[operators]'     # sensor presets (geotoolz operators)
pip install 'geotoolz-products[carbonmapper]'  # Carbon Mapper plume catalogue + STAC
```

The package depends on georeader, never on `geotoolz`: product readers churn
with the missions and APIs behind them and carry their own credentials and
client libraries, so they release independently of the operator library.

- [Adding a new sensor reader](sensor-readers.md) — the per-sensor contract.
- [Carbon Mapper](carbonmapper.md) — plume catalogue, sources and plume / scene rasters.
- [API reference](api.md)
