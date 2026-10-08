# Adding a new product reader

Sensor and product integrations live under `geoproducts.<sensor>` and keep
the same small surface so each one can be audited in isolation.

```text
geoproducts/<sensor>/
  __init__.py      # Reader, BANDS, CONSTANTS, presets
  reader.py        # ProductReader subclass
  constants.py     # lazy calibration table accessors
  presets.py       # zero-argument geotoolz operators bound to the band names
  data/            # packaged calibration files
```

A reader should subclass `geoproducts.ProductReader`, implement the
metadata properties (`_crs`, `_transform`, `_shape`, `_dtype`, `_bands`,
`_fill_value`, `_track`), and provide `_read_window(window)`. Track `"A"`
means a clean affine grid; track `"B"` is reserved for sensors with irregular
geolocation.

Calibration data should be packaged below `data/`, kept small, and loaded via
`geoproducts._src.constants.load_csv()` or `load_json()`. These loaders are
cached, so importing a sensor module does not read calibration files and later
accesses reuse the parsed table.

Format-specific dependencies belong in the sensor's optional extra in
`packages/geotoolz-products/pyproject.toml`. Import them lazily and re-raise a
missing import as an `ImportError` naming the extra
(`pip install 'geotoolz-products[<sensor>]'`) instead of surfacing a
library-internal error.

`geoproducts` never depends on `geotoolz`: readers produce `GeoTensor`s that
the operators consume. `presets.py` bridges the two with zero-argument
wrappers over generic operators — for example `toy_sensor.NDVI()` returning
`geotoolz.indices.NDVI(red="red", nir="nir")` — importing geotoolz inside the
function, behind the `[operators]` extra. An operator that is genuinely new
belongs in a geotoolz family, not in the reader package.

## Reference implementation: `toy_sensor`

`geoproducts.toy_sensor` is the worked example of the contract. It is an
in-memory reader (no external file format) used to exercise the framework in
tests and to demonstrate the layout for real sensors. Real product readers
(MODIS HDF4, VIIRS HDF5, GOES NetCDF, etc.) are tracked separately under
the per-sensor design issues.
