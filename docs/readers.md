# Adding a new sensor reader

Sensor integrations live under `geotoolz.readers.<sensor>` and keep the same
small surface so each sensor can be audited in isolation.

```text
geotoolz/readers/<sensor>/
  __init__.py      # Reader, BANDS, CONSTANTS, ops, presets
  reader.py        # SensorReader subclass
  constants.py     # lazy calibration table accessors
  ops.py           # sensor-specific operators
  presets.py       # zero-argument sensor-aware wrappers
  data/            # packaged calibration files
```

A reader should subclass `geotoolz.readers.SensorReader`, implement the
metadata properties (`_crs`, `_transform`, `_shape`, `_dtype`, `_bands`,
`_fill_value`, `_track`), and provide `_read_window(window)`. Track `"A"`
means a clean affine grid; track `"B"` is reserved for sensors with irregular
geolocation.

Calibration data should be packaged below `data/`, kept small, and loaded via
`geotoolz.readers._constants.load_csv()` or `load_json()`. These loaders are
cached, so importing a sensor module does not read calibration files and later
accesses reuse the parsed table.

Format-specific dependencies belong in the sensor optional extra in
`pyproject.toml`. Guard imports with `require_optional_dependency()` so missing
extras raise messages like `pip install 'geotoolz[viirs]'` instead of
library-internal errors. Extras with empty dependency lists are reserved no-op
extras for sensors whose parser is planned to be in-tree or whose dependency is
not yet published as a stable package.

Start new operators in `geotoolz.readers.<sensor>.ops`. Promote an operator to a
generic module only after at least two sensors need the same clean abstraction.
Use `presets.py` for zero-argument wrappers over generic operators, for example
`modis.NDVI()` returning `geotoolz.indices.NDVI(red="red", nir="nir")`.
