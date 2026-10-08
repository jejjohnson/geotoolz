# geotoolz-products

Readers for Earth-observation data products. Each reader turns a mission's
or provider's product into a [georeader](https://github.com/spaceml-org/georeader)
`GeoData` / `GeoTensor`, so the rest of the geotoolz stack — the operators
(`geotoolz`), the patchers (`geopatcher`) and the catalog loaders
(`geocatalog`) — consumes it unchanged.

```bash
pip install geotoolz-products                  # readers (georeader only)
pip install 'geotoolz-products[obstore]'       # pooled cloud byte-range reads
pip install 'geotoolz-products[operators]'     # sensor presets (geotoolz operators)
pip install 'geotoolz-products[carbonmapper]'  # Carbon Mapper plume catalogue + STAC
```

```python
import numpy as np
from rasterio.windows import Window

from geoproducts import toy_sensor

reader = toy_sensor.Reader("scene", data=np.zeros((4, 64, 64), dtype="float32"))
chip = reader.read_from_window(Window(0, 0, 32, 32)).load()  # GeoTensor with band_names
```

Import name: `geoproducts`. The package depends on georeader, never on
`geotoolz`; the optional `[operators]` extra only powers the per-sensor
`presets`.
