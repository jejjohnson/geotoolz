<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/geotoolz/main/docs/assets/logo-dark.svg">
    <img alt="geotoolz" src="https://raw.githubusercontent.com/jejjohnson/geotoolz/main/docs/assets/logo-light.svg" width="333">
  </picture>
</p>

# geotoolz

> The [geotoolz monorepo](https://github.com/jejjohnson/geotoolz) in one install.

`pip install geotoolz` installs the five packages of the stack. The umbrella
holds no code; import each package under its own name:

| Distribution | Import | Use it for |
|---|---|---|
| `geotoolz-ops` | `geoops` | Operator families on `GeoTensor` |
| `geotoolz-patcher` | `geopatcher` | Split a field into patches, process, stitch |
| `geotoolz-catalog` | `geocatalog` | Find, index, join and load files |
| `geotoolz-cloud` | `geocloud` | Object storage and COG reads |
| `geotoolz-products` | `geoproducts` | Readers for Earth-observation products |

```python
import geoops as gz
import geopatcher as gp
import geocatalog as gc
```

Extras forward to the package that owns them: `pip install 'geotoolz[viz]'`
is the same as `pip install 'geotoolz-ops[viz]'`.

Before 0.11, `geotoolz` was the operator package. It is now `geotoolz-ops`
(import `geoops`); `import geotoolz.<family>` no longer works.
