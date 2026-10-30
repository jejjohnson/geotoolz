# xrtoolz

The xarray side of the geostack in one install. This distribution holds no code: it installs the family, and each package is imported under its own name.

| Distribution | Import | Use it for |
| --- | --- | --- |
| `xrtoolz-ops` | `xrops` | Operator families on xarray: atm, ocn, budgets, geo, interpolate, metrics, transforms, viz |
| `xrtoolz-products` | `xrproducts` | Readers for CMEMS, CDS, AEMET and local L2 products |
| `xrtoolz-grad` | `xrgrad` | Finite-difference calculus on xarray |

```bash
pip install xrtoolz          # the whole family
pip install xrtoolz-ops      # only the operators
```
