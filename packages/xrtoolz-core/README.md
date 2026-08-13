# xrtoolz-core

The xarray-aware operator base for the xrtoolz stack (import name:
`xrcore`).

Two building blocks shared by every xarray-facing operator package:

- **`xrcore.Operator`** — subclasses `pipekit.Operator` and adds
  `xr.DataTree` leaf-wise dispatch: one `_apply` implementation handles
  eager `Dataset` calls, symbolic `Node` graph construction, and
  `DataTree` inputs (mapped over every leaf via
  `xr.map_over_datasets`).
- **`xrcore.Signature`** — a dict-keyed shape descriptor threaded
  through pipelines for keras-style `summary()` tables.

```bash
pip install xrtoolz-core
```

```python
import xarray as xr
from xrcore import Operator


class Double(Operator):
    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        return ds * 2


Double()(tree)  # works on Dataset, Node, or DataTree alike
```
