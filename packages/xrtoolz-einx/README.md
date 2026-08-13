# xrtoolz-einx

Labeled named-tensor algebra bridging xarray and
[einx](https://github.com/fferflo/einx) (import name: `xreinx`).

Pattern axis tokens are DataArray *dim names*; the bridge transposes to
match before dispatching to einx and rewraps the result as a labeled
DataArray with coords forwarded from the inputs.

Two surfaces:

- **Functions**: `einsum`, `rearrange`, `reduce`, `repeat`, plus
  `matmul` / `outer` / `batch_matmul` conveniences and
  `pack_dataset` / `unpack_dataset` for Dataset ↔ tensor round-trips.
- **Operators**: `Einsum`, `Rearrange`, `Reduce`, `Repeat`, `Matmul`,
  `Outer`, `BatchMatmul` — `xrcore.Operator` subclasses for pipeline
  composition.

`import xreinx` stays light: the einx backend loads lazily on the first
pattern call.

```bash
pip install xrtoolz-einx
```

```python
import xreinx as xnx

total = xnx.einsum("time lat lon, lat lon -> time", field, mask)
```
