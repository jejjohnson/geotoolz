# xrtoolz-einx — labeled named-tensor algebra

```bash
# Pre-PyPI, install from the workspace via git (with its xrcore/pipekit
# base); once published this becomes a plain `pip install xrtoolz-einx`.
uv pip install \
  "xrtoolz-einx @ git+https://github.com/jejjohnson/xrtoolz@main#subdirectory=packages/xrtoolz-einx" \
  "xrtoolz-core @ git+https://github.com/jejjohnson/xrtoolz@main#subdirectory=packages/xrtoolz-core" \
  "pipekit @ git+https://github.com/jejjohnson/pipekit@main#subdirectory=packages/pipekit"
```

`xreinx` bridges xarray and [einx](https://github.com/fferflo/einx):
you write einx patterns whose axis tokens are **DataArray dim names**,
and the bridge transposes to match, dispatches to einx on the raw
arrays, and rewraps the result as a labeled DataArray with coords
forwarded from the inputs. `import xreinx` stays light — the einx
backend loads lazily on the first pattern call.

## The pattern language

A pattern is `input_slot[, input_slot…] -> output_slot`, whitespace-
separated axis tokens per slot:

```python
import xreinx as xnx

total = xnx.einsum("time lat lon, lat lon -> time", field, mask)
```

Two matching semantics, chosen per function:

- **Name-matched** (`einsum`, `reduce`, `repeat`): each input slot
  lists that input's dim names *in any order* — the bridge transposes
  to the slot order before dispatch, so patterns are independent of how
  upstream code ordered the dims. Slots must be flat dim names; a
  mismatch between slot names and the array's dims raises
  `PatternError`.
- **Positional** (`rearrange`, like einx itself): merge/split groups
  such as `(lat_blk lat_in)` have no single dim name to match against,
  so the input slot describes the array's existing axes *in order*.
  The output slot names the result dims; a merged group `(a b)` becomes
  a single dim named `a_b`.

Axis sizes that the pattern cannot infer from the inputs are supplied
as keyword arguments (`lat_in=4`), exactly as in einx.

## Coordinate policy

Coordinates are forwarded from the first input that carries each
surviving dim. Shared dims with mismatched coords raise
`CoordMismatch` by default; pass `align=True` to `einsum` to
inner-join the inputs with `xr.align` first. Brand-new dims (from
`repeat` or a split) are unindexed unless you pass `coords=`.

Errors are typed: `PatternError` for malformed or mismatched patterns,
`CoordMismatch` for coordinate conflicts, both subclasses of
`EinxBridgeError`.

## Dataset round-trips

`pack_dataset` / `unpack_dataset` stack a Dataset's variables along a
new dim (default `"variable"`, indexed by the variable names) and back
— for models and solvers that want one array:

```python
packed = xnx.pack_dataset(ds)        # (variable, time, lat, lon)
restored = xnx.unpack_dataset(packed)  # Dataset with the original vars
```

The variables must share dims and coords; patterns can then treat
`variable` as an ordinary named axis.

## Operators

The seven pattern and linear-algebra functions have matching
`xrcore.Operator` wrappers — `Einsum`, `Rearrange`, `Reduce`, `Repeat`,
`Matmul`, `Outer`, `BatchMatmul` — for composition into
`pipekit.Sequential` chains and `Graph` pipelines. The wrappers
implement `compute_output_signature`, so structural `summary()` tables
see through them without executing anything. `pack_dataset` /
`unpack_dataset` are deliberately plain functions with no wrappers —
they are data-layout plumbing, not pipeline steps.

## Linear-algebra conveniences

`matmul(a, b, dim=…)` contracts one named dim; `outer(a, b)` requires
disjoint dims and concatenates them; `batch_matmul(a, b, dim=…,
batch_dims=[…])` contracts while broadcasting over shared batch dims.
All three are sugar over `einsum` patterns.

## API reference

- [Named-tensor algebra](../api/einx.md)

The bridge's design history (pattern grammar decisions, coord-policy
trade-offs) lives under
[Design → Bridge Modules → einx](../design/bridges/einx/vision.md).
