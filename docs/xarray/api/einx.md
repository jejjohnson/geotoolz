# Named-tensor algebra (einx)

Labeled named-tensor algebra bridging xarray and
[einx](https://github.com/fferflo/einx). Pattern axis tokens are DataArray
dimension names, so `einsum` / `rearrange` / `reduce` / `repeat` operate by
*name* rather than position. `pack_dataset` / `unpack_dataset` flatten a
Dataset's variables into a single packed axis and back.

!!! warning "`geo.Reduce` vs `einx.Reduce`"
    `einx.Reduce` reduces along an einx *pattern*; `xrtoolz.geo.Reduce`
    aggregates Dataset variables. They are different operators that happen to
    share a name — pick by import path.

## Operators

::: xreinx.Einsum

::: xreinx.Rearrange

::: xreinx.Reduce

::: xreinx.Repeat

::: xreinx.Matmul

::: xreinx.Outer

::: xreinx.BatchMatmul

## Functional primitives (Layer 0)

These pure functions back the operators above; each takes `xr.DataArray`/`xr.Dataset` and a `dim:` argument.

::: xreinx.einsum

::: xreinx.rearrange

::: xreinx.reduce

::: xreinx.repeat

::: xreinx.matmul

::: xreinx.outer

::: xreinx.batch_matmul

::: xreinx.pack_dataset

::: xreinx.unpack_dataset

## Errors

::: xreinx.EinxBridgeError

::: xreinx.PatternError

::: xreinx.CoordMismatch
