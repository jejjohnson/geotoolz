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

::: xrtoolz.einx.Einsum

::: xrtoolz.einx.Rearrange

::: xrtoolz.einx.Reduce

::: xrtoolz.einx.Repeat

::: xrtoolz.einx.Matmul

::: xrtoolz.einx.Outer

::: xrtoolz.einx.BatchMatmul

## Functional primitives (Layer 0)

These pure functions back the operators above; each takes `xr.DataArray`/`xr.Dataset` and a `dim:` argument.

::: xrtoolz.einx.einsum

::: xrtoolz.einx.rearrange

::: xrtoolz.einx.reduce

::: xrtoolz.einx.repeat

::: xrtoolz.einx.matmul

::: xrtoolz.einx.outer

::: xrtoolz.einx.batch_matmul

::: xrtoolz.einx.pack_dataset

::: xrtoolz.einx.unpack_dataset

## Errors

::: xrtoolz.einx.EinxBridgeError

::: xrtoolz.einx.PatternError

::: xrtoolz.einx.CoordMismatch
