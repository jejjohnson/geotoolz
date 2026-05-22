# Core — Composition Primitives

The composition primitives (`Sequential`, `Graph`, `Input`, `Node`,
`Tap`) live in the carrier-agnostic
[`pipekit`](https://github.com/jejjohnson/pipekit) framework and are
re-exported at the top level of `xrtoolz` for convenience (e.g.
`from xrtoolz import Sequential, Augment`).

`xrtoolz.Operator` is a thin xarray-aware subclass of `pipekit.Operator`
that adds `DataTree` dispatch to `__call__` — every operator in the
library inherits it and therefore gains free leaf-wise mapping over
`xarray.DataTree` inputs. The xarray-Dataset-specific combinators
(`Augment`, `ApplyToEach`) live in `xrtoolz.combinators`.

## Operator base class

::: xrtoolz.Operator

## Sequential pipelines

::: pipekit.Sequential

## Functional Graph API

::: pipekit.Input

::: pipekit.Node

::: pipekit.Graph

## Generic observation combinator (from pipekit)

::: pipekit.Tap

## xarray-specific combinators

::: xrtoolz.combinators.Augment

::: xrtoolz.combinators.ApplyToEach
