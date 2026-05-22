# Core — Composition Primitives

The composition primitives (`Operator`, `Sequential`, `Graph`, `Input`,
`Node`, `Tap`) live in the carrier-agnostic
[`pipekit`](https://github.com/jejjohnson/pipekit) framework and are
re-exported at the top level of `xrtoolz` for convenience (e.g.
`from xrtoolz import Sequential, Augment`). The xarray-Dataset-specific
combinators (`Augment`, `ApplyToEach`) live in `xrtoolz.combinators`.

## Operator base class

::: pipekit.Operator

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
