# Core — Composition Primitives

Domain-agnostic primitives for building and running operator pipelines.
Every public symbol lives at the top level of `xrtoolz` (e.g.
`from xrtoolz import Sequential, Augment`).

## Operator base class

::: xrtoolz.core.operator.Operator

## Sequential pipelines

::: xrtoolz.core.sequential.Sequential

## Functional Graph API

::: xrtoolz.core.graph.Input

::: xrtoolz.core.graph.Node

::: xrtoolz.core.graph.Graph

## Operator combinators

::: xrtoolz.core.combinators.Augment

::: xrtoolz.core.combinators.Tap

::: xrtoolz.core.combinators.ApplyToEach
