# Core — Composition Primitives

Domain-agnostic primitives for building and running operator pipelines.
Every public symbol lives at the top level of `xr_toolz` (e.g.
`from xr_toolz import Sequential, Augment`).

## Operator base class

::: xr_toolz.core.operator.Operator

## Sequential pipelines

::: xr_toolz.core.sequential.Sequential

## Functional Graph API

::: xr_toolz.core.graph.Input

::: xr_toolz.core.graph.Node

::: xr_toolz.core.graph.Graph

## Operator combinators

::: xr_toolz.core.combinators.Augment

::: xr_toolz.core.combinators.Tap

::: xr_toolz.core.combinators.ApplyToEach
