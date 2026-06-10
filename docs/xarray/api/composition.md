# Composition

The composition primitives are carrier-agnostic and live in
[`pipekit`](https://github.com/jejjohnson/pipekit); `xrtoolz` re-exports
them at the top level so the common names are one import away:

```python
from xrtoolz import Operator, Sequential, Graph, Input, Node, Augment
```

`xrtoolz.Operator` is a thin xarray-aware subclass of `pipekit.Operator`
that adds `xarray.DataTree` dispatch to `__call__` — every operator in the
library inherits it, so all of them map leaf-wise over a `DataTree` for
free. The xarray-Dataset-specific combinators (`Augment`, `ApplyToEach`)
live in `xrtoolz.combinators`.

!!! example "Linear and DAG composition, same operators"
    ```python
    from xrtoolz import Sequential, Graph, Input
    from xrtoolz.geo import RemoveMean
    from xrtoolz.ocn.operators import Streamfunction, GeostrophicVelocities

    # Linear pipeline
    pipe = Sequential(RemoveMean(var="ssh"), GeostrophicVelocities())
    out = pipe(ds)

    # The same operators wired as a DAG
    x = Input("ssh")
    psi = Streamfunction()(x)
    uv = GeostrophicVelocities()(psi)
    graph = Graph(inputs=x, outputs=uv)
    ```

## Operator base class

::: xrtoolz.Operator

## Sequential pipelines

::: pipekit.Sequential

## Functional Graph API

::: pipekit.Input

::: pipekit.Node

::: pipekit.Graph

## Observation taps

`Tap` is a generic pass-through observer from `pipekit` — it runs a
side-effecting callback (logging, plotting, assertions) without altering
the data flowing through.

::: pipekit.Tap

## xarray-Dataset combinators

`Augment` merges an operator's output back into its input Dataset (rather
than replacing it); `ApplyToEach` maps an operator over every data
variable independently.

::: xrtoolz.combinators.Augment

::: xrtoolz.combinators.ApplyToEach

## Shape signatures

`Signature` is the dict-keyed shape descriptor used by
`compute_output_signature` for static shape inference across a pipeline.

::: xrtoolz.signature.Signature
