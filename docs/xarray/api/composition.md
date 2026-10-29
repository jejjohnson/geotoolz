# Composition

The composition primitives are carrier-agnostic and live in
[`pipekit`](https://github.com/jejjohnson/pipekit); the xarray-aware
`Operator` base is `xrcore`, and the Dataset combinators are `xrops`:

```python
from pipekit import Sequential, Graph, Input, Node
from xrcore import Operator
from xrops import Augment
```

`xrcore.Operator` is a thin xarray-aware subclass of `pipekit.Operator`
that adds `xarray.DataTree` dispatch to `__call__` — every operator in the
library inherits it, so all of them map leaf-wise over a `DataTree` for
free. The xarray-Dataset-specific combinators (`Augment`, `ApplyToEach`)
live in `xrops.combinators`.

!!! example "Linear and DAG composition, same operators"
    ```python
    from pipekit import Sequential, Graph, Input
    from xrops.geo import RemoveMean
    from xrops.ocn import Streamfunction, GeostrophicVelocities

    # Linear pipeline
    pipe = Sequential(RemoveMean(dim=("lat", "lon")), GeostrophicVelocities())
    out = pipe(ds)

    # The same operators wired as a DAG
    x = Input("ssh")
    psi = Streamfunction()(x)
    uv = GeostrophicVelocities()(psi)
    graph = Graph(inputs=x, outputs=uv)
    ```

## Operator base class

::: xrcore.Operator

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

::: xrops.combinators.Augment

::: xrops.combinators.ApplyToEach

## Shape signatures

`Signature` is the dict-keyed shape descriptor used by
`compute_output_signature` for static shape inference across a pipeline.

::: xrcore.Signature
