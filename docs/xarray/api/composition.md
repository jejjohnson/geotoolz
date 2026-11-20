# Composition

The composition primitives are carrier-agnostic and live in
[`pipekit`](https://github.com/jejjohnson/pipekit); the xarray-aware
`Operator` base and the Dataset combinators are `xrops`:

```python
from pipekit_ops import Sequential, Graph, Input, Node
from xrops import Augment, Operator
```

`xrops.Operator` is a thin xarray-aware subclass of `pipekit_ops.Operator`
that adds `xarray.DataTree` dispatch to `__call__` — every operator in the
library inherits it, so all of them map leaf-wise over a `DataTree` for
free. The xarray-Dataset-specific combinators (`Augment`, `ApplyToEach`)
live in `xrops.combinators`.

!!! example "Linear and DAG composition, same operators"
    ```python
    from pipekit_ops import Sequential, Graph, Input
    from xrops.anomaly import RemoveMean
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

::: xrops.Operator

## Sequential pipelines

[`Sequential`](../../operators/api/core.md#pipekit_ops.Sequential) is pipekit's, documented with the geotoolz operators.

## Functional Graph API

[`Input`](../../operators/api/core.md#pipekit_ops.Input), [`Node`](../../operators/api/core.md#pipekit_ops.Node) and [`Graph`](../../operators/api/core.md#pipekit_ops.Graph) are pipekit's, documented with the geotoolz operators.

## Observation taps

`Tap` is a generic pass-through observer from `pipekit_ops` — it runs a
side-effecting callback (logging, plotting, assertions) without altering
the data flowing through.

It is documented with the geotoolz operators: [`Tap`](../../operators/api/core.md#pipekit_ops.Tap).

## xarray-Dataset combinators

`Augment` merges an operator's output back into its input Dataset (rather
than replacing it); `ApplyToEach` maps an operator over every data
variable independently.

::: xrops.combinators.Augment

::: xrops.combinators.ApplyToEach

## Shape signatures

`Signature` is pipekit's shape descriptor (ordered `(name, size)` dims and a dtype), used by `compute_output_signature` for static shape inference across a pipeline.

::: pipekit_ops.Signature
    options:
      docstring_section_style: "table"
      docstring_options: {"warn_unknown_params": false, "warn_missing_types": false}
      merge_init_into_class: true
      show_root_full_path: false
      show_symbol_type_heading: true
      show_symbol_type_toc: true
      separate_signature: true
      show_signature_annotations: true
      signature_crossrefs: true
      members_order: "source"
      filters: ["!^_"]
      heading_level: 3
