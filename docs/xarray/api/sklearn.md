# scikit-learn bridge (xrsklearn)

Run any scikit-learn estimator on labeled N-D data. `XarrayEstimator`
flattens a `DataArray` / `Dataset` / `DataTree` to sklearn's
`(n_samples, n_features)` matrix, delegates, and rebuilds labeled
outputs. It checks every input against the feature grid seen at fit
time, handles land masks and gaps through `nan_policy`, and fits trees
per node, pooled, or column-joined through `tree_mode`. `SklearnOp`
puts the same machinery into `pipekit.Sequential` / `Graph` pipelines,
and importing `xrsklearn` registers the `.sklearn` accessors.

See the [package guide](../packages/sklearn.md) for the lifecycle, the
NaN policies, and DataTree modes.

## Estimator

::: xrsklearn.XarrayEstimator

## Pipeline operator

::: xrsklearn.SklearnOp

## Accessors

::: xrsklearn._src.accessor.SklearnDataArrayAccessor
    options:
      inherited_members: true

::: xrsklearn._src.accessor.SklearnDatasetAccessor

::: xrsklearn._src.accessor.SklearnDataTreeAccessor

## Policies and modes

::: xrsklearn._src.nan
    options:
      members: false

::: xrsklearn._src.tree
    options:
      members: false
