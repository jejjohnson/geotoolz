# scikit-learn on xarray (`xrops.learn`)

The stack has one scikit-learn bridge: pipekit-array's `Estimator`, with `to_rows` / `from_rows`, which chooses samples by dim name on any carrier (geostack OP-O3).
On a single `DataArray` in a pipeline, use it directly.

`xrops.learn` adds what is xarray's own.
`XarrayEstimator` puts a `Dataset`'s variables side by side as one feature table, each on its own grid, and fits a `DataTree` per node or pooled over its nodes.
Importing `xrops.learn` registers the `.sklearn` accessor on `DataArray`, `Dataset` and `DataTree`.
scikit-learn is imported when an estimator is fitted, so install the `xrtoolz-ops[sklearn]` extra.

::: xrops.learn.XarrayEstimator

::: xrops.learn.TreeMode
