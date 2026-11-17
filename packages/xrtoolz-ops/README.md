# xrtoolz

Composable operator library for geoprocessing Earth System Data Cubes —
preprocess, infer, and evaluate xarray datasets with a uniform pipeline
abstraction built on [pipekit](https://github.com/jejjohnson/pipekit).

This is the main member of the xrtoolz workspace. The foundation layers
live in sibling packages and are installed automatically with it:

| Distribution | Import | Scope |
|---|---|---|
| `pipekit-array` | `pipekit_array` | The shared bridges: `to_rows` / `from_rows`, `Estimator`, and the `[einx]` bridge |

```bash
pip install xrtoolz
```

See the [repository root](https://github.com/jejjohnson/xrtoolz) for the
full documentation site, and `CHANGELOG.md` here for release history.
