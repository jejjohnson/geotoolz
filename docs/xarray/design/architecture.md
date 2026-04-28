---
status: draft
version: 0.1.0
---

!!! note "These design docs cover the planned operator surface for `xr_toolz`"
    Code snippets use class names directly. In the implementation, the
    submodule layout is:

    - **`xr_toolz.geo`** — domain-agnostic geoprocessing (CRS, validation,
      subset, masks, detrend)
    - **`xr_toolz.interpolate`** — value resampling: regrid, gap-fill,
      bin, coord-axis remap, time resample, smooth, learned downscale
      (D12)
    - **`xr_toolz.transforms`** — signal transforms / decompositions /
      encoders (D8)
    - **`xr_toolz.metrics`** — skill scores (D7)
    - **`xr_toolz.kinematics`** — domain-specific physical quantities,
      sub-organized by domain (D9)
    - **`xr_toolz.viz`** — plotting operators (D10)

    See `xr_toolz/__init__.py` for the current export surface.

# Architecture

## Three Layers — Progressive Disclosure of Complexity

The library follows Keras's principle of progressive disclosure: simple things are simple, complex things are possible. Each layer builds on the one below it.

**Layer 0 — Pure Functions.** The implementation layer. Every geoprocessing operation is first written as a pure function with a clear signature, typically `(xr.Dataset, ...) → xr.Dataset`. These live in `src/geo_toolz/_src/<module>/` and are always importable directly. They are the foundation: transparent, testable, pipeable via `toolz` or `xr.Dataset.pipe`.

```python
from geo_toolz._src.detrend.climatology import calculate_climatology, remove_climatology

clim = calculate_climatology(ds, freq="day", smoothing=60)
ds_anom = remove_climatology(ds, clim)
```

**Layer 1 — Operator + Sequential.** Thin callable wrappers around Layer 0 functions. They provide a uniform `__call__` interface, carry their configuration, support introspection (`get_config`, `__repr__`), and compose via `Sequential`. These live alongside their Layer 0 counterparts in the same `_src/<module>/` directories and are re-exported through the public API.

```python
from geo_toolz.detrend import CalculateClimatology, RemoveClimatology
from geo_toolz.core import Sequential

clim = CalculateClimatology(freq="day", smoothing=60)(ds_train)

pipeline = Sequential([
    ValidateCoords(),
    Regrid(target_grid=grid, method="linear"),
    RemoveClimatology(clim),
    Subset(region="mediterranean"),
])

ds_clean = pipeline(ds_raw)
```

Sequential covers ~80% of pipelines: linear chains of single-input operators. Layer 1 always delegates to Layer 0. No logic is duplicated.

**Layer 2 — Functional Graph API.** For workflows that need branching, merging, multiple inputs, or multiple outputs — the remaining 20%. Directly inspired by `keras.Model(inputs=..., outputs=...)`. You build a computation graph by calling operators on symbolic `Input` nodes, then compile the graph into a callable `Graph` operator.

```python
from geo_toolz.core import Input, Graph

# Declare symbolic inputs
raw = Input(name="satellite_data")
ref = Input(name="reference")

# Build the graph — each call returns a Node, not data
validated = ValidateCoords()(raw)
regridded = Regrid(target_lon, target_lat)(validated)
detrended = RemoveClimatology(clim)(regridded)

# Multi-input operator: metrics take (prediction, reference)
score = RMSE(variable="ssh", dims=["time"])(detrended, ref)

# Compile
pipeline = Graph(
    inputs={"satellite_data": raw, "reference": ref},
    outputs={"cleaned": detrended, "rmse": score},
)

# Execute
results = pipeline(satellite_data=ds_raw, reference=ds_ref)
# results["cleaned"] → xr.Dataset
# results["rmse"] → xr.DataArray
```

The `Graph` is itself an `Operator` — it can be nested inside other Graphs or Sequentials. This is what makes multi-input operators (metrics, merging, concatenation) first-class citizens of the composition system rather than one-off manual wiring.

## Type Contract — Three Tiers (Array, xarray, Operator)

The composition layers above describe *how* operators stack. The type contract describes *what* each tier accepts and returns. See [decisions.md §D11](decisions.md) for the decision record.

| Tier | Location | Input | Output | Coordinate semantics |
|---|---|---|---|---|
| **A — Array** | `xr_toolz.<module>.array` | array (numpy / JAX / numba-jitted / optionally CuPy) | array | `axis=` |
| **B — Layer 0 xarray** | `xr_toolz.<module>` (private `_src/`) | `xr.DataArray` (single-variable) or `xr.Dataset` + variable selectors (multi-variable) | `xr.DataArray` or `xr.Dataset` | `dim=` |
| **C — Layer 1 Operator** | `xr_toolz.<module>` | `xr.Dataset` (or two for multi-input) | `xr.Dataset` \| `xr.DataArray` \| scalar (terminal viz returns `matplotlib.Figure / Axes`, see D10) | constructor `variable=` / `dims=` |

Rules:

- Each tier delegates downward; logic is never duplicated. Tier B wraps Tier A; Tier C wraps Tier B.
- Tier A is **pragmatic, not strictly Array API-compliant**: numpy is the default backend, with JAX / numba / CuPy variants added per-function as the math benefits. Some functions dispatch via `array_namespace(x)`; others are hand-authored backend-specific kernels. The library never imports JAX / CuPy at the top level — optional backends are imported lazily.
- Tier B uses arity to disambiguate the input type: single-variable functions take `xr.DataArray`; multi-variable functions take `xr.Dataset` plus explicit variable selectors (`variable=`, `u_var=`, …).
- Tier C input is always `xr.Dataset` (or two `xr.Dataset` for multi-input operators). Output is **usually** `xr.Dataset` for transformations that preserve the dataset shape, but reduction-style operators (e.g., metrics) may return an `xr.DataArray` or scalar, and terminal viz operators return `matplotlib.Figure / Axes` (D10). Composition (`Sequential`, `Graph`) only sees Tier C.
- Modules whose math is inherently coord/attr-manipulation rather than arithmetic (`validation`, `crs`, `subset`, `masks`) skip Tier A; their Tier B takes `xr.Dataset` directly.

Example — `metrics.rmse`:

```python
# Tier A — duck array (numpy, JAX, CuPy, Dask)
xr_toolz.metrics.array.rmse(pred_arr, ref_arr, axis=-1)

# Tier B — Layer 0 xarray (DataArray in, DataArray out)
xr_toolz.metrics.rmse(pred_da, ref_da, dim="time")

# Tier C — Layer 1 Operator (Dataset in, DataArray out)
RMSE(variable="ssh", dims=["time"])(pred_ds, ref_ds)
```

## The `Operator` Base Class

```python
class Operator:
    """Base class for all geo_toolz operators.

    Every operator is a callable. Single-input operators map
    Dataset → Dataset. Multi-input operators accept multiple
    positional arguments. Reductions may return scalars or
    lower-dimensional Datasets. Terminal viz operators may return
    matplotlib Figure / Axes (see decisions.md §D10).

    Subclasses must implement `__call__`.
    """

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def get_config(self) -> dict:
        """Return a JSON-serializable dict of constructor arguments.

        This dict, combined with the operator's class name, is sufficient
        to reconstruct the operator (modulo rich state like learned
        climatologies, which are referenced by path).
        """
        ...

    def __repr__(self) -> str:
        """Human-readable representation showing class name and config."""
        config = self.get_config()
        params = ", ".join(f"{k}={v!r}" for k, v in config.items())
        return f"{self.__class__.__name__}({params})"

    def __or__(self, other: "Operator") -> "Sequential":
        """Pipe syntax: op_a | op_b creates Sequential([op_a, op_b])."""
        if isinstance(other, Sequential):
            return Sequential([self, *other.operators])
        return Sequential([self, other])
```

Design decisions:

- `get_config()` returns only JSON-serializable values. Rich state (climatology arrays, fitted grids) is passed via constructor as pre-computed objects. If serialization is needed, the user stores them as NetCDF/Zarr and references them by path in the config. This keeps the operator Hydra/DVC-friendly without building a custom serialization system.
- The `|` operator enables `ValidateCoords() | Regrid(grid) | Subset(region)` syntax as sugar for `Sequential`. This is optional — `Sequential([...])` is the primary interface.
- `__call__` is deliberately not typed with a fixed signature in the base class. Single-input ops take `(ds)`, multi-input ops take `(ds_a, ds_b)`, reductions return scalars. The base class is permissive; subclass docstrings document the specific contract.

## `Sequential`

```python
class Sequential(Operator):
    """A pipeline of single-input operators, applied left to right.

    Sequential is itself an Operator, so pipelines nest:
        preprocess = Sequential([ValidateCoords(), Regrid(grid)])
        full = Sequential([preprocess, Detrend(clim), Subset(region)])

    Non-Dataset returns (e.g., a terminal viz Operator that returns
    matplotlib.Figure — see decisions.md §D10) are allowed only as the
    LAST step. A non-Dataset return from any earlier step is a runtime
    error, since the next operator would receive an unexpected type.
    """

    def __init__(self, operators: list[Operator]):
        self.operators = operators

    def __call__(self, ds):
        for i, op in enumerate(self.operators):
            ds = op(ds)
            is_last = i == len(self.operators) - 1
            if not is_last and not isinstance(ds, (xr.Dataset, xr.DataArray)):
                raise TypeError(
                    f"Step [{i}] {op!r} returned {type(ds).__name__}; "
                    f"non-Dataset returns are only allowed at the final step "
                    f"of a Sequential (see decisions.md §D10)."
                )
        return ds

    def get_config(self) -> dict:
        return {
            "operators": [
                {"class": op.__class__.__name__, "config": op.get_config()}
                for op in self.operators
            ]
        }

    def describe(self) -> str:
        """Pretty-print the pipeline steps."""
        lines = [f"Sequential ({len(self.operators)} steps):"]
        for i, op in enumerate(self.operators):
            lines.append(f"  [{i}] {op!r}")
        return "\n".join(lines)
```

**Type rule (D10).** Most operators are `Dataset → Dataset`. Terminal viz operators (`PlotMap`, `PlotSpectrum`, …) return `matplotlib.Figure` / `Axes`. `Sequential` validates that any non-`Dataset` return appears only at the last step; otherwise it raises a clear `TypeError`. `Graph` already supports heterogeneous output types and needs no change — viz nodes slot in as one of N outputs.

## Stateful Operations: The Split-Object Pattern

Some operations require learning from data before they can be applied. Climatological detrending needs a climatology computed from a training period. Spatial scaling needs statistics from a reference dataset.

Rather than adding `fit` / `transform` methods to `Operator` (which would complicate `Sequential` and break the uniform `__call__` interface), we use separate objects for the learning and applying phases:

```python
# Learning phase: a function or operator that returns state
clim = CalculateClimatology(freq="day", smoothing=60)(ds_train)
scaler_params = CalculateSpatialStats(dims=["lat", "lon"])(ds_train)

# Applying phase: stateless operators parameterized by the learned state
pipeline = Sequential([
    RemoveClimatology(clim),
    NormalizeSpatial(scaler_params),
])

# Everything in the pipeline is Dataset → Dataset. No fit/transform duality.
ds_test_clean = pipeline(ds_test)
```

This means:

- Every operator in a `Sequential` is `Dataset → Dataset`, always.
- State computation is explicit and happens upstream, not hidden inside the pipeline.
- The learned state (a climatology, a set of statistics) is just an xarray object — it can be saved to disk, inspected, plotted.
- The applying operator is Hydra-serializable if the state object is referenced by path.


## The Functional Graph API (Layer 2)

Layer 2 provides Keras's functional API for geo_toolz: build arbitrary computation graphs with branching, merging, and multiple inputs/outputs by calling operators on symbolic nodes.

### Node and Input

When an `Operator` is called on an `Input` (or on the output of a previous operator call on an `Input`), it doesn't execute — it records the operation and returns a `Node` that represents the deferred computation.

```python
class Node:
    """A symbolic placeholder representing an intermediate result in a Graph.

    Nodes are created automatically when operators are called on Inputs
    or other Nodes. Users don't instantiate these directly.
    """
    def __init__(self, operator: Operator, parents: tuple["Node | Input", ...]):
        self.operator = operator
        self.parents = parents
        self.name = None  # optionally named for output lookup

class Input(Node):
    """A named entry point into a computation graph.

    Inputs have no parents and no operator. They are pure placeholders
    that get bound to real datasets at execution time.
    """
    def __init__(self, name: str):
        self.name = name
        self.operator = None
        self.parents = ()
```

The key mechanism is in `Operator.__call__`: it detects whether its arguments are `Node` instances and, if so, returns a new `Node` instead of executing:

```python
class Operator:
    def __call__(self, *args, **kwargs):
        # If any argument is a Node, we're in graph-building mode
        if any(isinstance(a, Node) for a in args):
            return Node(operator=self, parents=args)
        # Otherwise, execute normally
        return self._apply(*args, **kwargs)

    def _apply(self, *args, **kwargs):
        """Actual computation. Subclasses implement this."""
        raise NotImplementedError
```

This means every operator works in both modes — eager (Layer 1) and symbolic (Layer 2) — with zero changes to the operator itself. The mode is determined entirely by what you pass in.

### Graph

`Graph` takes named inputs and named outputs, topologically sorts the nodes, and executes them in order.

```python
class Graph(Operator):
    """A computation DAG compiled from symbolic Node connections.

    Graph is itself an Operator, so it composes: a Graph can be a step
    inside a Sequential or a node in a larger Graph.
    """

    def __init__(
        self,
        inputs: dict[str, Input],
        outputs: dict[str, Node],
    ):
        self.inputs = inputs
        self.outputs = outputs
        self._execution_order = self._topological_sort()

    def __call__(self, **kwargs) -> dict[str, xr.Dataset]:
        """Execute the graph.

        Args:
            **kwargs: Named datasets corresponding to each Input.
                e.g. graph(satellite_data=ds_raw, reference=ds_ref)

        Returns:
            dict mapping output names to their computed results.
        """
        # Bind inputs to real data
        cache = {}
        for name, input_node in self.inputs.items():
            cache[id(input_node)] = kwargs[name]

        # Execute in topological order
        for node in self._execution_order:
            parent_values = tuple(cache[id(p)] for p in node.parents)
            cache[id(node)] = node.operator._apply(*parent_values)

        # Collect outputs
        return {name: cache[id(node)] for name, node in self.outputs.items()}

    def _topological_sort(self) -> list[Node]:
        """Sort nodes in dependency order (Kahn's algorithm)."""
        ...

    def describe(self) -> str:
        """Pretty-print the graph structure."""
        lines = [f"Graph ({len(self.inputs)} inputs, {len(self.outputs)} outputs):"]
        lines.append(f"  Inputs: {list(self.inputs.keys())}")
        for i, node in enumerate(self._execution_order):
            parent_names = [p.name or f"node_{id(p)}" for p in node.parents]
            lines.append(f"  [{i}] {node.operator!r} ← {parent_names}")
        lines.append(f"  Outputs: {list(self.outputs.keys())}")
        return "\n".join(lines)

    def get_config(self) -> dict:
        """Serialize the graph structure for reproducibility."""
        ...
```

### Use Cases

The Graph API is specifically designed for workflows where Sequential falls short:

**Multi-input metrics** — the main motivation. Evaluate a prediction against a reference without manually wiring:

```python
pred = Input("prediction")
ref = Input("reference")
preprocessed = Sequential([ValidateCoords(), Regrid(grid)])(pred)
rmse = RMSE(variable="ssh", dims=["time"])(preprocessed, ref)
psd = PSDScore(variable="ssh", dims=["lat", "lon"])(preprocessed, ref)

eval_pipeline = Graph(
    inputs={"prediction": pred, "reference": ref},
    outputs={"rmse": rmse, "psd_score": psd},
)
```

**Branching** — apply different processing to the same input:

```python
raw = Input("data")
validated = ValidateCoords()(raw)
ssh_field = SelectVariables(["ssh"])(validated)
sst_field = SelectVariables(["sst"])(validated)
ssh_anom = RemoveClimatology(ssh_clim)(ssh_field)
sst_anom = RemoveClimatology(sst_clim)(sst_field)

pipeline = Graph(
    inputs={"data": raw},
    outputs={"ssh_anomaly": ssh_anom, "sst_anomaly": sst_anom},
)
```

**Merging** — combine results from different sources:

```python
obs = Input("observations")
model = Input("model_output")
obs_clean = Sequential([ValidateCoords(), Regrid(grid)])(obs)
model_clean = Sequential([ValidateCoords(), Regrid(grid)])(model)
bias = Bias(variable="ssh", dims=["time"])(model_clean, obs_clean)

pipeline = Graph(
    inputs={"observations": obs, "model_output": model},
    outputs={"obs": obs_clean, "model": model_clean, "bias": bias},
)
```

### Design Decisions

- **No special operator subclasses needed.** The dual-mode `__call__` (eager vs symbolic) is handled in the `Operator` base class. Every existing operator works in a Graph automatically.
- **Graph is an Operator.** This means graphs nest: a graph can be a node in a larger graph, or a step in a Sequential (if it has a single unnamed input/output).
- **Execution is eager and synchronous.** No lazy evaluation, no dask integration at the Graph level. The graph just determines execution order. Dask-awareness is a per-operator concern, deferred to a later version.
- **Dict-in, dict-out for multi-input/output.** Single-input/single-output graphs can also be called positionally: `graph(ds)` if there's only one input and one output, making them drop-in compatible with Sequential.


## Inference: Bring Your Own Model

geo_toolz is not just a preprocessing toolkit — it also serves as the inference backend. Any trained model can be wrapped as an `Operator` and slotted into a pipeline alongside preprocessing and evaluation steps. The library handles the xarray ↔ array marshalling; the user brings the model.

### `ModelOp` — the universal model wrapper

```python
class ModelOp(Operator):
    """Wrap any callable model as a geo_toolz Operator.

    Handles xarray → array conversion, prediction, and array → xarray
    reconstruction with coords/attrs preserved.

    Works with:
    - sklearn estimators (.predict / .predict_proba)
    - JAX/Equinox modules (pure functions or eqx.Module.__call__)
    - PyTorch models (via numpy round-trip)
    - Any callable: f(array) → array
    """

    def __init__(
        self,
        model,
        sample_dim: str = "time",
        feature_dims: list[str] | None = None,
        output_vars: list[str] | None = None,
        method: str = "predict",     # "predict", "predict_proba", "__call__", or callable
        batch_size: int | None = None,
    ):
        ...

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        # 1. Flatten spatial dims → 2D array (sample, features)
        # 2. Call model.predict / model.__call__ / custom method
        # 3. Reshape back to xarray with original coords
        ...
```

The key design decisions:

- **No framework dependency.** `ModelOp` never imports JAX, torch, or sklearn — it calls `getattr(model, method)` or `model(array)`. The user installs what they need.
- **Same Operator interface.** `ModelOp` composes with `Sequential`, `Graph`, `|` pipe, and `xrpatcher` exactly like any preprocessing operator.
- **Batch support.** For large grids or expensive models, `batch_size` chunks the flattened array and concatenates results — bounded memory without the user writing a loop.

### Framework-specific conveniences

For common backends, thin wrappers provide ergonomic defaults:

```python
# sklearn — delegates to .predict(), handles fitted state
from geo_toolz.inference import SklearnModelOp
pred = SklearnModelOp(fitted_ridge, sample_dim="time")(ds_clean)

# JAX/Equinox — jit-compiles, handles pytree params
from geo_toolz.inference import JaxModelOp
pred = JaxModelOp(eqx_model, sample_dim="time", jit=True)(ds_clean)

# Generic callable — any f(ndarray) → ndarray
from geo_toolz.inference import ModelOp
pred = ModelOp(my_func, sample_dim="time")(ds_clean)
```

These are all thin subclasses of `ModelOp` — they set sensible defaults for `method`, array dtype, and device transfer, but share the same xarray marshalling and composition logic.

### Inference in a Graph

Because `ModelOp` is an `Operator`, it participates in the functional Graph API:

```python
from geo_toolz.core import Input, Graph, Sequential
from geo_toolz.inference import SklearnModelOp, JaxModelOp
from geo_toolz.metrics import RMSE

raw = Input("features")
ref = Input("ground_truth")

cleaned = preprocess_pipeline(raw)

# Two competing models in the same graph
sklearn_pred = SklearnModelOp(fitted_rf, sample_dim="time")(cleaned)
jax_pred = JaxModelOp(neural_net, sample_dim="time")(cleaned)

sklearn_rmse = RMSE(variable="ssh", dims=["time"])(sklearn_pred, ref)
jax_rmse = RMSE(variable="ssh", dims=["time"])(jax_pred, ref)

comparison = Graph(
    inputs={"features": raw, "ground_truth": ref},
    outputs={
        "sklearn_pred": sklearn_pred,
        "jax_pred": jax_pred,
        "sklearn_rmse": sklearn_rmse,
        "jax_rmse": jax_rmse,
    },
)
```

## Package Layout

Following the `pypackage_template` conventions: `src/` layout, `uv` + `hatchling`, `ruff`, `ty`, `pytest`, `mkdocs`.

```
geo_toolz/
├── src/geo_toolz/
│   ├── __init__.py
│   ├── core.py                    # Operator, Sequential, Graph, Input, Node
│   │
│   ├── validation.py              # Public re-exports
│   ├── crs.py
│   ├── subset.py
│   ├── masks.py
│   ├── regrid.py
│   ├── interpolation.py
│   ├── detrend.py
│   ├── encoders.py
│   ├── discretize.py
│   ├── extremes.py
│   ├── spectral.py
│   ├── metrics.py
│   ├── kinematics.py
│   ├── sklearn.py                 # sklearn interop utilities
│   ├── inference.py               # ModelOp, SklearnModelOp, JaxModelOp
│   │
│   └── _src/                      # Layer 0 implementations + Layer 1 operators
│       ├── core/
│       │   ├── __init__.py
│       │   ├── operator.py        # Operator base class with dual-mode __call__
│       │   ├── sequential.py      # Sequential
│       │   ├── graph.py           # Node, Input, Graph
│       │   └── utils.py           # compose, identity, Lambda
│       ├── validation/
│       │   ├── __init__.py
│       │   ├── coords.py          # Layer 0: validate_longitude, etc.
│       │   └── operators.py       # Layer 1: ValidateCoords, etc.
│       ├── crs/
│       │   ├── __init__.py
│       │   ├── transforms.py      # Layer 0
│       │   └── operators.py       # Layer 1
│       ├── subset/
│       ├── masks/
│       ├── regrid/
│       ├── interpolation/
│       ├── detrend/
│       ├── encoders/
│       ├── discretize/
│       ├── extremes/
│       ├── spectral/
│       ├── metrics/
│       ├── kinematics/
│       ├── sklearn/
│       └── inference/             # ModelOp, framework-specific wrappers
│
├── tests/
│   ├── conftest.py                # Shared fixtures
│   ├── test_core.py
│   ├── test_validation.py
│   ├── test_regrid.py
│   ├── ...
│
├── docs/
│   ├── index.md
│   ├── getting_started.md
│   ├── concepts.md                # Operator model, Layer 0 vs 1
│   ├── api/                       # Auto-generated from docstrings
│   └── tutorials/                 # Jupyter notebooks
│       ├── 01_basic_pipeline.py
│       ├── 02_metrics.py
│       └── 03_hydra_integration.py
│
├── notebooks/
├── pyproject.toml
├── uv.lock
├── Makefile
├── mkdocs.yml
├── AGENTS.md
├── CODE_REVIEW.md
├── CHANGELOG.md
└── ...                            # Standard template files
```


## Dependencies

Core dependencies — required for all functionality:

| Package | Role |
|---|---|
| `numpy` | Array computation core |
| `scipy` | Interpolation, spatial algorithms, signal processing |
| `scikit-learn` | Nearest-neighbor regridding, spatial interpolation, preprocessing utilities |
| `xarray` | Labeled N-dimensional data interface |
| `pandas` | Time series operations, MultiIndex |

Geo dependencies — required for spatial operations:

| Package | Role |
|---|---|
| `rioxarray` | CRS assignment, reprojection, GeoTIFF I/O |
| `pyproj` | Coordinate reference system transforms |
| `regionmask` | Land/ocean/country masks from Natural Earth |

Analysis dependencies — required for spectral and metrics:

| Package | Role |
|---|---|
| `xrft` | Fourier transforms on xarray objects |
| `xskillscore` | Deterministic and probabilistic verification metrics |

Optional / domain-specific (not in core install):

| Package | Role |
|---|---|
| `metpy` | Atmospheric/oceanic physical calculations |
| `pint-xarray` | Unit-aware computation |
| `numba` | JIT compilation for custom kernels |
| `xarray_sklearn` | Full xarray-sklearn bridge with NaN policies, Pipeline/GridSearchCV compat |
| `xrpatcher` | Patch-wise tiling and reconstruction for large grids |
| `jax` + `equinox` | JAX-based model inference via `JaxModelOp` |
| `torch` | PyTorch model inference via `ModelOp` |
| `numpyro` | Bayesian posterior predictive inference |

The dependency philosophy: **numpy/scipy/sklearn for compute, xarray for interface, domain packages as extras.** No dependency should require system-level C library installation beyond what pip/uv can handle.


## Integration: xarray_sklearn and xrpatcher

The built-in `sklearn` submodule (`SklearnOp`, `to_2d` / `from_2d`) provides minimal numpy marshalling sufficient for inline use within geo_toolz pipelines. For heavier ML workflows, two optional companion libraries extend what is possible:

### xarray_sklearn — full sklearn bridge

[`xarray_sklearn`](../xarray_sklearn/README.md) provides a standalone `XarrayEstimator` wrapper that preserves dims, coords, and attrs through any sklearn estimator. It handles NaN policies (`"propagate"`, `"raise"`, `"mask"`), shape-changing transforms (e.g., PCA), Dataset column-concatenation, and full compatibility with `Pipeline` and `GridSearchCV`.

`SklearnOp` can delegate to `XarrayEstimator` when the package is installed, gaining NaN handling and richer metadata preservation for free:

```python
from geo_toolz.sklearn import SklearnOp
from sklearn.preprocessing import StandardScaler

# Built-in minimal wrapper (always available)
scale_op = SklearnOp(StandardScaler(), sample_dim="time")

# When xarray_sklearn is installed, SklearnOp can leverage it internally
# for NaN masking and metadata round-tripping — same Operator interface
scale_op = SklearnOp(StandardScaler(), sample_dim="time", nan_policy="mask")
```

### xrpatcher — patch-wise processing for large domains

[`xrpatcher`](https://github.com/jejjohnson/xrpatcher) tiles a DataArray into overlapping or non-overlapping spatial (or spatiotemporal) patches and reconstructs the full domain from processed patches. This is useful when the grid is too large to stack into a single 2-D array, or when locally-varying statistics are desired.

`xrpatcher` interacts with geo_toolz purely through xarray DataArrays — no adapter code is needed. Any `Operator` or `Sequential` pipeline can be applied per-patch:

```
              xrpatcher                      geo_toolz                    xrpatcher
          ┌──────────────┐              ┌─────────────────┐           ┌──────────────┐
DataArray │  extract N   │  patch_i     │  Operator /     │  out_i    │ reconstruct  │ DataArray
  (full)  │  patches     │ ──────────>  │  Sequential /   │ ───────>  │ from patches │  (full)
          │  {lat,lon,…} │  for each i  │  SklearnOp      │           │              │
          └──────────────┘              └─────────────────┘           └──────────────┘
```

### xarray_sklearn + xrpatcher — patch-wise ML

The two combine naturally: `xrpatcher` produces patches as `xr.DataArray`, and `XarrayEstimator` (or `SklearnOp`) consumes them. This enables per-region model fitting, chunked dimensionality reduction, and memory-bounded inference on large grids — all while preserving xarray metadata end-to-end.

```python
from xrpatcher import XRDAPatcher
from geo_toolz.sklearn import SklearnOp
from sklearn.decomposition import PCA

patcher = XRDAPatcher(da=ssh, patches={"lat": 32, "lon": 32}, strides={"lat": 32, "lon": 32})
pca_op = SklearnOp(PCA(n_components=5), sample_dim="time", new_feature_dim="mode")

regional_eofs = [pca_op(patcher[i]) for i in range(len(patcher))]
```

---

## CI / Quality Gates

| Check | Command | Scope |
|-------|---------|-------|
| Tests | `uv run pytest tests -x` | Full suite |
| Lint | `uv run ruff check .` | Entire repo |
| Format | `uv run ruff format --check .` | Entire repo |
| Typecheck | `uv run ty check src/geo_toolz` | Package only |

All four must pass before merge. GitHub Actions on push/PR.
Conventional commits required (`feat:`, `fix:`, `docs:`, `test:`, etc.).

**Build system:** hatchling (PEP 621), `src/` layout
**Python:** >= 3.12, < 3.14
**License:** MIT
