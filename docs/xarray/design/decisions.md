---
status: draft
version: 0.1.0
---

# Design Decisions

---

## D1: Everything is an Operator (uniform `__call__` interface)

**Status:** accepted

**Context:** Should preprocessing steps, models, and metrics share a common abstraction, or be separate systems?

**Options:**
- (A) Separate: preprocessing functions, model wrappers, metric functions — composed ad-hoc
- (B) Unified: everything is an `Operator` with `__call__`, composable via Sequential/Graph

**Decision:** Option B. A regridding step, a trained model, and a metric are all callable objects with the same interface. This enables `Sequential([preprocess, model, metric])` as a single pipeline.

**Consequences:**
- Any operator works in Sequential, Graph, and pipe syntax without special-casing
- Inference (ModelOp) is first-class, not a bolt-on
- Multi-input operators (metrics) require the Graph API for composition

---

## D2: Split-object pattern for stateful operations (no fit/transform)

**Status:** accepted

**Context:** Some operations need to learn from data (e.g., climatology). Should operators have `fit` / `transform` methods like sklearn?

**Options:**
- (A) `fit` / `transform` on Operator (sklearn pattern)
- (B) Separate learning and applying phases — `CalculateClimatology` returns state, `RemoveClimatology(state)` applies it

**Decision:** Option B. Every operator in a Sequential is `Dataset → Dataset`, always. State computation is explicit and happens upstream, not hidden inside the pipeline.

**Consequences:**
- Sequential stays simple — no `fit_transform` duality
- Learned state (a climatology, a scaler) is just an xarray object — saveable, inspectable
- The applying operator is Hydra-serializable if state is referenced by path

---

## D3: Dual-mode `__call__` for eager vs symbolic execution

**Status:** accepted

**Context:** Layer 2 Graph API requires operators to work in symbolic mode (building a DAG) as well as eager mode (executing on data). Should this require special operator subclasses?

**Options:**
- (A) Separate graph operators (`GraphOp`) vs regular operators
- (B) Dual-mode `__call__` — detect `Node` arguments automatically, return `Node` instead of executing

**Decision:** Option B. The detection is in the `Operator` base class. Every existing operator works in a Graph automatically with zero changes.

**Consequences:**
- No parallel operator hierarchy to maintain
- Graph is an Operator — it nests inside Sequential or larger Graphs
- Operators don't need to know about the graph system

---

## D4: No framework dependency for inference (ModelOp)

**Status:** accepted

**Context:** `ModelOp` wraps trained models from sklearn, JAX, PyTorch, etc. Should it import these frameworks?

**Options:**
- (A) Import each framework and provide typed wrappers
- (B) Framework-agnostic: call `getattr(model, method)` or `model(array)` — never import JAX/torch/sklearn

**Decision:** Option B. `ModelOp` never imports JAX, torch, or sklearn. It calls the model via duck typing. Framework-specific wrappers (`JaxModelOp`, `SklearnModelOp`) set ergonomic defaults but are thin subclasses.

**Consequences:**
- No transitive dependencies from inference
- Users install only what they need
- Same Operator interface regardless of backend

---

## D5: numpy/scipy/sklearn for compute, xarray for interface

**Status:** accepted

**Context:** What should the compute core be? JAX? Dask? Pure numpy?

**Options:**
- (A) JAX for everything (GPU, JIT, grad)
- (B) numpy/scipy/sklearn core, xarray interface, framework-agnostic inference
- (C) Dask-first for distributed computation

**Decision:** Option B. Preprocessing doesn't need GPU or autodiff. numpy/scipy/sklearn are universally available and fast enough. JAX/Dask enter only through inference backends or optional integrations.

**Consequences:**
- Zero-friction install (pip/uv, no system deps)
- No dask integration in v0.1 — operators work on in-memory arrays
- JAX acceleration available only through `JaxModelOp`

---

## D6: Graph is Dict-in, Dict-out

**Status:** accepted

**Context:** How should Graph handle multiple inputs and outputs?

**Decision:** `Graph(inputs={"name": Input}, outputs={"name": Node})`. Called with `graph(name=ds)`. Single-input/single-output graphs can be called positionally: `graph(ds)`.

**Consequences:**
- Multi-input operators (metrics taking prediction + reference) are first-class
- Drop-in compatible with Sequential when graph has one input/output
- Execution is eager and synchronous — no lazy evaluation

---

## D7: Metrics — own the implementation, two-layer (functions + Operator)

**Status:** accepted (resolved 2026-04-25)

**Context:** Should `xr_toolz.metrics` wrap `xskillscore`, depend on it optionally, or own the implementation?

**Options:**
- (A) Wrap xskillscore — small surface, inherits their tests, but no spectral / multiscale / masked-coverage variants and the API is function-style (no Operator)
- (B) Optional internal delegation — own the Operator API, fall through to xskillscore where it matches
- (C) Own it end-to-end as a two-layer module: pure-function skill scores at Layer 0, thin Operator wrappers at Layer 1

**Decision:** Option C.

- **Layer 0** — pure functions in `xr_toolz/metrics/_src/<family>.py` returning `xr.DataArray | xr.Dataset | float`. One file per family: `pixel.py` (RMSE, NRMSE, MAE, Bias, Correlation, Murphy, NSE, CRPS), `spectral.py` (PSDScore, ResolvedScale, Coherence-as-skill), `multiscale.py` (per-scale RMSE, wavelet-RMSE), `distributional.py` (KS, Wasserstein, energy distance), `masked.py` (mask-aware variants of the above).
- **Layer 1** — Operator wrappers in `xr_toolz/metrics/operators.py` (`RMSE`, `PSDScore`, `ResolvedScale`, …). Each is a thin call into the Layer 0 function with config carried on the operator. Multi-input: `__call__(prediction, reference) → DataArray | Dataset | float`.
- Custom user metrics: write a Layer 0 function with the standard signature, wrap once with `MetricOp(fn, **config)` (or a hand-authored Operator subclass).
- **No xskillscore dependency.** Implementations are short and well-known; tests pin them against analytic ground truth and (offline) against xskillscore for the overlapping subset.

**Consequences:**
- Single coherent Operator surface across pixel / spectral / multiscale / distributional metrics.
- Spectral and multiscale skill scores (the differentiator) sit naturally next to RMSE rather than in a parallel module.
- Custom skill scores have a low-friction extension path (write a function, optionally wrap).
- Test cost: ~10 pixel + ~5 spectral/multiscale + ~3 distributional implementations to author and verify, but each is small.
- Removes `xskillscore` from the dependency tree (it was never required, but D7 makes the choice explicit).

---

## D8: Encoders live under `transforms`, organized by what they encode

**Status:** accepted (resolved 2026-04-25)

**Context:** Coordinate encoders (`LonLatToCartesian`, `CyclicalTime`) and basis / feature encoders (`FourierFeatures`, `RandomFourierFeatures`, `PolynomialFeatures`) overlap conceptually with both `geo` (coordinate-aware) and `transforms` (basis expansion). Where do they live?

**Options:**
- (A) Coord encoders in `geo.encoders`, basis encoders in `transforms.encoders` — split by what they encode
- (B) Everything in `geo.encoders` — keeps `transforms` purely about signal transforms
- (C) Everything in `transforms.encoders` — single home, sub-organized by category

**Decision:** Option C. All encoders move under `xr_toolz.transforms.encoders`, sub-organized by category:

```
xr_toolz/transforms/encoders/
    coord_space.py    # LonLatToCartesian, GeocentricToENU, …
    coord_time.py     # CyclicalTimeEncoding, JulianDate, …
    basis.py          # FourierFeatures, RandomFourierFeatures, PolynomialFeatures
```

Rationale:
- One namespace to look in for "encode something into a feature representation".
- Mathematically, `FourierFeatures` and `DCT` are siblings (basis expansions); putting them in different top-level modules hides that.
- Sub-files (`coord_space`, `coord_time`, `basis`) preserve the conceptual split without forcing two parallel `encoders/` namespaces.

**Consequences:**
- `xr_toolz.geo._src/encoders.py` is removed; all encoder classes re-export from `transforms.encoders`.
- `xr_toolz.transforms` becomes the single home for: fourier / dct / wavelet / decompose / encoders.
- Existing imports from `xr_toolz.geo` need a one-time migration when this lands in code; design docs already reflect the new home.
- Future encoder families (e.g., spherical harmonic basis, learned positional encodings) get a natural home — likely a new sub-file under `transforms.encoders/`.

---

## D9: Domain stubs collapse into one `kinematics` submodule, sub-organized by domain

**Status:** accepted (resolved 2026-04-25)

**Context:** `xr_toolz` currently has empty stubs at `xr_toolz/atm/`, `xr_toolz/ocn/`, `xr_toolz/ice/`, `xr_toolz/rs/`. Each was intended to host derived physical-quantity operators (`GeostrophicVelocities`, `WindSpeed`, `NormalizedDifference`, etc.). Should they fill out as four parallel domain-named submodules, or collapse into one home?

**Options:**
- (A) Fill them — keep `atm/`, `ocn/`, `ice/`, `rs/` as top-level submodules, each with its own `kinematics`, `derived_variables`, etc. inside
- (B) Collapse into a single `xr_toolz.kinematics` submodule with one file per domain (`ocean.py`, `atmosphere.py`, `ice.py`, `remote.py`)

**Decision:** Option B.

```
xr_toolz/kinematics/_src/
    ocean.py
    atmosphere.py
    ice.py
    remote.py
```

Each sub-file follows the metrics two-layer pattern: Layer 0 pure functions + Layer 1 Operator wrappers.

Rationale:
- Removes nine "where does X go?" questions (`WindSpeed`-over-ocean, methane retrieval, sea-ice forcing on the atmosphere, etc.) by collapsing them into one cross-domain home with a clear disambiguation rule (the variable being *operated on* decides the file, not the variable being *produced*).
- Today's `atm/`, `ocn/`, `ice/`, `rs/` are empty namespaces — premature partitioning.
- One module surface to document; one place to look.
- A researcher who wants ocean physics imports `xr_toolz.kinematics.ocean.GeostrophicVelocities` — barely longer than `xr_toolz.ocn.GeostrophicVelocities` and the home is unambiguous.

**Consequences:**
- The `xr_toolz/atm/`, `xr_toolz/ocn/`, `xr_toolz/ice/`, `xr_toolz/rs/` packages are removed. Existing (currently minimal) code in `ocn/` migrates into `kinematics/_src/ocean.py`.
- A new `xr_toolz.kinematics` top-level module is reserved.
- Cross-domain operators that genuinely don't fit one file (rare) get a `kinematics/_src/shared.py`.
- Future domain growth (e.g., a `methane.py` if methane-retrieval operators get plentiful) is a new file in the same module, not a new top-level submodule.

---

## D10: Viz operators are first-class `Operator`s that return `Figure` / `Axes`

**Status:** accepted (resolved 2026-04-25)

**Context:** Plotting (`PlotMap`, `PlotSpectrum`, `PlotTimeseries`, `QuicklookPanel`, etc.) produces `matplotlib.Figure` / `Axes`, not `xr.Dataset`. The base contract states single-input operators are `Dataset → Dataset`. How are viz operators integrated?

**Options:**
- (A) Viz are `Operator` subclasses that return `Figure` / `Axes`. Documented exception to `Dataset → Dataset`. Compose inside `Sequential` (as terminal nodes) and `Graph` (as one of N outputs)
- (B) Separate `Plotter` protocol — viz lives outside the `Operator` system, called after a pipeline runs
- (C) Mutating viz — `Operator` returns the Dataset unchanged with a side-effecting figure

**Decision:** Option A.

- `xr_toolz.viz` operators are `Operator` subclasses with `__call__(ds) → matplotlib.Figure | matplotlib.Axes`.
- The `Operator` contract (architecture.md) is amended: terminal viz operators are an explicit exception to `Dataset → Dataset`.
- They compose inside `Graph` as terminal output nodes — the motivating use case is end-to-end evaluation graphs that emit both scalar scores and figures from one symbolic computation: `Graph(inputs={"pred": …, "ref": …}, outputs={"rmse": rmse_node, "psd_score": psd_node, "psd_fig": plot_psd_node})`.
- They compose inside `Sequential` only as the **last** step. A `Sequential` that emits a non-`Dataset` from a non-final step is a runtime error.

Rationale:
- Real end-to-end pattern: sequential preprocessing → graph that branches into both metrics and figures. Forcing viz into a parallel surface (Option B) means hand-wiring the figure side, defeating the symbolic-graph payoff.
- Option C (mutating, attaching figures to `attrs`) is a memory hazard and surprising — rejected.
- Option A's downside (the contract gets one exception) is small and well-localized to one module.

**Consequences:**
- `xr_toolz/viz/` is a new top-level submodule.
- `Sequential` validates that any non-`Dataset` return appears only at the final step; otherwise raises a clear error.
- `Graph` already supports heterogeneous output types — no change needed.
- Documented operator-contract exception in [architecture.md §Operator](architecture.md): "Terminal viz operators may return `Figure` / `Axes`".
- Plot operators carry their config (`figsize`, `cmap`, `projection`, …) so they're hydra-serializable like any other operator.

