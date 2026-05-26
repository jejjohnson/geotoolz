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

**Context:** Should `xrtoolz.metrics` wrap `xskillscore`, depend on it optionally, or own the implementation?

**Options:**
- (A) Wrap xskillscore — small surface, inherits their tests, but no spectral / multiscale / masked-coverage variants and the API is function-style (no Operator)
- (B) Optional internal delegation — own the Operator API, fall through to xskillscore where it matches
- (C) Own it end-to-end as a two-layer module: pure-function skill scores at Layer 0, thin Operator wrappers at Layer 1

**Decision:** Option C.

- **Layer 0** — pure functions in `xrtoolz/metrics/_src/<family>.py` returning `xr.DataArray | xr.Dataset | float`. One file per family: `pixel.py` (RMSE, NRMSE, MAE, Bias, Correlation, Murphy, NSE, CRPS), `spectral.py` (PSDScore, ResolvedScale, Coherence-as-skill), `multiscale.py` (per-scale RMSE, wavelet-RMSE), `distributional.py` (KS, Wasserstein, energy distance), `masked.py` (mask-aware variants of the above).
- **Layer 1** — Operator wrappers in `xrtoolz/metrics/operators.py` (`RMSE`, `PSDScore`, `ResolvedScale`, …). Each is a thin call into the Layer 0 function with config carried on the operator. Multi-input: `__call__(prediction, reference) → DataArray | Dataset | float`.
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

**Decision:** Option C. All encoders move under `xrtoolz.transforms.encoders`, sub-organized by category:

```
xrtoolz/transforms/encoders/
    coord_space.py    # LonLatToCartesian, GeocentricToENU, …
    coord_time.py     # CyclicalTimeEncoding, JulianDate, …
    basis.py          # FourierFeatures, RandomFourierFeatures, PolynomialFeatures
```

Rationale:
- One namespace to look in for "encode something into a feature representation".
- Mathematically, `FourierFeatures` and `DCT` are siblings (basis expansions); putting them in different top-level modules hides that.
- Sub-files (`coord_space`, `coord_time`, `basis`) preserve the conceptual split without forcing two parallel `encoders/` namespaces.

**Consequences:**
- `xrtoolz.geo._src/encoders.py` is removed; all encoder classes re-export from `transforms.encoders`.
- `xrtoolz.transforms` becomes the single home for: fourier / dct / wavelet / decompose / encoders.
- Existing imports from `xrtoolz.geo` need a one-time migration when this lands in code; design docs already reflect the new home.
- Future encoder families (e.g., spherical harmonic basis, learned positional encodings) get a natural home — likely a new sub-file under `transforms.encoders/`.

---

## D9: Domain stubs collapse into one `kinematics` submodule, sub-organized by domain

**Status:** accepted (resolved 2026-04-25)

**Context:** `xrtoolz` currently has empty stubs at `xrtoolz/atm/`, `xrtoolz/ocn/`, `xrtoolz/ice/`, `xrtoolz/rs/`. Each was intended to host derived physical-quantity operators (`GeostrophicVelocities`, `WindSpeed`, `NormalizedDifference`, etc.). Should they fill out as four parallel domain-named submodules, or collapse into one home?

**Options:**
- (A) Fill them — keep `atm/`, `ocn/`, `ice/`, `rs/` as top-level submodules, each with its own `kinematics`, `derived_variables`, etc. inside
- (B) Collapse into a single `xrtoolz.kinematics` submodule with one file per domain (`ocean.py`, `atmosphere.py`, `ice.py`, `remote.py`)

**Decision:** Option B.

```
xrtoolz/kinematics/_src/
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
- A researcher who wants ocean physics imports `xrtoolz.kinematics.ocean.GeostrophicVelocities` — barely longer than `xrtoolz.ocn.GeostrophicVelocities` and the home is unambiguous.

**Consequences:**
- The `xrtoolz/atm/`, `xrtoolz/ocn/`, `xrtoolz/ice/`, `xrtoolz/rs/` packages are removed. Existing (currently minimal) code in `ocn/` migrates into `kinematics/_src/ocean.py`.
- A new `xrtoolz.kinematics` top-level module is reserved.
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

- `xrtoolz.viz` operators are `Operator` subclasses with `__call__(ds) → matplotlib.Figure | matplotlib.Axes`.
- The `Operator` contract (architecture.md) is amended: terminal viz operators are an explicit exception to `Dataset → Dataset`.
- They compose inside `Graph` as terminal output nodes — the motivating use case is end-to-end evaluation graphs that emit both scalar scores and figures from one symbolic computation: `Graph(inputs={"pred": …, "ref": …}, outputs={"rmse": rmse_node, "psd_score": psd_node, "psd_fig": plot_psd_node})`.
- They compose inside `Sequential` only as the **last** step. A `Sequential` that emits a non-`Dataset` from a non-final step is a runtime error.

Rationale:
- Real end-to-end pattern: sequential preprocessing → graph that branches into both metrics and figures. Forcing viz into a parallel surface (Option B) means hand-wiring the figure side, defeating the symbolic-graph payoff.
- Option C (mutating, attaching figures to `attrs`) is a memory hazard and surprising — rejected.
- Option A's downside (the contract gets one exception) is small and well-localized to one module.

**Consequences:**
- `xrtoolz/viz/` is a new top-level submodule.
- `Sequential` validates that any non-`Dataset` return appears only at the final step; otherwise raises a clear error.
- `Graph` already supports heterogeneous output types — no change needed.
- Documented operator-contract exception in [architecture.md §Operator](architecture.md): "Terminal viz operators may return `Figure` / `Axes`".
- Plot operators carry their config (`figsize`, `cmap`, `projection`, …) so they're hydra-serializable like any other operator.

---

## D11: Two-tier type contract — xarray (Layer 0) → Operator (Layer 1)

**Status:** accepted (revised 2026-05-26)

**Historical context:** Originally accepted as a three-tier contract (array / xarray / Operator) on 2026-04-27. The array tier was a public surface for raw-numpy users at `xrtoolz.<module>.array`. After PRs α / β / γ (#206 / #208 / #209) flipped Layer 0 to DataArray-positional, the array tier added more surface than it earned — it duplicated Layer 0's signatures with a different positional / keyword convention and forced every public-surface review to discuss "should this also exist as `axis=` ?". The array tier is demoted to private `_*_kernels.py` modules. Users who want raw numpy can either drop into the private kernels (no stability promise) or pre-convert DataArrays.

**Context:** Layer 0 signatures across the package drift between `numpy`, `xr.DataArray`, and `xr.Dataset`. Without a clear contract, fourier / dct / wavelet / metrics / viz Layer 0 functions take a DataArray, kinematics functions take a Dataset, and users have no consistent expectation.

**Decision:** Two public tiers, plus a private kernel layer.

- **Layer 0 — xarray** (public): per-module functions in `xrtoolz/<module>/_src/<name>.py`. The Layer 0 primitive is **DataArray-positional**: one variable in, one variable out (`pred`, `ref`, …, `*, dim=`). Multi-variable physics primitives (kinematics, balances) take multiple DataArrays positionally plus `*, dim=`. Layer 0 dispatches to the private numpy / scipy kernels via `xr.apply_ufunc` and adds coord/attr handling.
- **Layer 1 — Operator** (public): input is always `xr.Dataset` (or multiple Datasets for multi-input operators). Output is **usually** `xr.Dataset` for transformations that preserve the dataset shape, but reduction-style operators (e.g., metrics) may return `xr.DataArray` or scalar, and terminal viz operators return `matplotlib.Figure` / `Axes` (D10). Operators carry Dataset-level selectors (`variable=`, `u_var=`, …) and delegate to Layer 0.
- **Private kernels — `_*_kernels.py`** (implementation detail, no stability guarantees): pure-array entry points (numpy / scipy) used internally by Layer 0 via `xr.apply_ufunc`. They live as `_*_kernels.py` siblings inside `_src/` — not under a `.array` namespace. No re-export from any package root.

Rationale:
- One public surface per scientific operation removes a category of review question ("should this also exist at the array tier?") and a category of test surface (forced numerical equivalence between tiers).
- The xarray Layer 0 contract is now unambiguous: DataArray-positional, one in / one out (or multi-in / one out for physics).
- The Operator contract has a uniform *input* shape (Dataset(s)); outputs may narrow (DataArray / scalar for reductions; Figure / Axes for terminal viz) without breaking composition because `Sequential` and `Graph` enforce that narrowed outputs only appear at terminal nodes.
- Numpy / JAX / numba / CuPy users who want a raw-array entry point can either pre-convert (`da.values` in, wrap the result back as a DataArray) or import from `_src/_*_kernels.py` and accept the no-stability-promise contract.

**Modules where private kernels are not meaningful** (`validation`, `crs`, `subset`, `masks`, whose math is inherently coord/attr-manipulation rather than arithmetic): they have no kernel module. Layer 0 takes `xr.Dataset` directly. The per-module section in `api/components.md` documents this.

**Consequences:**
- No public `xrtoolz.<module>.array` namespace. Anyone who was importing from there should pre-convert or drop into `xrtoolz.<module>._src._<name>_kernels` (private, no stability).
- Tests are split by layer: `tests/<module>/test_layer0.py` / `test_operators.py`. Kernel-level numerical tests still exist, but they test the kernel as an implementation detail rather than as a contract.
- Documentation: the Type Contract section in `architecture.md` codifies the rule, and each module's `components.md` entry shows the two public tiers.
- `xskillscore`-style numpy paths inside metrics live in the private kernel modules; D7 (own the implementation rather than depending on `xskillscore`) is unaffected — the implementation just isn't surfaced publicly any more.

---

## D12: `interpolate` — unified resampling, aggregation, and smoothing module

**Status:** accepted (structural decision 2026-04-27; open questions resolved 2026-05-04)

**Context:** The v0.1 design split value resampling across three small modules: `regrid` (grid → grid), `interpolation` (gap fill + time resample), and `discretize` (binning). They were artificially separated; in practice users reach for "the interpolation module" as one concept. Adjacent operations — vertical coord remapping, temporal smoothing, learned super-resolution — have no clear home today (`detrend.LowpassFilter` is the canonical orphan).

**Options:**

- (A) Keep the three modules separate, add new modules per concern (`coord_remap`, `smooth`, `downscale`). Six top-level modules for one conceptual space.
- (B) Collapse into a single `xrtoolz.interpolate` module organized by source/target structure (grid↔grid, grid↔points, points→grid, in-place gap fill) plus axis-specific submodules (`coord_remap`, `resample`, `smooth`, `downscale`).
- (C) Two-module split: `interpolate` (deterministic) + `downscale` (learned). Cleaner separation of deterministic vs ML, but splits closely-related concepts and creates a parallel hierarchy.

**Decision (structural):** Option B.

```
xrtoolz/interpolate/
    _src/
        grid_to_grid.py    # Regrid, Coarsen (deterministic aggregation), Refine (deterministic interpolation)
        grid_to_points.py  # SampleAtPoints, AlongTrack
        points_to_grid.py  # ScatterToGrid, Kriging
        binning.py         # Bin2D, BinND, Bin2DTime
        gap_fill.py        # FillNaN, FillNaNRBF, FillNaNKriging
        coord_remap.py     # generic RemapAxis + vertical presets (ToSigma, ToIsopycnal, ToPressureLevels, …) and temporal preset (ToPhase)
        resample.py        # Resample (down), Upsample
        smooth.py          # MovingAverage, GaussianSmooth, LowpassFilter (KalmanSmoother → future assimilate.smooth, see resolution 3 below)
        downscale.py       # Downscale, Upscale (both wrap a ModelOp)
```

Naming convention:

- **Deterministic refinement** is `Refine`; **learned refinement** (super-resolution) is `Downscale`.
- **Deterministic aggregation** is `Coarsen`; **learned aggregation** (subgrid-scale surrogates) is `Upscale`.

`coord_remap` is generalized: vertical coord remapping (depth ↔ σ ↔ isopycnal ↔ pressure-level) is the canonical usage, but the same primitive `RemapAxis` handles temporal phase remapping, curvilinear-orthogonal coord transforms, and Lagrangian ↔ Eulerian rebinning. Named subclasses are presets over the generic operator.

Modules outside `interpolate` that handle adjacent concerns: `crs.Reproject` (CRS-aware regridding — calls into `interpolate.Regrid` internally), `transforms.encoders.coord_{space,time}` (coord *relabeling*, not value resampling).

**Open-question resolutions** (2026-05-04, F3.5):

1. **Super-resolution patch tiling — resolved.** `Downscale` is a pure `ModelOp` wrapper with no `patch_size` / `overlap` constructor args. Tiling is delegated to `xrpatcher` upstream of the operator. Rationale: keeps `Downscale` orthogonal to tiling strategy and avoids duplicating xrpatcher's API surface.
2. **Data fusion home — deferred.** No fusion code lands in F3; revisit when the first fusion operator is proposed. `interpolate.fusion` remains the working assumption for deterministic-only fusion, but no commitment until `assimilate` exists.
3. **`KalmanSmoother` home — resolved.** `KalmanSmoother` is **out of scope for `interpolate.smooth`**. It lives under future `assimilate.smooth` because it requires a state-space model. `interpolate.smooth` is restricted to deterministic, non–state-space smoothers (`MovingAverage`, `GaussianSmooth`, `LowpassFilter`); these still take parameters (window, sigma, cutoff), but none requires fitting a model from data.
4. **`coord_remap` preset scope — resolved.** Ship vertical (`ToSigma`, `FromSigma`, `ToIsopycnal`, `ToPressureLevels`, `ToHeight`) + temporal (`ToPhase`) only. `ToTropopauseRelative`, `ToBoundaryLayerCoord`, and other domain-specific presets are deferred — add them on demand as new issues, each as a thin subclass over the generic `RemapAxis`.

**Consequences:**
- `xrtoolz.regrid`, `xrtoolz.interpolation`, `xrtoolz.discretize` are removed in favor of `xrtoolz.interpolate`. Pre-1.0 design doc — no compatibility shim planned.
- `detrend.LowpassFilter` migrates to `interpolate.smooth.LowpassFilter`. `detrend` becomes climatology-only.
- `Downscale` introduces a soft `ModelOp` dependency in `interpolate`. `ModelOp` itself has no framework dep (per D4), so the inference module is the only transitive surface added.
- Two public tiers per D11 throughout (xarray Layer 0 + Operator Layer 1). The underlying private numpy / scipy kernels are rich here — most algorithms are pure array math (linear / cubic / RBF / kriging / FFT-based filters).

