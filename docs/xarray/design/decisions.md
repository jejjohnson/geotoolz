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
