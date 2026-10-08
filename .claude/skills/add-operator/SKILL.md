---
name: add-operator
description: Add a pipekit Operator (and its numpy primitive) to a geotoolz operator family — radiometry, indices, mask, geom, viz, learn, … Use when asked to add, port or implement an operator, index, filter, transform or primitive in packages/geotoolz.
---

# Add a geotoolz operator

Read "The two contracts" in the root `AGENTS.md` (the pipekit `Operator`
and the `GeoTensor` carrier), then `packages/geotoolz/AGENTS.md` (family
layout, shared plumbing, parameter vocabulary). This skill is the
step-by-step.

## 1. Make sure it does not exist yet

- Search `docs/capabilities.md` for the concept (name, synonyms, the
  formula's usual name) and grep `packages/geotoolz/src` for the maths.
- If a close operator exists, extend it (a new parameter, a new mode) rather
  than adding a sibling. If another family already has the primitive, call it
  (`compositing` → `indices.ndvi`); never copy it.
- Pick the one family it belongs to. A new family needs the user's go-ahead.

## 2. Tier A — the primitive (`<family>/_src/array.py`)

- A pure numpy function: jaxtyping-annotated
  (`Num[np.ndarray, "*batch c h w"]` → `Float[np.ndarray, "*batch h w"]`),
  integer `*_idx` band positions, `axis: int = -3`, `eps` for denominators.
- No `GeoTensor`, no metadata, no nodata judgement, no operator state.
- Google docstring: the equation (`.. math::`), the physics in a few lines,
  typical values, a reference, `Args` / `Returns`.
- Export it from the family `__init__.py` (not from `geotoolz/__init__.py`).

## 3. Tier B — the Operator (`<family>/_src/operators.py`)

- Subclass `pipekit.Operator`; **keyword-only** constructor using the
  vocabulary table names (`red=`, `nir=`, `bands=`, `window=`, `size=`,
  `fill_value=`, `seed=`, …); store each argument on `self` under the same
  name so `get_config()` round-trips.
- `_apply(self, gt)`: resolve bands with `geotoolz._src.bands.resolve_band`,
  judge nodata with `geotoolz._src.valid` (`invalid_values`, `restore_fill`,
  `wrap_filled`), call the Tier-A primitive, rewrap with
  `geotoolz._src.wrap.wrap_like`. Promote integer DNs with
  `geotoolz._src.dtype.as_float` before arithmetic.
- Keep the contracts. `test_operator_contract.py` checks each of these:
  - **Constructor:** holds configuration only. A second raster is a
    positional `_apply(self, gt, other)` argument, checked with
    `geotoolz._src.geo.require_grid_match`.
  - **`get_config()`:** no override that only restates the constructor.
    Set `forbid_in_yaml = True` when the operator holds a callable, model
    or estimator.
  - **Non-carrier output** (a number, a table, `None`): set
    `_terminal = True`.
  - **Output fill** follows the output: `False` for masks, `0` for labels
    and counts, `NaN` for new float quantities, and
    `geotoolz._src.valid.carried_fill` for outputs that carry the input's
    values.
  - **`(T, C, H, W)` stacks:** handle them frame by frame
    (`geotoolz._src.shape.over_frames`), or reject them with an error that
    names the operator.
  - **Learned state:** `fit` / `transform` with trailing-underscore
    attributes, using `geotoolz._src.fitted.fit_once`.
- Need something two families share (a new band helper, a nodata rule)?
  Add it to `geotoolz/_src/`, not inline.
- Docstring: what it returns (shape, dtype, `fill_value_default`), `Args`,
  and an `Examples` block with a realistic call and a shape comment.
- Export the class from the family `__init__.py` **and** import it in
  `geotoolz/__init__.py` and add it to that `__all__` — every public
  Operator is top-level (`tests/test_geotoolz.py::test_public_operators_exported`).

## 4. Tests

- Family tests in `packages/geotoolz/tests/test_<family>.py`: the maths
  against a hand-computed or published value, nodata handling, a `GeoTensor`
  and a plain ndarray input, a time stack `(T, C, H, W)`.
- The contract suite (`tests/test_operator_contract.py`) discovers the class
  automatically; if its constructor needs arguments, register them in
  `CTOR_KWARGS` (or `RUNTIME_CTOR_KWARGS` for live objects).
- Mark anything slow or networked (`slow`, `integration`).

## 5. Docs and index

- Add it to the family page `docs/api/<family>.md`.
- `make capabilities` to regenerate `docs/capabilities.md`.
- Run the `pre-pr-check` skill.
