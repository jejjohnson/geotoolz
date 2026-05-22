# OB-1.2 — CF `standard_name` rename helpers (`rename_to_cf` / `rename_from_cf`)

**Source survey item:** [oceanbench-survey.md §B.1.1](oceanbench-survey.md)
**Status:** proposed
**Maps to upstream:** `rename_dataset_with_standard_names` from
[`mercator-ocean/oceanbench/oceanbench/core/climate_forecast_standard_names.py`](https://github.com/mercator-ocean/oceanbench/blob/main/oceanbench/core/climate_forecast_standard_names.py).

---

## 1. Motivation

Heterogeneous Datasets in ocean / climate workflows use wildly
different variable naming conventions:

| Source | SSH variable name |
|---|---|
| DUACS / CMEMS L4 | `sla` |
| GLORYS reanalysis | `zos` |
| GLO12 forecast | `zos` |
| MITgcm output | `Eta` |
| 4DVarNet | `ssh` |
| Mercator's CF normalizer | `sea_surface_height_above_geoid` |

xrtoolz already maintains a comprehensive
[`Variable`](../../src/xrtoolz/types/_src/variable.py) registry —
78+ entries each carrying a canonical short name (`ssh`, `u`, `v`,
`mld`, …) plus a CF `standard_name` (e.g.
`sea_surface_height_above_geoid`). The registry is the natural
authority for normalizing names across providers.

What's missing is a thin pair of helpers that *use* the registry to
normalize a Dataset:

1. **`rename_to_cf_standard_names(ds)`** — exact mirror of Mercator's
   one-liner: rename variables to their declared `standard_name` attr.
   Useful when *exporting* a Dataset to a downstream tool that expects
   CF-named variables.
2. **`rename_from_cf_standard_names(ds)`** — the inverse, registry-
   driven: rename variables from CF `standard_name` to xrtoolz
   canonical names. Useful when *ingesting* a CF-compliant Dataset
   (DUACS, GLORYS, GLO12, etc.) and normalizing into xrtoolz idiom.

Both are 1-line operations conceptually — but they're recurring,
worth a stable named API, and the `from_cf` direction is the high-
value addition (most pipelines start with "open a CF NetCDF →
normalize").

## 2. User stories

### 2.1 Ingest a CF-compliant GLORYS Dataset (primary)

> *I have a GLORYS reanalysis NetCDF with variables named
> `sea_surface_height_above_geoid`, `sea_water_potential_temperature`,
> etc. I want them renamed to xrtoolz canonical names so the rest of
> my pipeline composes cleanly.*

```python
import xarray as xr
from xrtoolz.geo import rename_from_cf_standard_names

ds = xr.open_dataset("glorys_reanalysis.nc")
ds = rename_from_cf_standard_names(ds)
# ds now has 'ssh', 'thetao', 'so', 'uo', 'vo', etc.
```

### 2.2 Export to a CF-strict downstream tool

> *I have an xrtoolz-canonical Dataset (`ssh`, `u`, `v`, `mld`) and
> need to hand it to a CF-strict reader.*

```python
from xrtoolz.geo import rename_to_cf_standard_names

ds_cf = rename_to_cf_standard_names(ds)
# Variables with standard_name attrs are renamed to those attrs.
```

### 2.3 Strict validation on ingest

> *I want the ingest to fail fast if the source uses a CF name we
> don't recognize.*

```python
ds = rename_from_cf_standard_names(ds, fallback="raise")
# raises KeyError listing the unknown CF names if any
```

### 2.4 As Layer-1 Operators inside a Sequential

```python
from xrtoolz.core import Sequential
from xrtoolz.geo import RenameFromCFStandardNames, BandpassWavelength

pipeline = Sequential([
    RenameFromCFStandardNames(),               # ingest normalization
    BandpassWavelength(...),                   # ODC-1.1
    # ...
])
```

## 3. What we already have / what's missing

| Capability | Current | This proposal |
|---|---|---|
| `Variable` registry with `standard_name` field | [`types/_src/variable.py`](../../src/xrtoolz/types/_src/variable.py) — 78+ entries | reuse |
| `validate_longitude` / `validate_latitude` | [`geo/_src/validation.py`](../../src/xrtoolz/geo/_src/validation.py) | unchanged |
| Generic `rename_coords` / `rename_variables` (dict-driven) | same file | unchanged |
| Data-var CF normalization driven by `standard_name` attr | — | **add** `rename_to_cf_standard_names` |
| Inverse normalization (CF → registry canonical) | — | **add** `rename_from_cf_standard_names` |
| Operator wrappers | — | **add** `RenameToCFStandardNames`, `RenameFromCFStandardNames` |

## 4. Design

### 4.1 Layer-0 functions

```python
# src/xrtoolz/geo/_src/validation.py — alongside existing rename helpers
def rename_to_cf_standard_names(
    ds: xr.Dataset, *,
    include_coords: bool = True,
) -> xr.Dataset:
    """Rename variables to their declared CF ``standard_name`` attribute.

    Mirror of ``mercator-ocean/oceanbench:rename_dataset_with_standard_names``.
    For every variable / coord with a non-empty ``standard_name`` attr,
    rename to that value. Variables without the attr are left unchanged.

    Parameters
    ----------
    ds
        Input Dataset.
    include_coords
        If True (default), rename coords too. Set False to limit to
        data_vars.

    Returns
    -------
    xr.Dataset
        Renamed Dataset.

    Raises
    ------
    ValueError
        If two source vars resolve to the same CF standard_name (rename
        collision).
    """
```

Implementation:

```python
def rename_to_cf_standard_names(ds, *, include_coords=True):
    candidates = list(ds.variables) if include_coords else list(ds.data_vars)
    mapping = {}
    for name in candidates:
        sn = ds[name].attrs.get("standard_name")
        if sn and sn != name:
            mapping[name] = sn
    # Detect collisions
    inverse = {}
    for src, tgt in mapping.items():
        if tgt in inverse:
            raise ValueError(
                f"rename_to_cf_standard_names: two source variables "
                f"({inverse[tgt]!r} and {src!r}) both map to "
                f"standard_name={tgt!r}; cannot rename both."
            )
        inverse[tgt] = src
    return ds.rename(mapping) if mapping else ds
```

```python
def rename_from_cf_standard_names(
    ds: xr.Dataset, *,
    fallback: Literal["passthrough", "raise"] = "passthrough",
    include_coords: bool = True,
) -> xr.Dataset:
    """Rename CF ``standard_name``-shaped variables to xrtoolz canonical names.

    Uses the :mod:`xrtoolz.types.Variable` registry as the authoritative
    ``standard_name → canonical_name`` mapping (78+ entries spanning
    ocean kinematics, atmospheric, cryospheric, and remote-sensing
    fields).

    Variables / coords whose name is not a registered CF standard_name
    pass through unchanged (``fallback="passthrough"``, default) or
    raise ``KeyError`` (``fallback="raise"``).

    Parameters
    ----------
    ds
        Input Dataset.
    fallback
        ``"passthrough"`` (default) leaves unrecognized vars unchanged.
        ``"raise"`` errors on the first unrecognized var.
    include_coords
        If True (default), rename coords too.
    """
```

Implementation:

```python
def rename_from_cf_standard_names(ds, *, fallback="passthrough",
                                   include_coords=True):
    cf_to_canonical = _build_cf_index()       # cached on first call
    candidates = list(ds.variables) if include_coords else list(ds.data_vars)
    mapping = {}
    unknown = []
    for name in candidates:
        if name in cf_to_canonical:
            canon = cf_to_canonical[name]
            if canon != name:
                mapping[name] = canon
        else:
            # Only flag as unknown if the name *looks like* a CF standard_name
            # (snake_case multi-word) — single-word names like "ssh" are
            # already canonical and shouldn't trigger a "raise".
            if "_" in name:
                unknown.append(name)
    if unknown and fallback == "raise":
        raise KeyError(
            f"rename_from_cf_standard_names: unknown CF standard_name(s) "
            f"{unknown!r}. Pass fallback='passthrough' to ignore, or "
            "extend the Variable registry."
        )
    return ds.rename(mapping) if mapping else ds


@functools.cache
def _build_cf_index() -> dict[str, str]:
    """Build standard_name → canonical_name index from the Variable registry."""
    from xrtoolz.types.variable import REGISTRY
    index = {}
    for var in REGISTRY:
        if var.standard_name and var.standard_name not in index:
            index[var.standard_name] = var.name
    return index
```

The `_build_cf_index` cache means the first call walks the registry
once; subsequent calls reuse the dict.

### 4.2 Layer-1 Operators

```python
# src/xrtoolz/geo/operators.py
class RenameToCFStandardNames(Operator):
    def __init__(self, *, include_coords: bool = True): ...
    def __call__(self, ds): return rename_to_cf_standard_names(ds, include_coords=self.include_coords)
    def get_config(self): ...

class RenameFromCFStandardNames(Operator):
    def __init__(self, *,
                 fallback: str = "passthrough",
                 include_coords: bool = True): ...
    def __call__(self, ds): return rename_from_cf_standard_names(
        ds, fallback=self.fallback, include_coords=self.include_coords)
    def get_config(self): ...
```

Standard pattern; round-trips through `get_config`.

### 4.3 Re-exports

```python
# src/xrtoolz/geo/__init__.py
from xrtoolz.geo._src.validation import (
    rename_to_cf_standard_names,
    rename_from_cf_standard_names,
    # ... existing exports
)
from xrtoolz.geo.operators import (
    RenameToCFStandardNames,
    RenameFromCFStandardNames,
    # ...
)
```

## 5. Library leverage

| Need | Library |
|---|---|
| Variable rename | `xarray.Dataset.rename` (built-in) |
| Registry lookup | `xrtoolz.types.Variable` (existing) |
| Cached index | `functools.cache` (stdlib) |

No new dependencies.

## 6. Public API surface

```python
# Layer-0 functions
xrtoolz.geo.rename_to_cf_standard_names(ds, *, include_coords=True)
xrtoolz.geo.rename_from_cf_standard_names(ds, *, fallback="passthrough",
                                            include_coords=True)

# Layer-1 Operators
xrtoolz.geo.RenameToCFStandardNames(*, include_coords=True)
xrtoolz.geo.RenameFromCFStandardNames(*, fallback="passthrough",
                                        include_coords=True)
```

## 7. Tests

| Test | Asserts |
|---|---|
| `rename_to_cf` renames vars with `standard_name` attr | exact mapping |
| `rename_to_cf` leaves vars without attr unchanged | identity |
| `rename_to_cf` `include_coords=False` | data vars renamed, coords untouched |
| `rename_to_cf` collision (two vars → same standard_name) | informative `ValueError` listing both source names |
| `rename_from_cf` on registry-known CF name (`sea_surface_height_above_geoid`) | renames to `ssh` |
| `rename_from_cf` `fallback="passthrough"` on unknown CF-shaped var | left unchanged |
| `rename_from_cf` `fallback="raise"` on unknown CF-shaped var | informative `KeyError` listing unknown names |
| `rename_from_cf` ignores already-canonical names like `"ssh"` | no rename, no error |
| `rename_from_cf` ↔ `rename_to_cf` round-trip on registry-known vars | identity |
| Operators round-trip via `get_config` | reconstructed produce identical output |

Target: ~10 cases.

## 8. Out of scope

- **Hard-coded CF enum types** (Mercator's `StandardVariable` enum)
  — our registry covers this; no parallel enum needed.
- **CF time / depth coord normalizers** — covered by existing
  `validate_longitude` / `validate_latitude` pattern; extend the
  registry for new coord types if needed, separately.
- **Standardizing units alongside names** — pint-aware unit conversion
  is its own concern; out of scope here.
- **Multi-name CF aliases** (e.g. CMIP `tos` → `sea_surface_temperature`)
  — `Variable.aliases` already supports this; expansion is a registry
  concern, not a helper concern.
- **`source="attrs|registry|both"` priority dial** — declined; two
  separate functions with clear semantics is cleaner than one
  multi-mode function.

## 9. Effort

≈40 LOC implementation + ≈60 LOC tests. Smallest item in the survey
so far.

| Slice | LOC |
|---|---|
| `rename_to_cf_standard_names` (incl. collision check) | 12 |
| `rename_from_cf_standard_names` (incl. `_build_cf_index` cache) | 18 |
| `RenameToCFStandardNames`, `RenameFromCFStandardNames` operators | 20 |
| Tests | ~60 |
| Docs / re-exports | 5 |

## 10. Risks / open questions

1. **Where it lives.** [`geo/_src/validation.py`](../../src/xrtoolz/geo/_src/validation.py)
   alongside `rename_coords` / `rename_variables` /
   `validate_longitude` / `validate_latitude`. Co-located naturally.
2. **`fallback` heuristic in `rename_from_cf`.** We only flag a name
   as "unknown CF" when it looks like one (`_` in name, multi-word).
   Single-word vars like `"ssh"` pass through silently — they're
   already canonical. Document.
3. **Registry coverage gaps.** First check: `ocean_mixed_layer_thickness`
   is already in the registry; full coverage of Mercator's
   `StandardVariable` enum spot-checked. Add any missing entries as
   a one-liner registry update bundled with this issue.
4. **Collision detection.** `rename_to_cf` raises on collision.
   `rename_from_cf` doesn't — multiple CF names *can* map to the same
   canonical (registry aliases), but the registry's `_build_cf_index`
   takes the first match per `standard_name` and dedupes implicitly.
5. **`include_coords=True` default.** Coord names like `"latitude"`
   carry `standard_name="latitude"` so `rename_to_cf` is a no-op for
   already-CF coords. For `rename_from_cf`, `"latitude"` would be
   recognized but our existing `validate_latitude` already canonicalizes
   to `"lat"` — not a conflict, just a different normalization path.
6. **Operator placement.** `geo/operators.py` (existing module).
