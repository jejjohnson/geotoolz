# OB-1.4 — Class-IV-style profile validation primitives

**Source survey item:** [oceanbench-survey.md §B.1.11](oceanbench-survey.md)
**Status:** proposed
**Maps to upstream:** `classIV.py` + `classIV_support.py` (~630 LOC) from
[`mercator-ocean/oceanbench/oceanbench/core/`](https://github.com/mercator-ocean/oceanbench/tree/main/oceanbench/core).

---

## 1. Motivation

Validation against in-situ profile observations (Argo floats, Class-IV
moored buoys, drifter velocity fixes) is a canonical operational-ocean
diagnostic. Mercator's pipeline does the full job:

```text
forecast cube (first_day, lead_day, depth, lat, lon)
   ┓ → SSH→SLA via MDT subtraction (variable-specific)
   ┃ → horizontal interp model onto observation (lat, lon) per (first_day, lead_day)
   ┃ → vertical interp (variable-specific: bracket-linear for u/v/T/SSH;
   ┃                    NaN-aware cubic spline for everything else)
   ┃ → groupby (depth_bin, lead_day) → RMSD
   ┛ → pivot to (variable × depth_bin) × lead_day, formatted scoreboard
```

xrtoolz today has the **2-D surface** version of grid→points
(ODC-1.2 `sample_at_points`) and **2-D spatial** RMSD bins (ODC-1.4
`bin_residuals_2d`). Neither handles depth-stratified profiles.

This issue extracts the **generic primitives** from Mercator's pipeline
— vertical interp, depth-aware grid→profile interp, depth-bin
assignment, and an RMSD pivot scoreboard — and ships them as
composable Layer-0/Layer-1 functions. The full Mercator driver
(forecast-cube schema + variable-specific MDT/dispatch) becomes a
**recipe page** that wires the primitives together, not a baked-in
Operator.

This keeps the surface generic — anyone with `(time, lat, lon, depth)`
scattered observations and a gridded model can validate, regardless of
the forecast-cube convention.

## 2. User stories

### 2.1 Validate a 4-D model against profile observations (primary)

> *I have a model `(time, depth, lat, lon)` and an Argo profile
> Dataset (one row per profile-depth observation, with `time`, `lat`,
> `lon`, `depth`, `temperature` columns). I want a scoreboard.*

```python
import pandas as pd
import xarray as xr
from xrtoolz.interpolate import interp_grid_to_profiles, assign_depth_bins
from xrtoolz.metrics import rmsd_scoreboard

ds_model = xr.open_dataset("glorys_thetao.nc")        # (time, depth, lat, lon)
df_obs   = pd.read_parquet("argo_profiles.parquet")   # rows: time, lat, lon, depth, observation_value

df_obs["model_value"] = interp_grid_to_profiles(
    ds_model["thetao"],
    df_obs,
    vertical_method="spline",
)
df_obs["depth_bin"] = assign_depth_bins(df_obs["depth"])
df_obs["variable"]  = "sea_water_potential_temperature"

scoreboard = rmsd_scoreboard(
    df_obs,
    index_cols=("variable", "depth_bin"),
    columns_col="time_bin",          # or "lead_day", or any categorical column
)
print(scoreboard)
# Sea water potential temperature (degC) [sea_water_potential_temperature]{0-5m}    0.42
# Sea water potential temperature (degC) [sea_water_potential_temperature]{5-100m}  0.31
# ...
```

### 2.2 Variable-specific vertical interp dispatch (Mercator convention)

> *Use bracket-linear interp for u/v/T/SSH (sharp-gradient handling)
> and cubic spline for everything else.*

```python
SHARP_GRADIENT_VARS = {"thetao", "uo", "vo", "ssh"}

def model_value_with_dispatch(model_da, df_obs, variable):
    method = "bracket" if variable in SHARP_GRADIENT_VARS else "spline"
    return interp_grid_to_profiles(model_da, df_obs, vertical_method=method)
```

### 2.3 Custom depth bins

```python
my_bins = {
    "surface":   (0, 10),
    "thermocline": (10, 200),
    "deep":      (200, 2000),
}
df["depth_bin"] = assign_depth_bins(df["depth"], bins=my_bins)
```

### 2.4 Operators in a Sequential

```python
from xrtoolz.core import Sequential
from xrtoolz.interpolate import InterpGridToProfiles
from xrtoolz.metrics import RMSDScoreboard

pipeline = Sequential([
    InterpGridToProfiles(observations=df_obs, vertical_method="spline"),
    # → DataFrame with model_value column
    RMSDScoreboard(index_cols=("variable", "depth_bin"), columns_col="lead_day"),
])
scoreboard = pipeline(ds_model)
```

### 2.5 2-D model passes through (compat with ODC-1.2)

```python
# Same call, model has no depth dim → equivalent to sample_at_points
df_obs["model_value"] = interp_grid_to_profiles(ds_2d["ssh"], df_obs)
```

## 3. What we already have / what's missing

| Capability | Current | This proposal |
|---|---|---|
| 2-D `sample_at_points` (horizontal) | proposed in ODC-1.2 | reuse / extend |
| 2-D `bin_residuals_2d` (lat/lon RMSD) | proposed in ODC-1.4 | unchanged |
| `Variable` registry (display name / units / standard_name) | [`types/_src/variable.py`](../../src/xrtoolz/types/_src/variable.py) | reuse for label formatter |
| Vertical bracket-linear interp | — | **add** (~30 LOC) |
| Vertical NaN-aware cubic spline interp | — | **add** (~50 LOC) |
| Depth-aware grid→profile interp (3-D) | — | **add** `interp_grid_to_profiles` |
| Depth-bin categorical assignment | — | **add** `assign_depth_bins` |
| RMSD pivot scoreboard with `(var × depth) × lead` shape + formatted labels | — | **add** `rmsd_scoreboard` |
| Operator wrappers | — | **add** `InterpGridToProfiles`, `RMSDScoreboard` |
| Class-IV driver (forecast cube + MDT + dispatch) | — | **doc/recipe**, not a baked-in operator |

## 4. Design

### 4.1 What we extract vs what we leave behind

We ship **generic primitives** that compose into the Mercator pipeline
*and* into other validation patterns. We don't ship the
`rmsd_class4_validation` driver because its forecast-cube schema
(`first_day_datetime`, `lead_day_index`), MDT-subtraction, and
variable-specific dispatch are operational-Mercator-specific.

### 4.2 Tier A — vertical interp kernels

```python
# src/xrtoolz/interpolate/_src/array_vertical.py — new module
def interp_vertical_bracket(
    profiles: ArrayLike,            # shape (n_depths, n_obs)
    depths: ArrayLike,              # shape (n_depths,)
    targets: ArrayLike,             # shape (n_obs,)
) -> NDArray[np.floating]:
    """NaN-aware linear interp between bracketing model depths.

    For each obs ``i``: find the two model depths bracketing
    ``targets[i]``, linearly interpolate between
    ``profiles[lower, i]`` and ``profiles[upper, i]``. NaN if either
    bracket is NaN, or if target is outside ``[min(depths), max(depths)]``.

    Vectorised via ``np.searchsorted``. ~30 LOC.
    """
```

```python
def interp_vertical_spline(
    profiles: ArrayLike,
    depths: ArrayLike,
    targets: ArrayLike,
    *,
    bc_type: str = "natural",
) -> NDArray[np.floating]:
    """NaN-aware cubic-spline interp via NaN-pattern grouping.

    Groups observations by their model-profile NaN bitmask (int64),
    runs one ``scipy.interpolate.CubicSpline`` per group. Efficient
    for sparse NaN patterns.

    Cap of 64 model depths (bitmask encoding); raises ValueError on
    overflow with a hint to reduce levels or use ``"bracket"`` method.
    """
```

Both are pure-numpy / scipy, no xarray. The bitmask grouping trick
(from Mercator) avoids per-observation spline construction when many
observations share the same NaN pattern — typical for model output
where `np.isnan(profiles[:, i])` patterns cluster by dataset gaps.

### 4.3 Tier B — grid→profile interp

```python
# src/xrtoolz/interpolate/_src/grid_to_profile.py — new module
def interp_grid_to_profiles(
    model_data: xr.Dataset | xr.DataArray,
    observations: xr.Dataset | pd.DataFrame,
    *,
    vertical_method: Literal["linear", "bracket", "spline"] = "linear",
    horizontal_method: str = "linear",          # forwarded to xr.interp
    lon: str = "longitude",
    lat: str = "latitude",
    depth: str = "depth",
    observation_dim: str = "observation",
) -> xr.DataArray | np.ndarray:
    """Interpolate gridded model fields onto scattered profile observations.

    Generalises ODC-1.2 ``sample_at_points`` to depth-aware profile
    observations.

    - **2-D model** (no depth dim): reduces to ``sample_at_points``;
      vertical_method ignored.
    - **3-D model** (with depth): horizontal interp at every model
      depth, then vertical interp per observation.

    Returns same shape as ``observations`` (Dataset → Dataset, DataFrame →
    Series with ``model_value`` name).
    """
```

Implementation outline:

```python
def interp_grid_to_profiles(model_data, observations, *,
                             vertical_method="linear", lon, lat, depth,
                             observation_dim, horizontal_method):
    # 1. Coerce observations → DataFrame-like with lat/lon/depth columns
    # 2. If model has no depth dim → sample_at_points fast-path
    if depth not in model_data.dims:
        return sample_at_points(model_data, ...)
    # 3. Horizontal interp per model depth
    horizontal = model_data.interp(
        {lon: xr.DataArray(obs_lon, dims=observation_dim),
         lat: xr.DataArray(obs_lat, dims=observation_dim)},
        method=horizontal_method,
    )  # shape (n_depths, n_obs)
    # 4. Vertical interp per obs
    method_fn = {
        "linear": _interp_vertical_linear,    # delegates to xr.interp on depth
        "bracket": interp_vertical_bracket,
        "spline": interp_vertical_spline,
    }[vertical_method]
    return method_fn(horizontal.values, model_data[depth].values, obs_depth)
```

### 4.4 Depth-bin assignment

```python
# src/xrtoolz/interpolate/_src/binning.py — extend existing module
DEPTH_BINS_DEFAULT: dict[str, tuple[float, float]] = {
    "0-5m":     (0, 5),
    "5-100m":   (5, 100),
    "100-300m": (100, 300),
    "300-600m": (300, 600),
}

def assign_depth_bins(
    depth_values: ArrayLike,
    bins: dict[str, tuple[float, float]] = DEPTH_BINS_DEFAULT,
) -> NDArray:
    """Assign categorical depth-bin labels per depth value.

    Values outside any bin → ``None``. Right-edge inclusive, left-edge
    inclusive (matches Mercator convention).
    """
```

~15 LOC via `pandas.cut` or numpy `digitize`.

### 4.5 RMSD scoreboard pivot

```python
# src/xrtoolz/metrics/_src/scoreboard.py — new module
def rmsd_scoreboard(
    df: pd.DataFrame, *,
    model_col: str = "model_value",
    obs_col: str = "observation_value",
    index_cols: Sequence[str] = ("variable", "depth_bin"),
    columns_col: str = "lead_day",
    label_format: str = "{display_name} ({units}) [{standard_name}]{{{depth_bin}}}",
    registry: dict | None = None,         # defaults to xrtoolz Variable registry
) -> pd.DataFrame:
    """Generic RMSD pivot scoreboard.

    Long-form DataFrame with (variable, depth_bin, lead_day,
    model_value, observation_value) → pivot of RMSD with rows =
    formatted ``(variable × depth_bin)`` labels and columns = lead_day
    values.

    Variable display_name / units / standard_name resolved via the
    ``xrtoolz.types.Variable`` registry; falls back to raw column
    values if a variable isn't registered.
    """
```

Implementation:

```python
def rmsd_scoreboard(df, *, ...):
    err_sq = (df[model_col] - df[obs_col]) ** 2
    grouped = (
        df.assign(_sq=err_sq)
        .groupby(list(index_cols) + [columns_col], as_index=False)
        ._sq.agg(rmsd=lambda v: np.sqrt(v.mean()), count="size")
    )
    pivot = grouped.pivot_table(values="rmsd", index=index_cols,
                                 columns=columns_col, aggfunc="first")
    pivot.index = pivot.index.map(lambda key: _format_label(key, label_format, registry))
    return pivot
```

`_format_label` looks up `display_name`, `units`, `standard_name` in
the `Variable` registry; uses the raw column value as a fallback.

### 4.6 Layer-1 Operators

```python
# src/xrtoolz/interpolate/operators.py
class InterpGridToProfiles(Operator):
    """Operator wrapping interp_grid_to_profiles.

    Observations captured at construction; model flows through __call__.
    """
    def __init__(self, *,
                 observations: xr.Dataset | pd.DataFrame,
                 vertical_method: str = "linear",
                 horizontal_method: str = "linear",
                 lon: str = "longitude", lat: str = "latitude",
                 depth: str = "depth",
                 observation_dim: str = "observation"): ...

# src/xrtoolz/metrics/operators.py
class RMSDScoreboard(Operator):
    """Operator wrapping rmsd_scoreboard."""
    def __init__(self, *,
                 model_col="model_value", obs_col="observation_value",
                 index_cols=("variable", "depth_bin"),
                 columns_col="lead_day",
                 label_format=...): ...
```

Tier A array kernels are not Operator-promoted — they're
shape-specific numpy functions, not Dataset-shaped.

### 4.7 Recipe page (not a baked-in driver)

`docs/recipes/class4_validation.md` walks through assembling the
Mercator pipeline:

```python
# 1. Standardize forecast → CF names (OB-1.2)
ds = rename_from_cf_standard_names(ds_forecast)

# 2. SSH → SLA via MDT (issue #135)
ds["ssh"] = ds["ssh"] - ds_mdt["mdt"]

# 3. For each variable, interpolate model onto observations:
def validate_variable(ds, df_obs, variable, *, vertical_method):
    df_obs = df_obs.query(f"variable == '{variable}'").copy()
    df_obs["model_value"] = interp_grid_to_profiles(
        ds[variable], df_obs, vertical_method=vertical_method,
    )
    return df_obs

# 4. Variable-specific dispatch (Mercator convention)
SHARP = {"thetao", "uo", "vo", "ssh"}
results = pd.concat([
    validate_variable(ds, df_obs, v,
                      vertical_method="bracket" if v in SHARP else "spline")
    for v in ds.data_vars
])

# 5. Depth-bin assignment + scoreboard
results["depth_bin"] = assign_depth_bins(results["depth"])
scoreboard = rmsd_scoreboard(results,
                             index_cols=("variable", "depth_bin"),
                             columns_col="lead_day")
```

Compositional, transparent, easy to tweak per use case. No hidden
forecast-cube assumptions.

## 5. Library leverage

| Need | Library |
|---|---|
| Horizontal grid→points interp | `xr.Dataset.interp` advanced indexing (already in ODC-1.2) |
| Vertical bracket interp | `numpy.searchsorted` |
| Vertical cubic spline | `scipy.interpolate.CubicSpline` |
| NaN-pattern grouping | numpy bitmask via `int64` powers of 2 |
| Depth-bin categorical | `pandas.cut` (or `numpy.digitize`) |
| RMSD pivot table | `pandas.DataFrame.groupby` + `pivot_table` |
| Variable / depth labels | `xrtoolz.types.Variable` registry |

No new top-level deps. scipy + pandas + numpy already in.

## 6. Public API surface

```python
# Tier A — array kernels
xrtoolz.interpolate.array.interp_vertical_bracket(profiles, depths, targets)
xrtoolz.interpolate.array.interp_vertical_spline(profiles, depths, targets,
                                                   *, bc_type="natural")

# Tier B — depth-aware grid→profile
xrtoolz.interpolate.interp_grid_to_profiles(
    model_data, observations, *,
    vertical_method="linear", horizontal_method="linear",
    lon, lat, depth, observation_dim,
)

# Depth-bin assignment
xrtoolz.interpolate.assign_depth_bins(depth_values, bins=DEPTH_BINS_DEFAULT)
xrtoolz.interpolate.DEPTH_BINS_DEFAULT

# Scoreboard
xrtoolz.metrics.rmsd_scoreboard(df, *, ...)

# Operators
xrtoolz.interpolate.InterpGridToProfiles(...)
xrtoolz.metrics.RMSDScoreboard(...)
```

## 7. Tests

| Test | Asserts |
|---|---|
| `interp_vertical_bracket` known sparse profile | exact at boundary, mid-bracket interpolated correctly |
| `interp_vertical_bracket` target outside range | NaN |
| `interp_vertical_bracket` identity at exact depth match | exact |
| `interp_vertical_bracket` NaN propagation | bracket with NaN → NaN result |
| `interp_vertical_spline` smooth analytic profile | reproduces analytic within tol |
| `interp_vertical_spline` two distinct NaN patterns | each group correctly handled |
| `interp_vertical_spline` 65+ depths | informative `ValueError` with hint to use bracket |
| `interp_grid_to_profiles` 2-D model | matches `sample_at_points` (no depth dim) |
| `interp_grid_to_profiles` 3-D model | analytic field reproduced |
| `interp_grid_to_profiles` all three vertical_method values | distinct results, all finite |
| `assign_depth_bins` standard cases | correct labels |
| `assign_depth_bins` outside bins | None |
| `assign_depth_bins` custom bins | custom labels |
| `rmsd_scoreboard` end-to-end | pivot shape correct; RMSD per cell matches `sqrt(mean(err²))` |
| `rmsd_scoreboard` row labels via `Variable` registry | Mercator-style label formatting |
| `rmsd_scoreboard` unregistered variable | falls back to raw column value |
| `InterpGridToProfiles` Operator round-trip | reconstructed produces identical output |
| `RMSDScoreboard` Operator round-trip | identical scoreboard |

Target: ~18 cases.

## 8. Out of scope

- **`Class4Validation` driver** — Mercator-specific forecast-cube
  schema baked in; ship as a recipe instead.
- **Forecast-cube schema** (`first_day_datetime`, `lead_day_index`)
  — out of library scope; user concern.
- **MDT-aware SSH→SLA conversion** — covered by issue #135.
- **In-situ observation file parsers** (Argo, drifter, Class IV `.nc`)
  — V3.5 (#54) covers drifter ingest separately. Argo is a follow-up.
- **Lead-day label formatter** (`"Lead day 1"`, ...) — trivially
  inlinable in the recipe; ship as `lead_day_labels` only if a use
  case demands it.
- **Variable-specific vertical-interp dispatch** (`bracket` for u/v/T,
  `spline` for the rest) — Mercator's choice; expose as a parameter,
  document in the recipe.
- **Depth-bin convention enforcement** — bins are user-tunable;
  Mercator's `0-5/5-100/100-300/300-600m` is the default.
- **Non-monotone model depths** — assume sorted ascending;
  validate-with-error rather than auto-sort.

## 9. Effort

≈195 LOC implementation + ≈150 LOC tests + ≈25 LOC recipe doc.
Largest oceanbench item so far. Single PR.

| Slice | LOC |
|---|---|
| `interp_vertical_bracket` (Tier A) | 30 |
| `interp_vertical_spline` (NaN-bitmask grouping, Tier A) | 50 |
| `interp_grid_to_profiles` (Tier B) | 60 |
| `assign_depth_bins` + `DEPTH_BINS_DEFAULT` | 15 |
| `rmsd_scoreboard` + label formatter | 40 |
| `InterpGridToProfiles`, `RMSDScoreboard` operators | 30 |
| Tests | ~150 |
| Recipe doc + re-exports | 25 |

## 10. Risks / open questions

1. **Where Tier A vertical kernels live.** New
   `interpolate/_src/array_vertical.py`. Sibling of the existing
   `array_smooth.py`, `array_coord_remap.py` — matches the
   "tier A array kernels per concern" pattern.
2. **`interp_grid_to_profiles` location.** New
   `interpolate/_src/grid_to_profile.py`. Sibling of
   `grid_to_points.py` (ODC-1.2). The two could share a module if
   we'd rather; chosen separate because depth-aware logic is a
   distinct responsibility.
3. **64-depth cap on `interp_vertical_spline`.** Inherited from
   Mercator's `int64` bitmask trick. Document; raise informative
   error suggesting the `bracket` method or pre-reducing model
   depths if the user has more.
4. **`vertical_method="linear"` default.** Least surprising,
   delegates to `xr.interp` which uses scipy's `interp1d`. Mercator's
   `bracket` and `spline` are opt-in — power-user choices.
5. **Bracket vs cubic-spline accuracy trade-off.** Bracket avoids
   spline overshoots near sharp gradients (typical of u/v/T at the
   thermocline); spline gives smoother fits where the field is well-
   resolved. Document the trade-off; let the recipe show Mercator's
   variable-specific dispatch.
6. **`rmsd_scoreboard` output shape.** `index_cols` and `columns_col`
   are configurable, defaulting to Mercator's `(var × depth_bin) ×
   lead_day`. Single function, multiple use cases.
7. **Pandas inputs to Operators.** `RMSDScoreboard` takes pandas
   long-form. Unusual for xrtoolz Operators (normally Dataset →
   Dataset). Acceptable here because validation tables are inherently
   tabular; document the deviation.
8. **`Variable` registry coverage for label formatter.** Already
   covers standard ocean / atmospheric vars. Missing entries fall
   back to the raw column value with a logging warning (one-time
   per missing variable).
9. **Operator promotion of Tier A kernels.** Skipped — kernels are
   numpy-shape-specific; the Tier B `interp_grid_to_profiles`
   Operator is the right level of abstraction.
