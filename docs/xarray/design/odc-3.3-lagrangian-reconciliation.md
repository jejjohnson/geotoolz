# ODC-3.3 — Lagrangian advection: reconciliation with Epic V3

**Source survey item:** [ocean-data-challenges-survey.md §3.3 — `mod_traj.py`](ocean-data-challenges-survey.md)
**Status:** reconciled — no new design doc; amendments applied to existing V3 issues.
**Maps to upstream:** `src/mod_traj.py` from `2024c_DC_4DMedSea-ESA`.

---

## TL;DR

The pre-existing **Epic V3 (#49)** already designs the entire Lagrangian
sub-domain in xrtoolz, with six well-decomposed sub-issues. The
ocean-data-challenges `mod_traj.py` upstream maps almost exactly onto
that epic. Rather than open a parallel ODC-3.2 issue, we amend the V3
sub-issues with three small additions drawn from the upstream:

1. **scipy backend** for `AdvectParticles` (#51).
2. **`EndpointErrorMap`** as a sixth metric in `metrics.lagrangian` (#53).
3. **`from_cmems_drifter` reference**: the upstream `prepare_drifter_data`
   is the canonical reformatter for the CMEMS preset (#54).

This document records the mapping for future reference.

## V3 epic — at a glance

| Issue | Title | Status |
|---|---|---|
| [#49](https://github.com/jejjohnson/xrtoolz/issues/49) | Epic V3: Lagrangian — particle advection, transport diagnostics, trajectory metrics | OPEN |
| [#50](https://github.com/jejjohnson/xrtoolz/issues/50) | V3.1: Trajectory schema + `SeedParticles` operator | OPEN |
| [#51](https://github.com/jejjohnson/xrtoolz/issues/51) | V3.2: `AdvectParticles` operator (scipy backend) | OPEN — amended |
| [#52](https://github.com/jejjohnson/xrtoolz/issues/52) | V3.3: Transport diagnostics — `PairDispersion`, `ResidenceTime`, `ConnectivityMatrix`, `FTLE` | OPEN |
| [#53](https://github.com/jejjohnson/xrtoolz/issues/53) | V3.4: Trajectory metrics — `TrajectoryRMSE`, `EndpointError`, `EndpointErrorMap`, `DispersionError`, `ResidenceTimeError`, `ConnectivityError` | OPEN — amended |
| [#54](https://github.com/jejjohnson/xrtoolz/issues/54) | V3.5: Drifter ingestion adapter | OPEN — amended |
| [#55](https://github.com/jejjohnson/xrtoolz/issues/55) | V3.6: Demo notebook — twin advection on Gulf Stream | OPEN |

## Upstream → V3 mapping

| Upstream `mod_traj.py` symbol | V3 issue | Notes |
|---|---|---|
| `prepare_drifter_data(ds_drifters, maps)` | #54 V3.5 | Canonical CMEMS in-situ TAC reformatter; reference for `from_cmems_drifter`. |
| `adv_eul(t, lon, lat, fu, fv, dt)` | #51 V3.2 | One-step Euler kernel; covered by `method="euler"`. |
| `adv_rk4(t, lon, lat, fu, fv, dt)` | #51 V3.2 | One-step RK4 kernel; covered by `method="rk4"` (default). |
| `compute_traj(...)` | #51 V3.2 | Driver — multi-particle multi-step orchestration plus file I/O; the algorithm folds into `advect_particles`, the I/O is out of scope. |
| `dist_drifters(lat1, lon1, lat2, lon2)` | #53 V3.4 | Haversine distance; used inside `EndpointError` and `EndpointErrorMap`. |
| `compute_deviation(...)` | #53 V3.4 | The binned `(horizon, lat, lon)` deviation map; the algorithm is the new `EndpointErrorMap` metric. |
| `plot_traj_deviation_maps`, `plot_meantraj_deviation` | (covered by existing viz) | `FacetPanel(SpatialMapPanel(...), facet_dim="horizon")` from ODC-2.3 + `bin_residuals_2d` from ODC-1.4 reproduces these without a new panel. |

## Decisions already locked by V3 (do not redo)

The following were settled in #49–#55 and should not be relitigated:

- **Module placement**: `xrtoolz.lagrangian` (top-level), not under `geo` or `ocn`.
- **Schema** (V3.1, #50): dims `(particle, time)`; vars `lon(particle, time)`, `lat(particle, time)`; coord `particle_id(particle)` int64; attrs `featureType="trajectory"` (CF).
- **Validator helper**: `validate_trajectory(ds)` in `xrtoolz.lagrangian`.
- **No `parcels` dep**: own implementation per the epic's acceptance criteria.
- **2-D surface only** for v1; 3-D advection deferred.
- **Operator pattern**: every Layer-0 function gets a Layer-1 Operator wrapper, same as everywhere else in xrtoolz.
- **Metrics live in `xrtoolz.metrics.lagrangian`** (per D12), not in the physics module.
- **Drifter ingest in `xrtoolz.lagrangian.io`**: `from_gdp`, `from_cmems_drifter`.

## Amendments applied (this iteration)

### #51 V3.2 — scipy backend + position-update mode

Added to the `method` enum: `"adaptive"` → wraps
`scipy.integrate.solve_ivp(method="RK45", ...)` (Dormand–Prince embedded
4(5) pair). Output cadence controlled by `dt` via `t_eval`; internal
step adaptive with default `rtol=1e-3, atol=1e-6`.

Default remains `method="rk4"` (fixed-step, ~15 LOC numpy) for byte-comparable
behaviour against ocean-data-challenges fixtures.

Also added: `position: Literal["flat", "geodesic"] = "flat"`. `"flat"`
matches the upstream `dx / (1852·60·cos(lat))` convention; `"geodesic"`
uses `pyproj.Geod.fwd` for high-latitude / long-horizon accuracy.
`pyproj` is already a top-level dep.

The shared RHS is the same across all four methods:

```python
def rhs(t, y):
    lon, lat = y[:N], y[N:]
    u, v = sample_velocity_at(ds, t, lon, lat)
    dlon_dt, dlat_dt = _to_lonlat_rate(u, v, lat, position)
    return np.concatenate([dlon_dt, dlat_dt])
```

### #53 V3.4 — `EndpointErrorMap` added as the sixth metric

Geographic `(horizon, lat_bin, lon_bin)` map of paired-track distances.
The canonical drifter-eval visualisation. Sits between scalar
`EndpointError` and a generic `bin_residuals_2d` reduction.

Implementation:

1. Per `(particle, time)`: paired great-circle distance between
   `traj_pred` and `traj_ref`.
2. Time-slice by `horizons` (sequence of `pd.Timedelta`).
3. `scipy.stats.binned_statistic_2d` with statistic-by-name dispatch
   (`"mean"`, `"median"`, `"rmse"`).

Reuses ODC-1.4's `bin_residuals_2d` machinery.

### #54 V3.5 — CMEMS reference

Annotated `from_cmems_drifter` with the upstream
`prepare_drifter_data` reference: isolate a single depth, strip QC
variables, rename `LATITUDE`/`LONGITUDE`/`TIME` to lower-case, tag each
row with `platform_code` as `particle_id` (replacing the upstream's
`sensor_id`).

## Out of scope (also per V3)

- 3-D advection with vertical velocity `w`.
- JAX / numba acceleration.
- Backward-in-time integration (`direction="backward"`).
- Stochastic Lagrangian (Brownian sub-grid diffusion).
- Adaptive seeding (e.g. Lagrangian coherent structures).
